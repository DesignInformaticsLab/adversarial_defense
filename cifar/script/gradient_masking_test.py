# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pgd_attack import *
import os
import numpy as np
from pgd_attack import LinfPGDAttack
slim = tf.contrib.slim
import matplotlib.pyplot as plt
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
import tensorflow as tf
from multiGPU_utils import *
import json

with open('config.json') as config_file:
    config = json.load(config_file)


class Model_crop(object):
    """ResNet model."""

    def __init__(self, config, mode, input_images, input_label, theta_ratio=10.):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.mode = mode
        self.x_input = input_images
        self.y_input = input_label
        self.config = config
        self.y_pred = []
        self.loc = [[14, 14], [14, 18], [18, 14], [18, 18]]

        # Setting up the optimizer
        step_size_schedule = config['step_size_schedule']
        weight_decay = config['weight_decay']
        momentum = config['momentum']
        boundaries = [int(sss[0]) for sss in step_size_schedule]
        boundaries = boundaries[1:]
        values = [sss[1] for sss in step_size_schedule]
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.piecewise_constant(tf.cast(self.global_step, tf.int32), boundaries, values)
        self.opts = tf.train.MomentumOptimizer(learning_rate, momentum)

        xent = []
        prediction = []
        tower_grads = []
        adv_grad = []

        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for ii in xrange(len(self.loc)):
                gpu_i = ii if mode == 'train' else 0
                with tf.device('/gpu:%d' % gpu_i):
                    loc_x, loc_y = self.loc[ii]
                    x_crop_i = self.x_input[:, loc_x - 14:loc_x + 14, loc_y - 14:loc_y + 14, :]
                    pre_softmax = self._build_model(x_crop_i)

                    # reuse variables
                    tf.get_variable_scope().reuse_variables()
                    crop_prediction = tf.argmax(pre_softmax, 1)
                    crop_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_softmax, labels=self.y_input)

                    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=vscope)
                    total_loss = tf.reduce_mean(crop_xent) + weight_decay * tf.reduce_mean(self._decay())
                    # training gradient per mini-batch
                    grad_i = self.opts.compute_gradients(total_loss)
                    # summing up over crops
                    xent += [crop_xent]
                    tower_grads += [grad_i]
                    prediction += [crop_prediction]

                    # adversarial gradient per mini-batch
                    adv_grad_i = tf.gradients(crop_xent, self.x_input)[0]
                    adv_grad += [adv_grad_i]

        self.xent = tf.stack(xent, 1)
        self.mean_xent = tf.reduce_mean(self.xent)
        self.prediction = tf.stack(prediction, 1)
        update_batchnorm_op = tf.group(*batchnorm_updates)

        self.grads = average_gradients(tower_grads)
        update_network_op = self.opts.apply_gradients(self.grads, global_step=self.global_step)
        self.train_step = tf.group(update_network_op, update_batchnorm_op)

        self.adv_grad = tf.reduce_mean(adv_grad, 0)
        self.voted_pred = []
        batch_size = config['training_batch_size'] if mode == "train" else config['eval_batch_size']
        for i in range(batch_size):  # loop over a batch
            y, idx, count = tf.unique_with_counts(self.prediction[i])
            majority = tf.argmax(count)
            self.voted_pred += [tf.gather(y, majority)]
        self.voted_pred = tf.stack(self.voted_pred)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.voted_pred, self.y_input), tf.float32))

        self.vars = tf.trainable_variables()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self, x_input):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input'):

            input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                           x_input)
            x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        filters = [16, 160, 320, 640]

        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            pre_softmax = self._fully_connected(x, 10)

        return pre_softmax

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                inputs=x,
                decay=.9,
                center=True,
                scale=True,
                activation_fn=None,
                updates_collections=None,
                is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])


def get_model(theta_ratio):
    img_size = 32
    batch_size = config['eval_batch_size']
    input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 3))
    input_label = tf.placeholder(tf.int64,shape=(batch_size))

    ## MODEL to be attacked ##
    model = Model_crop(config, mode='eval', input_images=input_images, input_label=input_label, theta_ratio=theta_ratio)

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, '/home/hope-yao/Documents/adversarial_defense/cifar/ckpt/crop_4_20_adv/half_half/lr_config1_adv/checkpoint-25001')


    import cifar10_input
    if 1:
        bstart = 0
        bend = bstart+batch_size
        data_path = config['data_path']
        cifar = cifar10_input.CIFAR10Data(data_path)
        x_batch_eval = cifar.eval_data.xs[bstart:bend, :]
        x_batch_eval = np.asarray(x_batch_eval, 'float32') / 255.
        y_batch_eval = cifar.eval_data.ys[bstart:bend]
        nat_dict_eval = {input_images: x_batch_eval,
                          input_label: y_batch_eval}
    grad = sess.run(model.adv_grad, feed_dict=nat_dict_eval)
    return grad, sess, model, input_images, input_label, nat_dict_eval

def get_loss(sess, model, input_images, input_label, nat_dict_train):
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    pred1 = []
    pred2 = []
    pred3 = []
    pred4 = []
    predvoted1 = []
    predvoted2 = []
    predvoted3 = []
    predvoted4 = []
    n_samples = 40
    for i in range(n_samples):
        epsilon = 8./255. * i / n_samples
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images] + epsilon * np.sign(grad1), 0, 1),
                          input_label: nat_dict_train[input_label] }
        loss_i, pred_voted_i, pred_i = sess.run([model.xent,model.voted_pred,model.prediction], feed_dict=adv_dict_train)
        loss1 += [loss_i]
        predvoted1 += [pred_voted_i]
        pred1 += [pred_i]
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images]  + epsilon * np.sign(grad2), 0, 1),
                          input_label: nat_dict_train[input_label]}
        loss_i, pred_voted_i, pred_i = sess.run([model.xent,model.voted_pred,model.prediction], feed_dict=adv_dict_train)
        loss2 += [loss_i]
        predvoted2 += [pred_voted_i]
        pred2 += [pred_i]
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images]  + epsilon * np.sign(grad3), 0, 1),
                          input_label: nat_dict_train[input_label]}
        loss_i, pred_voted_i, pred_i = sess.run([model.xent,model.voted_pred,model.prediction], feed_dict=adv_dict_train)
        loss3 += [loss_i]
        predvoted3 += [pred_voted_i]
        pred3 += [pred_i]
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images]  + epsilon * np.sign(grad4), 0, 1),
                          input_label: nat_dict_train[input_label]}
        loss_i, pred_voted_i, pred_i = sess.run([model.xent,model.voted_pred,model.prediction], feed_dict=adv_dict_train)
        loss4 += [loss_i]
        predvoted4 += [pred_voted_i]
        pred4 += [pred_i]

    loss1 = np.mean(np.asarray(loss1),2).transpose((1,0))
    loss2 = np.mean(np.asarray(loss2),2).transpose((1,0))
    loss3 = np.mean(np.asarray(loss3),2).transpose((1,0))
    loss4 = np.mean(np.asarray(loss4),2).transpose((1,0))

    label = np.tile(np.expand_dims(nat_dict_train[input_label], -1), (1, n_samples))
    correct1 = np.squeeze(np.asarray(predvoted1)).transpose((1,0)) == label
    correct2 = np.squeeze(np.asarray(predvoted2)).transpose((1,0)) == label
    correct3 = np.squeeze(np.asarray(predvoted3)).transpose((1,0)) == label
    correct4 = np.squeeze(np.asarray(predvoted4)).transpose((1,0)) == label
    return loss1, loss2, loss3, loss4, correct1, correct2, correct3, correct4

def plot_loss(grad1,grad2,grad3,grad4,loss1, loss2, loss3, loss4, pred1, pred2, pred3, pred4):
    fig = plt.figure(figsize=(8,8))
    x = 0.3 * np.arange(0,len(loss1),1) / len(loss1)
    ax1 = fig.add_subplot(3,2,1)
    ax1.plot(x, loss1, label='theta=0.01')
    ax1.plot(x, loss2, label='theta=0.1')
    ax1.plot(x, loss3, label='theta=1')
    ax1.plot(x, loss4, label='theta=10')
    plt.legend()
    plt.grid('on')
    ax2 = fig.add_subplot(3,2,2)
    ax2.plot(x, pred1, label='theta=0.01')
    ax2.plot(x, pred2, label='theta=0.1')
    ax2.plot(x, pred3, label='theta=1')
    ax2.plot(x, pred4, label='theta=10')
    plt.legend()
    plt.grid('on')
    ax3 = fig.add_subplot(3,2,3)
    grad1 = (grad1-np.min(grad1))/(np.max(grad1)-np.min(grad1))
    img3 = ax3.imshow(np.squeeze(grad1), cmap='seismic')
    fig.colorbar(img3)
    plt.grid('off')
    ax4 = fig.add_subplot(3,2,4)
    grad2 = (grad2-np.min(grad2))/(np.max(grad2)-np.min(grad2))
    img4 = ax4.imshow(np.squeeze(grad2), cmap='seismic')
    fig.colorbar(img4)
    plt.grid('off')
    ax5 = fig.add_subplot(3,2,5)
    grad3 = (grad3-np.min(grad3))/(np.max(grad3)-np.min(grad3))
    img5 = ax5.imshow(np.squeeze(grad3), cmap='seismic')
    fig.colorbar(img5)
    plt.grid('off')
    ax6 = fig.add_subplot(3,2,6)
    grad4 = (grad4-np.min(grad4))/(np.max(grad4)-np.min(grad4))
    img6 = ax6.imshow(np.squeeze(grad4), cmap='seismic')
    fig.colorbar(img6)
    plt.grid('off')
    return fig

if __name__ == '__main__':
    if 0:
        import argparse
        parser = argparse.ArgumentParser()
        # settings for system
        parser.add_argument('--theta', type=float, help='theta [default: 1.0]')
        cfg = parser.parse_args()
        print('running theta = {}'.format(cfg.theta))

        # single step
        grad, sess, model, input_images, input_label, nat_dict_train = get_model(cfg.theta)
        np.save('theta_{}'.format(cfg.theta), grad)

    else:
        grad1 = np.load('theta_10.0.npy')
        grad2 = np.load('theta_1.0.npy')
        grad3 = np.load('theta_0.1.npy')
        grad4 = np.load('theta_0.01.npy')
        grad, sess, model, input_images, input_label, nat_dict_train = get_model(theta_ratio=1.0)
        loss1, loss2, loss3, loss4, pred1, pred2, pred3, pred4 = get_loss(sess, model, input_images, input_label, nat_dict_train)
        for i in range(20):
            plot_loss(grad1[i], grad2[i], grad3[i], grad4[i], loss1[i], loss2[i], loss3[i], loss4[i], pred1[i], pred2[i],
                  pred3[i], pred4[i])


