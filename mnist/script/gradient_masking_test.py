import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pgd_attack import *
import os
import numpy as np
from pgd_attack import LinfPGDAttack
slim = tf.contrib.slim
import matplotlib.pyplot as plt
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Model_crop():
    def __init__(self, x, y, idx=None):
        self.x_input = x
        self.y_input = y

        self.x_voting = []
        self.x_crop = []
        loc = np.arange(10,18,3, dtype='int64')
        loc = [(i, j) for i in loc for j in loc]
        if idx!=None:
            loc = loc[idx:idx+1]
        self.xent = []
        with tf.variable_scope('classifier') as scope:
            self.y_pred = []
            for i, loc_i in enumerate(loc):
                # crop
                loc_x, loc_y = loc_i
                x_crop_i = self.x_input[:, loc_x-10:loc_x+10, loc_y-10:loc_y+10, :]
                self.x_crop += [x_crop_i]
                # x = slim.max_pool2d(x, kernel_size=2)
                x = slim.conv2d(x_crop_i, kernel_size=5, num_outputs=32, scope='conv1')
                x = slim.max_pool2d(x, kernel_size=2)
                x = slim.conv2d(x, kernel_size=5, num_outputs=64, scope='conv2')
                x = slim.max_pool2d(x, kernel_size=2)
                x = slim.flatten(x, scope='flatten')
                x = slim.fully_connected(x, num_outputs=1024, scope='fc1')
                pre_softmax = x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')
                self.y_pred += [tf.argmax(pre_softmax, 1)]
                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=pre_softmax)
                self.xent += [y_xent]
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse == True
            self.y_pred = tf.stack(self.y_pred)
            self.xent_indv = self.xent
            self.xent = tf.reduce_mean(self.xent)
            self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')

class Model_crop_insftmx():#
    def __init__(self, x, y, theta_ratio):
        self.x_input = x
        self.y_input = y
        self.theta_ratio = theta_ratio

        self.x_voting = []
        self.x_crop = []
        loc = np.arange(10,18,1, dtype='int64')
        loc = [(i, j) for i in loc for j in loc]
        self.softmax = []
        with tf.variable_scope('classifier') as scope:
            for i, loc_i in enumerate(loc):
                # crop
                loc_x, loc_y = loc_i
                x_crop_i = self.x_input[:, loc_x-10:loc_x+10, loc_y-10:loc_y+10, :]
                self.x_crop += [x_crop_i]
                # x = slim.max_pool2d(x, kernel_size=2)
                x = slim.conv2d(x_crop_i, kernel_size=5, num_outputs=32, scope='conv1')
                x = slim.max_pool2d(x, kernel_size=2)
                x = slim.conv2d(x, kernel_size=5, num_outputs=64, scope='conv2')
                x = slim.max_pool2d(x, kernel_size=2)
                x = slim.flatten(x, scope='flatten')
                x = slim.fully_connected(x, num_outputs=1024, scope='fc1')
                x = slim.fully_connected(self.theta_ratio*x, num_outputs=10, activation_fn=tf.nn.softmax, scope='fc2')
                self.softmax += [x]
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse == True
            self.mean_softmax = tf.reduce_mean(self.softmax, 0)
            # reference: https://github.com/tensorflow/tensorflow/issues/2462
            epsilon = 1e-11
            self.mean_softmax = tf.clip_by_value(self.mean_softmax, epsilon, 1 - epsilon)
            self.y_pred = tf.argmax(self.softmax,2)
            self.pre_mean_softmax = tf.log(self.mean_softmax)
            self.y_pred_voted = tf.argmax(self.mean_softmax,1)
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_mean_softmax)
            self.xent_indv = tf.expand_dims(y_xent,0)
            self.xent = tf.reduce_sum(y_xent)
            self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')

def get_model():
    img_size = 28
    batch_size = 32
    input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64,shape=(batch_size))

    ## MODEL to be attacked ##
    model = Model_crop_insftmx(input_images, input_label, theta_ratio=10.)

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
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_batch_train, y_batch_train = mnist.train.next_batch(batch_size)
    x_batch_train = x_batch_train.reshape(batch_size, img_size, img_size, 1)
    nat_dict_train = {input_images: x_batch_train,
                      input_label: y_batch_train}
    return sess, input_images, input_label, nat_dict_train, model

def get_grads(sess, input_images, input_label, nat_dict_train):
    # ckpt_path = '../ckpt/crop64_20_nat_itr35k/bb_64crop_ckpt'
    ckpt_path = '../ckpt/crop9_20_insftmx_itr170k/insftmx_crop_ckpt'

    ## MODEL1 ##
    tf.get_variable_scope().reuse_variables()
    model1 = Model_crop_insftmx(input_images, input_label, theta_ratio=0.01)
    attack1 = LinfPGDAttack(model1, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
    saver1 = tf.train.Saver()
    saver1.restore(sess, ckpt_path)
    grad1 = sess.run(attack1.grad, feed_dict=nat_dict_train)


    ## MODEL2 ##
    tf.get_variable_scope().reuse_variables()
    model2 = Model_crop_insftmx(input_images, input_label, theta_ratio=0.1)
    attack2 = LinfPGDAttack(model2, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
    saver2 = tf.train.Saver()
    saver2.restore(sess, ckpt_path)
    grad2 = sess.run(attack2.grad, feed_dict=nat_dict_train)

    ## MODEL3 ##
    tf.get_variable_scope().reuse_variables()
    model3 = Model_crop_insftmx(input_images, input_label, theta_ratio=1.)
    attack3 = LinfPGDAttack(model3, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
    saver3 = tf.train.Saver()
    saver3.restore(sess, ckpt_path)
    grad3 = sess.run(attack3.grad, feed_dict=nat_dict_train)

    ## MODEL4 ##
    tf.get_variable_scope().reuse_variables()
    model4 = Model_crop_insftmx(input_images, input_label, theta_ratio=10.)
    attack4 = LinfPGDAttack(model4, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
    saver4 = tf.train.Saver()
    saver4.restore(sess, ckpt_path)
    grad4 = sess.run(attack4.grad, feed_dict=nat_dict_train)

    if 1:
        import seaborn as sns
        plt.figure()
        plt.subplot(2,2,1)
        sns.distplot([np.mean(np.abs(grad1[i])) for i in range(32)], label='grad1')
        plt.legend()
        plt.subplot(2,2,2)
        sns.distplot([np.mean(np.abs(grad2[i])) for i in range(32)], label='grad2')
        plt.legend()
        plt.subplot(2,2,3)
        sns.distplot([np.mean(np.abs(grad3[i])) for i in range(32)], label='grad3')
        plt.legend()
        plt.subplot(2,2,4)
        sns.distplot([np.mean(np.abs(grad4[i])) for i in range(32)], label='grad4')
        plt.legend()
        plt.show()

    return grad1, grad2, grad3, grad4

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
        epsilon = 0.3 * i / n_samples
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images] + epsilon * np.sign(grad1), 0, 1),
                          input_label: nat_dict_train[input_label] }
        loss_i, pred_voted_i, pred_i = sess.run([model.xent_indv,model.y_pred_voted,model.y_pred], feed_dict=adv_dict_train)
        loss1 += [loss_i]
        predvoted1 += [pred_voted_i]
        pred1 += [pred_i]
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images]  + epsilon * np.sign(grad2), 0, 1),
                          input_label: nat_dict_train[input_label]}
        loss_i, pred_voted_i, pred_i = sess.run([model.xent_indv,model.y_pred_voted,model.y_pred], feed_dict=adv_dict_train)
        loss2 += [loss_i]
        predvoted2 += [pred_voted_i]
        pred2 += [pred_i]
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images]  + epsilon * np.sign(grad3), 0, 1),
                          input_label: nat_dict_train[input_label]}
        loss_i, pred_voted_i, pred_i = sess.run([model.xent_indv,model.y_pred_voted,model.y_pred], feed_dict=adv_dict_train)
        loss3 += [loss_i]
        predvoted3 += [pred_voted_i]
        pred3 += [pred_i]
        adv_dict_train = {input_images: np.clip(nat_dict_train[input_images]  + epsilon * np.sign(grad4), 0, 1),
                          input_label: nat_dict_train[input_label]}
        loss_i, pred_voted_i, pred_i = sess.run([model.xent_indv,model.y_pred_voted,model.y_pred], feed_dict=adv_dict_train)
        loss4 += [loss_i]
        predvoted4 += [pred_voted_i]
        pred4 += [pred_i]

    loss1 = np.squeeze(np.asarray(loss1)).transpose((1,0))
    loss2 = np.squeeze(np.asarray(loss2)).transpose((1,0))
    loss3 = np.squeeze(np.asarray(loss3)).transpose((1,0))
    loss4 = np.squeeze(np.asarray(loss4)).transpose((1,0))

    label = np.tile(np.expand_dims(nat_dict_train[input_label], -1), (1, n_samples))
    correct1 = np.squeeze(np.asarray(predvoted1)).transpose((1,0)) == label
    correct2 = np.squeeze(np.asarray(predvoted2)).transpose((1,0)) == label
    correct3 = np.squeeze(np.asarray(predvoted3)).transpose((1,0)) == label
    correct4 = np.squeeze(np.asarray(predvoted4)).transpose((1,0)) == label
    return loss1, loss2, loss3, loss4, correct1, correct2, correct3, correct4
#
# def get_loss_bim():
#     img_size = 28
#     batch_size = 32
#     input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 1))
#     input_label = tf.placeholder(tf.int64,shape=(batch_size))
#     theta_ratio = tf.placeholder(tf.int64,shape=())
#
#     ## MODEL1 ##
#     model = Model_crop_insftmx(input_images, input_label, theta_ratio=theta_ratio)
#     attack = LinfPGDAttack(model, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
#
#     ## training starts ###
#     FLAGS = tf.app.flags.FLAGS
#     tfconfig = tf.ConfigProto(
#         allow_soft_placement=True,
#         log_device_placement=True,
#     )
#     tfconfig.gpu_options.allow_growth = True
#     sess = tf.Session(config=tfconfig)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     saver = tf.train.Saver()
#     saver.restore(sess, '../ckpt/crop64_20_nat_itr35k/bb_64crop_ckpt')
#
#     mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#     x_batch_train, y_batch_train = mnist.train.next_batch(batch_size)
#     x_batch_train = x_batch_train.reshape(batch_size, img_size, img_size, 1)
#
#     theta1_dict = {input_images: x_batch_train,
#                    input_label: y_batch_train,
#                    theta_ratio: 1.0}
#     grad = sess.run(attack.grad, feed_dict=theta1_dict)
#
#
#     ## MODEL2 ##
#     tf.get_variable_scope().reuse_variables()
#     model2 = Model_crop_insftmx(input_images, input_label, theta_ratio=0.1)
#     attack2 = LinfPGDAttack(model2, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
#     saver2 = tf.train.Saver()
#     saver2.restore(sess, '../ckpt/crop64_20_nat_itr35k/bb_64crop_ckpt')
#     grad2 = sess.run(attack2.grad, feed_dict=nat_dict_train)
#
#     ## MODEL3 ##
#     tf.get_variable_scope().reuse_variables()
#     model3 = Model_crop_insftmx(input_images, input_label, theta_ratio=0.3)
#     attack3 = LinfPGDAttack(model3, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
#     saver3 = tf.train.Saver()
#     saver3.restore(sess, '../ckpt/crop64_20_nat_itr35k/bb_64crop_ckpt')
#     grad3 = sess.run(attack3.grad, feed_dict=nat_dict_train)
#
#     ## MODEL4 ##
#     tf.get_variable_scope().reuse_variables()
#     model4 = Model_crop_insftmx(input_images, input_label, theta_ratio=1.)
#     attack4 = LinfPGDAttack(model4, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
#     saver4 = tf.train.Saver()
#     saver4.restore(sess, '../ckpt/crop64_20_nat_itr35k/bb_64crop_ckpt')
#     grad4 = sess.run(attack3.grad, feed_dict=nat_dict_train)
#
#     loss1 = []
#     loss2 = []
#     loss3 = []
#     loss4 = []
#     for i in range(40):
#         epsilon = 0.3 * i / 40.
#         adv_dict_train = {input_images: np.clip(x_batch_train + epsilon * grad1, 0, 1),
#                           input_label: y_batch_train}
#         loss_i = sess.run(model2.xent_indv, feed_dict=adv_dict_train)
#         loss1 += [loss_i]
#         adv_dict_train = {input_images: np.clip(x_batch_train + epsilon * grad2, 0, 1),
#                           input_label: y_batch_train}
#         loss_i = sess.run(model2.xent_indv, feed_dict=adv_dict_train)
#         loss2 += [loss_i]
#         adv_dict_train = {input_images: np.clip(x_batch_train + epsilon * grad3, 0, 1),
#                           input_label: y_batch_train}
#         loss_i = sess.run(model2.xent_indv, feed_dict=adv_dict_train)
#         loss3 += [loss_i]
#         adv_dict_train = {input_images: np.clip(x_batch_train + epsilon * grad4, 0, 1),
#                           input_label: y_batch_train}
#         loss_i = sess.run(model2.xent_indv, feed_dict=adv_dict_train)
#         loss4 += [loss_i]
#
#     loss1 = np.squeeze(np.asarray(loss1)).transpose((1,0))
#     loss2 = np.squeeze(np.asarray(loss2)).transpose((1,0))
#     loss3 = np.squeeze(np.asarray(loss3)).transpose((1,0))
#     loss4 = np.squeeze(np.asarray(loss4)).transpose((1,0))
#     return loss1, loss2, loss3, loss4

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
    img3 = ax3.imshow(np.squeeze(grad1), cmap='seismic')
    fig.colorbar(img3)
    plt.grid('off')
    ax4 = fig.add_subplot(3,2,4)
    img4 = ax4.imshow(np.squeeze(grad2), cmap='seismic')
    fig.colorbar(img4)
    plt.grid('off')
    ax5 = fig.add_subplot(3,2,5)
    img5 = ax5.imshow(np.squeeze(grad3), cmap='seismic')
    fig.colorbar(img5)
    plt.grid('off')
    ax6 = fig.add_subplot(3,2,6)
    img6 = ax6.imshow(np.squeeze(grad4), cmap='seismic')
    fig.colorbar(img6)
    plt.grid('off')
    return fig

if __name__ == '__main__':
    # single step
    sess, input_images, input_label, nat_dict_train, model = get_model()
    grad1, grad2, grad3, grad4 = get_grads(sess, input_images, input_label, nat_dict_train)
    loss1, loss2 ,loss3 ,loss4, pred1, pred2, pred3, pred4 = get_loss(sess, model, input_images, input_label, nat_dict_train)


    # iterative
    # loss1, loss2 ,loss3 ,loss4 = get_loss_bim()
    for i in range(30):
        fig = plot_loss(grad1[i],grad2[i],grad3[i],grad4[i], loss1[i],loss2[i],loss3[i],loss4[i],pred1[i],pred2[i],pred3[i],pred4[i])
        plt.show()
    print('done')

    # plt.close('all')
    # x = np.arange(0,40,1)/40.*0.3
    # idx += 1
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(x, loss1[idx])
    # ax1.set_ylabel('y1')
    #ls

    # ax2 = ax1.twinx()
    # ax2.plot(x, loss2[idx], 'r-')
    # ax2.set_ylabel('y2', color='r')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')
    #
