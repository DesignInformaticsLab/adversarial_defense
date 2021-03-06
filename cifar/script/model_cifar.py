# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from multiGPU_utils import *

class Model(object):
  """ResNet model."""

  def __init__(self, config, mode):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    self.y_input = tf.placeholder(tf.int64, shape=None)
    self.config = config
    self.y_pred = []
    # self.loc = [[14, 14], [14, 18], [18, 14], [18, 18]]
    # self.loc = [[12, 12], [12, 20], [20, 12], [20, 20]]
    # self.loc = [[12, 12], [12, 16], [12, 20], [16, 12], [16, 20], [20, 12], [20, 16], [20, 20]]
    self.loc = [[14, 14], [14, 16], [14, 18], [16, 14], [16, 18], [18, 14], [18, 16], [18, 18]]
    # self.loc = [[14, 14]]

    # Setting up the optimizer
    step_size_schedule = config['step_size_schedule']
    weight_decay = config['weight_decay']
    momentum = config['momentum']
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(tf.cast(self.global_step, tf.int32), boundaries, values)
    self.opts = tf.train.MomentumOptimizer(learning_rate,momentum)

    self.x_crop = []
    self.pre_softmax = []
    self.softmax = []
    self.adv_grads = []

    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for ii in xrange(len(self.loc)):
            gpu_i=ii if mode == 'train' else 0
            with tf.device('/gpu:%d' % gpu_i):#tf.device('/cpu'):#
                loc_x, loc_y = self.loc[ii]
                #x_crop_i = self.x_input[:, loc_x - 12:loc_x + 12, loc_y - 12:loc_y + 12, :]
                self.x_crop += [self.x_input[:, loc_x - 14:loc_x + 14, loc_y - 14:loc_y + 14, :]]
                self.pre_softmax += [self._build_model(self.x_crop[-1])]
                self.softmax += [tf.nn.softmax(self.pre_softmax[-1])]
                # reuse variables
                tf.get_variable_scope().reuse_variables()
                batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=vscope)

    self.crop_prediction = tf.argmax(self.pre_softmax, 2)
    self.mean_softmax = tf.reduce_mean(self.softmax, 0)
    epsilon = 1e-7
    self.mean_softmax = tf.clip_by_value(self.mean_softmax, epsilon, 1 - epsilon)
    self.y_pred = tf.argmax(self.mean_softmax,1)
    self.pre_mean_softmax = tf.log(self.mean_softmax)
    self.prediction = tf.argmax(self.pre_mean_softmax, 1)
    self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_mean_softmax)

    total_loss = tf.reduce_mean(self.xent) + weight_decay * tf.reduce_mean(self._decay())
    self.grad = self.opts.compute_gradients(total_loss)
    update_network_op =  self.opts.apply_gradients(self.grad,global_step=self.global_step)
    update_batchnorm_op = tf.group(*batchnorm_updates)
    self.train_step = tf.group(update_network_op, update_batchnorm_op)

    # adversarial gradient per mini-batch
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for ii in xrange(len(self.loc)):
            gpu_i=ii if mode == 'train' else 0
            with tf.device('/gpu:%d' % gpu_i):#tf.device('/cpu'):#
                loc_x, loc_y = self.loc[ii]
                padding = tf.constant([[0, 0], [loc_x - 14, 32 - loc_x - 14], [loc_y - 14, 32 - loc_y - 14], [0, 0]])
                self.adv_grad_i_pre_pad = tf.gradients(self.xent, self.x_crop[ii])[0]
                self.adv_grad_i = tf.pad(self.adv_grad_i_pre_pad, padding)
                self.adv_grads += [self.adv_grad_i]
                tf.get_variable_scope().reuse_variables()
    self.adv_grad = tf.reduce_mean(self.adv_grads)


    if 0:
        self.adv_grad = tf.reduce_sum(self.adv_grads,0)
        self.voted_pred = []
        batch_size = config['training_batch_size'] if mode == "train" else config['eval_batch_size']
        for i in range(batch_size) :  # loop over a batch
            y, idx, count = tf.unique_with_counts(self.prediction[i])
            majority = tf.argmax(count)
            self.voted_pred += [tf.gather(y, majority)]
            self.voted_pred = tf.stack(self.voted_pred)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.voted_pred, self.y_input), tf.float32))
    else:
        self.voted_pred = self.prediction
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
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
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
              stddev=np.sqrt(2.0/n)))
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



