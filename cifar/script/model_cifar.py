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
    self.n_gpu = 4
    self.mode = mode
    self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    self.y_input = tf.placeholder(tf.int64, shape=None)

    self.y_pred = []
    # self.mean_xent = []
    # self.weight_decay_loss = []
    loc = [[14, 14], [14, 18], [18, 14], [18, 18], [16, 16]]
    sub_batch_size = self.sub_batch_size = config['training_batch_size'] // self.n_gpu

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

    xent = []
    prediction = []
    tower_grads = []
    adv_grad = []
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for gpu_i in xrange(self.n_gpu):
            with tf.device('/gpu:%d' % gpu_i):
                with tf.name_scope('%s_%d' % ('TOWER', gpu_i)) as tower_scope:
                    x_input_i = self.x_input[gpu_i * sub_batch_size:(gpu_i + 1) * sub_batch_size]
                    y_input_i = self.y_input[gpu_i * sub_batch_size:(gpu_i + 1) * sub_batch_size]
                    xent_i, prediction_i = self.get_voting_model(vscope, tower_scope, x_input_i, y_input_i, loc)

            vscope.reuse_variables()
            assert vscope.reuse == True
            total_loss = tf.reduce_mean(xent_i) + weight_decay * tf.reduce_mean(self._decay())
            # training gradient per mini-batch
            grad_i = self.opts.compute_gradients(total_loss)
            # adversarial gradient per mini-batch
            adv_grad_i = tf.gradients(tf.reduce_mean(xent_i,1), x_input_i)[0]
            # summing up over mini-batches
            xent += [xent_i]
            prediction += [prediction_i]
            tower_grads += [grad_i]
            adv_grad += [adv_grad_i]
        # self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope())

    self.xent = tf.concat(xent, 0)
    self.mean_xent = tf.reduce_mean(self.xent)
    self.prediction = tf.concat(prediction,0)

    self.grads = average_gradients(tower_grads)
    self.train_step =  self.opts.apply_gradients(self.grads,global_step=self.global_step)

    self.adv_grad = tf.concat(adv_grad,0)
    self.voted_pred = []
    for i in range(config['training_batch_size']) :  # loop over a batch
        y, idx, count = tf.unique_with_counts(self.prediction[i])
        majority = tf.argmax(count)
        self.voted_pred += [tf.gather(y, majority)]
    self.voted_pred = tf.stack(self.voted_pred)
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.voted_pred, self.y_input), tf.float32))


  def get_voting_model(self, vscope, name_scope, x_input_i, y_input_i, loc):
      xent_i = []
      prediction_i = []
      for j, loc_i in enumerate(loc):
          loc_x, loc_y = loc_i
          x_crop_i = x_input_i[:, loc_x - 14:loc_x + 14, loc_y - 14:loc_y + 14, :]
          pre_softmax = self._build_model(x_crop_i)
          vscope.reuse_variables()
          assert vscope.reuse == True

          prediction_i += [tf.argmax(pre_softmax, 1)]
          xent_i_j = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_softmax, labels=y_input_i)
          xent_i += [xent_i_j]
      return tf.stack(xent_i,1), tf.stack(prediction_i,1)

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



