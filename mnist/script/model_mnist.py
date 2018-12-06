"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim



class Model_madry(object):
  def __init__(self, x, y ):
    self.x_input = x
    self.y_input = y

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_input, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(self.y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.vars = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

import numpy as np
class Model_crop():
    def __init__(self, x, y, idx=None):
        self.x_input = x
        self.y_input = y

        self.x_voting = []
        self.x_crop = []
        self.pre_softmax = []
        loc = [[10, 10], [10, 14], [10, 18], [14, 10], [14, 14], [14, 18], [18, 10], [18, 14], [18, 18]]
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
                pre_softmax = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')
                self.pre_softmax += [pre_softmax]
                self.y_pred += [tf.argmax(pre_softmax, 1)]
                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=pre_softmax)
                self.xent += [y_xent]
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse == True
            self.pre_softmax = tf.reduce_mean(self.pre_softmax,0)
            self.y_pred = tf.stack(self.y_pred)
            self.xent_indv = self.xent
            self.xent = tf.reduce_mean(self.xent)
            self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')


class Model_crop_presftmx():#
    def __init__(self, x, y):
        self.x_input = x
        self.y_input = y

        self.x_voting = []
        self.x_crop = []
        loc = np.arange(10,18,3, dtype='int64')
        loc = [(i, j) for i in loc for j in loc]
        self.pre_softmax = []
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
                x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')
                self.pre_softmax += [x]
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse == True
            self.pre_softmax = tf.reduce_mean(self.pre_softmax, 0)
            self.y_pred = tf.argmax(self.pre_softmax,1)
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_softmax)
            self.xent_indv = tf.expand_dims(y_xent,0)
            self.xent = tf.reduce_sum(y_xent)
            self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')


class Model_crop_insftmx():#
    def __init__(self, x, y):
        self.x_input = x
        self.y_input = y

        self.x_voting = []
        self.x_crop = []
        loc = np.arange(10,18,3, dtype='int64')
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
                x = slim.fully_connected(x, num_outputs=10, activation_fn=tf.nn.softmax, scope='fc2')
                self.softmax += [x]
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse == True
            self.mean_softmax = tf.reduce_mean(self.softmax, 0)
            # reference: https://github.com/tensorflow/tensorflow/issues/2462
            epsilon = 1e-7
            self.mean_softmax = tf.clip_by_value(self.mean_softmax, epsilon, 1 - epsilon)
            self.y_pred = tf.argmax(self.softmax,2)
            self.pre_softmax = self.pre_mean_softmax = tf.log(self.mean_softmax)
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_mean_softmax)
            self.xent_indv = tf.expand_dims(y_xent,0)
            self.xent = tf.reduce_sum(y_xent)
            self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
