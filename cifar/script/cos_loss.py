"""Functions for building the face recognition network.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import time
import tensorflow.contrib.slim as slim
import pickle
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile


def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):
    rand_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rand_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc': rand_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def coco_forward(xw, y, m, name=None):
    # pdb.set_trace()
    xw_copy = xw.copy()
    num = len(y)
    orig_ind = range(num)
    xw_copy[orig_ind, y] -= m
    return xw_copy


def coco_help(grad, y):
    grad_copy = grad.copy()
    return grad_copy


def coco_backward(op, grad):
    y = op.inputs[1]
    m = op.inputs[2]
    grad_copy = tf.py_func(coco_help, [grad, y], tf.float32)
    return grad_copy, y, m


def coco_func(xw, y, m, name=None):
    #with tf.op_scope([xw, y, m], name, "Coco_func") as name:
    coco_out = py_func(coco_forward, [xw, y, m], tf.float32, name=name, grad_func=coco_backward)
    return coco_out


def cos_loss(x, y, w, reuse=False, alpha=0.25, scale=64, name='cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights
    xs = x.get_shape()
    #with tf.variable_scope('centers_var', reuse=reuse) as center_scope:

    # normalize the feature and weight
    # (N,D)
    x_feat_norm = tf.nn.l2_normalize(x, 1, 1e-10)
    # (D,C)
    w_feat_norm = tf.nn.l2_normalize(w, 0, 1e-10)

    # get the scores after normalization
    # (N,C)
    xw_norm = tf.matmul(x_feat_norm, w_feat_norm)
    # implemented by py_func
    # value = tf.identity(xw)
    # substract the marigin and scale it
    value = coco_func(xw_norm, y, alpha) * scale

    # implemented by tf api
    # margin_xw_norm = xw_norm - alpha
    # label_onehot = tf.one_hot(y,num_cls)
    # value = scale*tf.where(tf.equal(label_onehot,1), margin_xw_norm, xw_norm)

    # compute the loss as softmax loss
    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=value))

    tmp = {'xw_norm': xw_norm, 'x_feat_norm': x_feat_norm, 'w_feat_norm': w_feat_norm}
    return cos_loss, value, tmp

