"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import cifar10_input
from model_cifar import Model

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
eval_batch_size = 1
eval_on_cpu = 1
data_path = '../cifar10_data'
# # Set upd the data, hyperparameters, and the model
# cifar = cifar10_input.CIFAR10Data(data_path)

model = Model(config, mode='eval')
saver = tf.train.Saver()
filename = "/home/hope-yao/Documents/adversarial_defense/cifar/ckpt/crop_8_28/a_very_robust_model_crop9_28_earlydrop/checkpoint-35002"
FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
  allow_soft_placement=True,
  log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
saver.restore(sess, filename)
print('done')