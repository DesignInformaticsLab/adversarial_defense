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

import tensorflow as tf

import cifar10_input
from model_cifar import Model
#from pgd_attack import LinfPGDAttack
from pgd_multiGPU import *

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
data_path = config['data_path']

model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
cifar = cifar10_input.CIFAR10Data(data_path)

model = Model(config, mode='eval')

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

check_dict = {
'nat_img': [],
'adv_img': [],
'nat_pred': [],
'adv_pred': [],
'nat_preds': [],
'adv_preds': [],
'label': [],
'ckeck_idx': []
}

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  FLAGS = tf.app.flags.FLAGS
  tfconfig = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True,
  )
  tfconfig.gpu_options.allow_growth = True
  with tf.Session() as sess:#config=tfconfig

    # Restore the checkpoint
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      x_batch = np.asarray(x_batch, 'float32') / 255.
      y_batch = cifar.eval_data.ys[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      #x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      x_batch_adv = get_PGD(sess, model.adv_grad, dict_nat, model.x_input, epsilon=8. / 255, a=0.2 / 255, k=70)

      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

      nat_prediction, nat_voted_pred, acc_i, cur_xent_nat = sess.run(
                                      [model.prediction, model.voted_pred, model.accuracy,model.mean_xent],
                                      feed_dict = dict_nat)
      cur_corr_nat = acc_i*config['eval_batch_size']
      adv_prediction, adv_voted_pred, acc_i, cur_xent_adv = sess.run(
                                      [model.prediction, model.voted_pred, model.accuracy,model.mean_xent],
                                      feed_dict = dict_adv)
      cur_corr_adv = acc_i*config['eval_batch_size']
      print(eval_batch_size)
      print("Correctly classified natural examples: {}".format(cur_corr_nat))
      print("Correctly classified adversarial examples: {}".format(cur_corr_adv))
      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

      check_idx = ~ np.equal(nat_voted_pred, y_batch) * np.equal(adv_voted_pred, y_batch)
      check_dict['nat_img'] += [x_batch[check_idx]]
      check_dict['adv_img'] += [x_batch_adv[check_idx]]
      check_dict['nat_pred'] += [nat_voted_pred[check_idx]]
      check_dict['adv_pred'] += [adv_voted_pred[check_idx]]
      check_dict['nat_preds'] += [nat_prediction[check_idx]]
      check_dict['adv_preds'] += [adv_prediction[check_idx]]
      check_dict['label'] += [y_batch[check_idx]]
      check_dict['ckeck_idx'] += [check_idx]

    check_dict['nat_img'] = np.concatenate(check_dict['nat_img'], 0)
    check_dict['adv_img'] = np.concatenate(check_dict['adv_img'], 0)
    check_dict['nat_pred'] = np.concatenate(check_dict['nat_pred'], 0)
    check_dict['adv_pred'] = np.concatenate(check_dict['adv_pred'], 0)
    check_dict['nat_preds'] = np.concatenate(check_dict['nat_preds'], 0)
    check_dict['adv_preds'] = np.concatenate(check_dict['adv_preds'], 0)
    check_dict['label'] = np.concatenate(check_dict['label'], 0)
    check_dict['ckeck_idx'] = np.concatenate(check_dict['ckeck_idx'], 0)

    if 1:
        import matplotlib.pyplot as plt
        jj = 20
        plt.figure(figsize=(8.8, 2))
        for idx in range(10):
            plt.subplot(2, 10, idx + 1)
            plt.imshow(check_dict['nat_img'][jj + idx])
            plt.title('{}'.format(check_dict['nat_preds'][jj + idx]))
            plt.axis('off')
            plt.subplot(2, 10, idx + 1 + 10)
            plt.imshow(check_dict['adv_img'][jj + idx])
            plt.title('{}'.format(check_dict['adv_preds'][jj + idx]))
            plt.axis('off')

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))

if __name__ == '__main__':
    evaluate_checkpoint('/home/hope-yao/Documents/adversarial_defense/cifar/ckpt/crop_4_20_adv/half_half/lr_config1_adv/checkpoint-25001')