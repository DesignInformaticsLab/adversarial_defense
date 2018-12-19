"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from model_cifar import Model
import cifar10_input
# from pgd_attack import LinfPGDAttack
from pgd_multiGPU import *

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
# num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
data_path = config['data_path']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
model = Model(config, mode='train')


# Set up adversary
# attack = LinfPGDAttack(model,
#                        config['epsilon'],
#                        config['num_steps'],
#                        config['step_size'],
#                        config['random_start'],
#                        config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=30)

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
with tf.Session() as sess:#config=tfconfig

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  nat_acc = []
  adv_acc = []
  nat_xent = []
  adv_xent = []
  training_time = 0.0
  for ii in range(max_num_training_steps):
    start = timer()
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)
    x_batch = np.asarray(x_batch, 'float32') / 255.

    nat_dict = {model.x_input: x_batch.reshape(batch_size, 32, 32, 3),
                      model.y_input: y_batch}
    x_batch_adv = get_PGD(sess, model.adv_grad, model.x_input, model.y_input, x_batch, y_batch, epsilon=8. / 255, a=2. / 255, k=7)
    adv_dict = {model.x_input: x_batch_adv.reshape(batch_size, 32, 32, 3),
                      model.y_input: y_batch}
    # x_batch_adv = get_PGD(sess, model.adv_grad, nat_dict, model.x_input, epsilon=8. / 255, a=2. / 255, k=7)
    # x_batch_adv_near = get_PGD(sess, model.adv_grad, nat_dict, model.x_input, epsilon=1. / 255, a=0.5 / 255, k=3)
    # adv_dict = {model.x_input: np.concatenate([x_batch_adv,x_batch_adv_near],0).reshape(batch_size, 32, 32, 3),
    #                   model.y_input: np.concatenate([y_batch,y_batch],0)}
    # nat_dict = {model.x_input: np.concatenate([x_batch,x_batch],0).reshape(batch_size, 32, 32, 3),
    #                   model.y_input: np.concatenate([y_batch,y_batch],0)}
    _, adv_acc_i, adv_xent_i = sess.run([model.train_step, model.accuracy, model.xent], feed_dict=adv_dict)
    nat_acc_i, nat_xent_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict)
    nat_acc += [nat_acc_i]
    adv_acc += [adv_acc_i]
    adv_xent += [adv_xent_i]
    nat_xent += [nat_xent_i]
    end = timer()
    training_time += end - start

    if ii % num_output_steps == 0:
        print('Step {}:    ({})'.format(ii, datetime.now()))
        print('    training nat accuracy {:.4}%'.format(np.mean(nat_acc) * 100))
        print('    training adv accuracy {:.4}%'.format(np.mean(adv_acc) * 100))
        print('    training nat xent {:.4} '.format(np.mean(nat_xent) ))
        print('    training adv xent {:.4} '.format(np.mean(adv_xent) ))
        print('    {} samples per second'.format(num_output_steps * batch_size / training_time))
        training_time = 0.0

    # # Tensorboard summaries
    # if ii % num_summary_steps == 0:
    #   summary = sess.run(merged_summaries_adv, feed_dict=adv_dict)
    #   summary_writer.add_summary(summary, model.global_step.eval(sess))
    #   summary = sess.run(merged_summaries_nat, feed_dict=nat_dict)
    #   summary_writer.add_summary(summary, model.global_step.eval(sess))


    # Write a checkpoint
    # if ii % num_checkpoint_steps == 0:
    #   saver.save(sess,
    #              os.path.join(model_dir, 'checkpoint'),
    #              global_step=model.global_step)

    if 0:
        # history replay
        start = timer()
        sess.run(model.train_step, feed_dict=adv_dict)

        if ii > 20000 and adv_acc > 0.3 and nat_acc > 0.6:
            nat_correct_prediction = sess.run(model.correct_prediction, feed_dict=nat_dict)
            adv_correct_prediction = sess.run(model.correct_prediction, feed_dict=adv_dict)
            idx = nat_correct_prediction * ~ adv_correct_prediction
            if 'adv_x_pool' in locals():
                adv_x_pool = np.concatenate([adv_x_pool, adv_dict[model.x_input][idx]], 0)
                adv_y_pool = np.concatenate([adv_y_pool, adv_dict[model.y_input][idx]], 0)
                prob_pool = np.concatenate([prob_pool, np.ones(np.sum(idx))])
            else:
                adv_x_pool = adv_dict[model.x_input][idx]
                adv_y_pool = adv_dict[model.y_input][idx]
                prob_pool = np.ones(np.sum(idx))
            if ii % 1000 == 0:
                adv_replay_dict = {'adv_x': adv_x_pool, 'adv_y': adv_y_pool, 'adv_p': prob_pool}
                np.save('adv_replay_dict', adv_replay_dict)

        # history replay
        if 'adv_x_pool' in locals():
            if adv_x_pool.shape[0] >= batch_size:
                from numpy.random import choice

                rand_idx = choice(adv_x_pool.shape[0], batch_size, p=prob_pool / np.sum(prob_pool))
                adv_x_replay = adv_x_pool[rand_idx]
                adv_y_replay = adv_y_pool[rand_idx]
                adv_dict_replay = {model.x_input: adv_x_replay,
                                   model.y_input: adv_y_replay}
                sess.run(model.train_step, feed_dict=adv_dict_replay)
                adv_replay_correct_prediction = sess.run(model.correct_prediction, feed_dict=adv_dict_replay)
                prob_pool[rand_idx[~adv_replay_correct_prediction]] /= 2.

        end = timer()
        training_time += end - start
