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
num_summary_steps = config['num_summary_steps']
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

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('xent adv train', model.mean_xent)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
with tf.Session(config=tfconfig) as sess:

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  nat_acc = []
  adv_acc = []
  training_time = 0.0
  for ii in range(max_num_training_steps):
    start = timer()
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)
    nat_dict = {model.x_input: x_batch.reshape(batch_size, 32, 32, 3),
                      model.y_input: y_batch}
    x_batch_adv = get_PGD(sess, model.adv_grad, nat_dict, model.x_input, epsilon=8. / 255, a=2. / 255, k=7)
    adv_dict = {model.x_input: x_batch_adv.reshape(batch_size, 32, 32, 3),
                      model.y_input: y_batch}
    _, adv_acc_i, adv_xent_i = sess.run([model.train_step, model.accuracy, model.xent], feed_dict=adv_dict)
    nat_acc_i, nat_xent_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict)
    nat_acc += [nat_acc_i]
    adv_acc += [adv_acc_i]
    end = timer()
    training_time += end - start

    if ii % num_output_steps == 0:
        print('Step {}:    ({})'.format(ii, datetime.now()))
        print('    training nat accuracy {:.4}%'.format(np.mean(nat_acc) * 100))
        print('    training adv accuracy {:.4}%'.format(np.mean(adv_acc) * 100))
        nat_acc = []
        adv_acc = []
        if ii != 0:
            print('    {} examples per second'.format( num_output_steps * batch_size / training_time))
            training_time = 0.0
        # Tensorboard summaries
        if ii % num_summary_steps == 0:
          summary = sess.run(merged_summaries, feed_dict=adv_dict)
          summary_writer.add_summary(summary, model.global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
          saver.save(sess,
                     os.path.join(model_dir, 'checkpoint'),
                     global_step=model.global_step)

    #
    # # Compute Adversarial Perturbations
    # start = timer()
    # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    # end = timer()
    # training_time += end - start
    #
    # nat_dict = {model.x_input: x_batch,
    #             model.y_input: y_batch}
    #
    # adv_dict = {model.x_input: x_batch_adv,
    #             model.y_input: y_batch}
    #
    # Output to stdout

    # # Actual training step
    # start = timer()
    # sess.run(model.train_step, feed_dict=adv_dict)
    # end = timer()
    # training_time += end - start
