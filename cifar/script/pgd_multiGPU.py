"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import copy

def get_PGD(sess, adv_grad, feed_dict_pgd, x_input_pl, epsilon=0.1, a=0.002, k=50, rand=True, dist='Linf'):
  if dist == 'Linf':
    x = get_PGD_Linf(sess, adv_grad, feed_dict_pgd, x_input_pl, epsilon, a, k, rand)
  elif dist == 'L2':
    x = get_PGD_L2(sess, adv_grad, feed_dict_pgd, x_input_pl, epsilon, a, k, rand)
  else:
    print('not implemented')
  return x


def get_PGD_Linf(sess, adv_grad, feed_dict_pgd, x_input_pl, epsilon, a, k, rand):
  """Given a set of examples (x_nat, y), returns a set of adversarial
     examples within epsilon of x_nat in l_infinity norm."""

  x_nat = feed_dict_pgd[x_input_pl]
  if rand:
    x = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape)
  else:
    x = np.copy(x_nat)

  for i in range(k):
    grad = sess.run(adv_grad, feed_dict=feed_dict_pgd)

    x += a * np.sign(grad)
    x = np.clip(x, x_nat - epsilon, x_nat + epsilon)
    x = np.clip(x, 0, 1)  # ensure valid pixel range

  return x


def sphere_rand(input_size, epsilon):
  '''
  algrithm adapted from: https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
  :param epsilon:
  :return:
  '''
  bs = input_size[0]
  img_size = input_size[1:]
  x = []
  for i in range(bs):
    perturb = np.random.normal(0, 1, img_size)
    norm = np.linalg.norm(np.reshape(perturb,[-1]),2)
    U = np.random.uniform(0, 1, img_size)
    U = np.power(U,  1/(img_size[0]*img_size[1]*img_size[2]))
    perturb = perturb / norm * epsilon * U
    x += [np.expand_dims(perturb,0)]
  return  np.concatenate(x,0)


def get_PGD_L2(sess, adv_grad, feed_dict_pgd, x_input_pl, epsilon, a, k, rand):
  """Given a set of examples (x_nat, y), returns a set of adversarial
     examples within epsilon of x_nat in l_infinity norm."""

  x_nat = feed_dict_pgd[x_input_pl]
  input_size = x_input_pl.get_shape().as_list()
  bs = input_size[0]

  if rand:
    sphere_perturb = sphere_rand(input_size, np.random.uniform(0,epsilon))
    # start from a random point inside L2 sphere
    x = x_nat + sphere_perturb
  else:
    x = np.copy(x_nat)

  for i in range(k):
    grad = sess.run(adv_grad, feed_dict=feed_dict_pgd)

    if 1:
      # attack normalize
      att_norm2 = np.linalg.norm(np.reshape(grad, [bs, -1]), ord=2, axis=1)
      x_i = x + a * grad/np.reshape(att_norm2, [bs,1,1,1]) #perturb along the spherical projection with step size a
      # adv img normalize
      x_diff = x_i - x_nat #accumulated perturbation
      img_norm2 = np.linalg.norm(np.reshape(x_diff, [bs, -1]), ord=2, axis=1)
      # bounded_norm = np.clip(img_norm2, 0 ,epsilon)
      ratio = np.asarray([img_norm2[i] if img_norm2[i]<epsilon else epsilon for i in range(bs)])#clip accumulated perturbation inside sphere radius epsilon
      x = x_nat + x_diff/np.reshape(img_norm2,[bs,1,1,1]) * np.reshape(ratio,[bs,1,1,1])
      # ensure valid pixel range
      x = np.clip(x, 0, 1)
    else:
      # attack normalize
      att_norm2 = np.linalg.norm(np.reshape(grad, [bs, -1]), ord=2, axis=1)
      x_i = x + epsilon * grad / np.reshape(att_norm2, [bs, 1, 1, 1])  # perturb along the spherical projection with step size a
      # ensure valid pixel range
      x = np.clip(x_i, 0, 1)

  return x