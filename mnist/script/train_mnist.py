import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model_mnist import *
from pgd_attack import *
import os
import numpy as np
from pgd_attack import LinfPGDAttack
#from utils import creat_dir
from tqdm import tqdm
slim = tf.contrib.slim

def main(cfg):
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    num_glimpse = cfg['num_glimpse']
    glimpse_size = cfg['glimpse_size']
    lr = cfg['lr']
    input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64,shape=(batch_size))

    # build classifier
    #model = Model_att(input_images, input_label, glimpse_size, num_glimpse)
    #model = Model_madry(input_images, input_label)
    #model = Model_crop(input_images, input_label)
    model = Model_crop(input_images, input_label)

    # setup attacker
    attack = LinfPGDAttack(model, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')

    ## OPTIMIZER ##
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer.compute_gradients(model.xent)
    train_op=optimizer.apply_gradients(grads)
    saver = tf.train.Saver()
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
    if 1: #saver.restore(sess,'/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/crop9_20_nat_itr50k/crop_ckpt')
        # saver.restore(sess, '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/crop9_20_itr150k/crop_ckpt')
        # saver.restore(sess,'/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/crop4_itr250000/crop_ckpt')
        # saver.restore(sess, '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/rotation4/crop_ckpt')

        checkpoint_dir = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/crop4_itr250000/crop_ckpt'
        var_names = tf.contrib.framework.list_variables(checkpoint_dir)
        b1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/conv1/biases')[0]
        b1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv1/biases')
        assign_op_b1 = b1.assign(b1_restore)
        w1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/conv1/weights')[0]
        w1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv1/weights')
        assign_op_w1 = w1.assign(w1_restore)
        b2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/conv2/biases')[0]
        b2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv2/biases')
        assign_op_b2 = b2.assign(b2_restore)
        w2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/conv2/weights')[0]
        w2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv2/weights')
        assign_op_w2 = w2.assign(w2_restore)
        fcb1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/fc1/biases')[0]
        fcb1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc1/biases')
        assign_op_fcb1 = fcb1.assign(fcb1_restore)
        fcw1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/fc1/weights')[0]
        fcw1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc1/weights')
        assign_op_fcw1 = fcw1.assign(fcw1_restore)
        fcb2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/fc2/biases')[0]
        fcb2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc2/biases')
        assign_op_fcb2 = fcb2.assign(fcb2_restore)
        fcw2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/fc2/weights')[0]
        fcw2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc2/weights')
        assign_op_fcw2 = fcw2.assign(fcw2_restore)

        sess.run([assign_op_b1, assign_op_w1, assign_op_b2, assign_op_w2,
                  assign_op_fcb1, assign_op_fcw1, assign_op_fcb2, assign_op_fcw2])  # or `assign_op.op.run()`

        checkpoint_dir = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/rotation4/crop_ckpt'
        var_names = tf.contrib.framework.list_variables(checkpoint_dir)
        b1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/conv1/biases')[0]
        b1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv1/biases')
        assign_op_b1 = b1.assign(b1_restore)
        w1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/conv1/weights')[0]
        w1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv1/weights')
        assign_op_w1 = w1.assign(w1_restore)
        b2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/conv2/biases')[0]
        b2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv2/biases')
        assign_op_b2 = b2.assign(b2_restore)
        w2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/conv2/weights')[0]
        w2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/conv2/weights')
        assign_op_w2 = w2.assign(w2_restore)
        fcb1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/fc1/biases')[0]
        fcb1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc1/biases')
        assign_op_fcb1 = fcb1.assign(fcb1_restore)
        fcw1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/fc1/weights')[0]
        fcw1_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc1/weights')
        assign_op_fcw1 = fcw1.assign(fcw1_restore)
        fcb2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/fc2/biases')[0]
        fcb2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc2/biases')
        assign_op_fcb2 = fcb2.assign(fcb2_restore)
        fcw2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier_rot/fc2/weights')[0]
        fcw2_restore = tf.contrib.framework.load_variable(checkpoint_dir, 'classifier/fc2/weights')
        assign_op_fcw2 = fcw2.assign(fcw2_restore)

        sess.run([assign_op_b1, assign_op_w1, assign_op_b2, assign_op_w2,
                  assign_op_fcb1, assign_op_fcw1, assign_op_fcb2, assign_op_fcw2])  # or `assign_op.op.run()`

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    hist = {'train_acc': [],
            'train_adv_acc': [],
            'test_acc': [],
            'test_adv_acc': [],
            'train_loss': [],
            'test_loss': [],
            'train_adv_loss': [],
            'test_adv_loss': []}
    train_iters=500000
    mnist.train._num_examples = 300

    for itr in tqdm(range(train_iters)):
        x_batch_train, y_batch_train = mnist.train.next_batch(batch_size)
        if 1: # adv train
            # x_batch_train_adv = attack.perturb(x_batch_train.reshape(batch_size, img_size, img_size, 1), y_batch_train, sess)
            # adv_dict_train = {input_images: x_batch_train_adv.reshape(batch_size, img_size, img_size, 1),
            #                  input_label: y_batch_train}
            nat_dict_train = {input_images: x_batch_train.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_train}
            # sess.run(train_op, feed_dict=adv_dict_train)
        else: # nat train
            nat_dict_train = {input_images: x_batch_train.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_train}
            sess.run(train_op, feed_dict=nat_dict_train)

        if itr % 100 == 0:
            train_acc_i, train_loss_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict_train)
            # counts = np.asarray([np.argmax(np.bincount(y_pred[:,i])) for i in range(batch_size)])
            # train_acc_i = np.mean(counts == nat_dict_train[input_label])
            x_batch_test, y_batch_test = mnist.test.next_batch(batch_size)
            nat_dict_test = {input_images: x_batch_test.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_test}
            test_acc_i, test_loss_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict_test)
            # counts = np.asarray([np.argmax(np.bincount(y_pred[:,i])) for i in range(batch_size)])
            # test_acc_i = np.mean(counts == nat_dict_test[input_label])
            print("iter: {}, train_acc:{}  test_acc:{} train_loss:{}  test_loss:{} "
                  .format(itr, train_acc_i, test_acc_i, train_loss_i, test_loss_i))

            x_batch_train_adv = attack.perturb(x_batch_train.reshape(batch_size, img_size, img_size, 1), y_batch_train, sess)
            adv_dict_train = {input_images: x_batch_train_adv.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_train}
            train_adv_acc_i, train_adv_loss_i = sess.run([model.accuracy, model.xent], feed_dict=adv_dict_train)
            # counts = np.asarray([np.argmax(np.bincount(y_pred[:,i])) for i in range(batch_size)])
            # train_adv_acc_i = np.mean(counts == adv_dict_train[input_label])
            x_batch_test_adv = attack.perturb(x_batch_test.reshape(batch_size, img_size, img_size, 1), y_batch_test, sess)
            adv_dict_test = {input_images: x_batch_test_adv.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_test}
            test_adv_acc_i, test_adv_loss_i = sess.run([model.accuracy, model.xent], feed_dict=adv_dict_test)
            # counts = np.asarray([np.argmax(np.bincount(y_pred[:,i])) for i in range(batch_size)])
            # test_adv_acc_i = np.mean(counts == adv_dict_test[input_label])
            print("iter: {}, train_adv_acc:{}  test_adv_acc:{} train_adv_loss:{}  test_adv_loss:{} "
                .format(itr, train_adv_acc_i, test_adv_acc_i, train_adv_loss_i, test_adv_loss_i))
            hist['train_acc'] += [train_acc_i]
            hist['train_adv_acc'] += [train_adv_acc_i]
            hist['test_acc'] += [test_acc_i]
            hist['test_adv_acc'] += [test_adv_acc_i]
            hist['train_loss'] += [train_loss_i]
            hist['test_loss'] += [test_loss_i]
            hist['train_adv_loss'] += [train_adv_loss_i]
            hist['test_adv_loss'] += [test_adv_loss_i]
            np.save('hist',hist)
            saver.save(sess,'crop_ckpt')
    print('done')


if __name__ == "__main__":


    cfg = {'batch_size': 32,
           'img_dim': 2,
           'img_size': 28,
           'num_glimpse': 5,
           'glimpse_size': 20,
           'lr': 1e-4
           }
    main(cfg)
