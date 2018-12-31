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
    # model = Model_madry(input_images, input_label)
    #model = Model_crop(input_images, input_label)
    model = Model_crop_cosine(input_images, input_label)

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
    if 0:
        saver.restore(sess, '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/crop9_20_itr150k/crop_ckpt')
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
    for itr in tqdm(range(train_iters)):
        x_batch_train, y_batch_train = mnist.train.next_batch(batch_size)
        if 1: # adv train
            x_batch_train_adv = attack.perturb(x_batch_train.reshape(batch_size, img_size, img_size, 1), y_batch_train, sess)
            adv_dict_train = {input_images: x_batch_train_adv.reshape(batch_size, img_size, img_size, 1),
                             input_label: y_batch_train}
            nat_dict_train = {input_images: x_batch_train.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_train}
            sess.run(train_op, feed_dict=adv_dict_train)
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
            np.save('hist_0.15',hist)
            saver.save(sess,'/home/hyao23/cos_loss_test/adversarial_defense/mnist/script/crop_ckpt_0.15')
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
