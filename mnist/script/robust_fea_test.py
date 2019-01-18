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

def main(model, ckpt_path):

    # setup attacker
    attack = LinfPGDAttack(model, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')

    ## OPTIMIZER ##
    saver = tf.train.Saver()

    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, ckpt_path)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    hist = {'train_fea':[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]],#[[]]*10,
            'test_fea': [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]],#[[]]*10,
            'test_adv_fea': [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]],#[[]]*10,
            }
    train_iters=500000
    mnist.train._num_examples = 300

    for itr in tqdm(range(330)):
        x_batch_train, y_batch_train = mnist.train.next_batch(batch_size)
        nat_dict_train = {input_images: x_batch_train.reshape(batch_size, img_size, img_size, 1),
                          input_label: y_batch_train}
        train_fea_i = sess.run(model.fea, feed_dict=nat_dict_train)

        x_batch_test, y_batch_test = mnist.test.next_batch(batch_size)
        nat_dict_test = {input_images: x_batch_test.reshape(batch_size, img_size, img_size, 1),
                          input_label: y_batch_test}
        test_fea_i = sess.run(model.fea, feed_dict=nat_dict_test)

        x_batch_test_adv = attack.perturb(x_batch_test.reshape(batch_size, img_size, img_size, 1), y_batch_test, sess)
        adv_dict_test = {input_images: x_batch_test_adv.reshape(batch_size, img_size, img_size, 1),
                         input_label: y_batch_test}
        test_adv_fea_i = sess.run(model.fea, feed_dict=adv_dict_test)

        for i, y_i in enumerate(y_batch_train):
           hist['train_fea'][int(y_i)] += [train_fea_i[i]]
        for i, y_i in enumerate(y_batch_test):
           hist['test_fea'][int(y_i)] += [test_fea_i[i]]
        for i, y_i in enumerate(y_batch_test):
           hist['test_adv_fea'][int(y_i)] += [test_adv_fea_i[i]]

    np.save('hist',hist)
    w = sess.run(model.vars[-2])
    print('done')
    return w

if __name__ == "__main__":


    cfg = {'batch_size': 32,
           'img_dim': 2,
           'img_size': 28,
           'num_glimpse': 5,
           'glimpse_size': 20,
           'lr': 1e-4
           }
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    num_glimpse = cfg['num_glimpse']
    glimpse_size = cfg['glimpse_size']
    lr = cfg['lr']
    input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64,shape=(batch_size))

    # build classifier
    if 1:
        model = Model_madry(input_images, input_label)
        ckpt_path = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/madry/natural/checkpoint-24900'
        #ckpt_path = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/madry/secret/checkpoint-99900'
    else:
        model = Model_crop(input_images, input_label)
        #ckpt_path = '../ckpt/crop9_20_nat_itr50k/crop_ckpt'
        ckpt_path = '../ckpt/crop9_20_itr150k/crop_ckpt'

    w = main(model, ckpt_path)

    hist = np.load('hist.npy').item()
    test_fea = np.asarray(hist['test_fea'])
    mean_fea = [[]]
    std_fea = [[]]
    for i in range(10):
        fea_value = np.asarray(test_fea[i][1:])
        mean_fea += [np.mean(fea_value,0)]
        std_fea += [np.std(fea_value,0)]


    import  matplotlib.pyplot as plt
    mean_fea = np.asarray(mean_fea[1:])
    std_fea = np.asarray(std_fea[1:])
    used_w = []
    for i, mean_fea_i in enumerate(mean_fea):
        std_fea_i = std_fea[i]
        idx = np.argsort(mean_fea_i)[::-1]
        k = 1024
        x = np.arange(k)
        plt.errorbar(x, mean_fea_i[idx[:k].tolist()], std_fea_i[idx[:k].tolist()], linestyle='None', marker='^', label ='class_{}'.format(i))
        #used_w += [np.sort(w[idx[:k], i])[::-1]]
        used_w += [np.sort(abs(w[idx[:k], i]))[::-1]]
    plt.legend()
    plt.axis([0,100,-1,7])
    plt.grid('on')
    
    plt.figure()
    for i in range(10):
        ww = used_w[i]/np.sum(used_w[i])
        plt.plot(ww[:100], label='class_{}'.format(i))
        #plt.plot(used_w[i], label='class_{}'.format(i))
    plt.legend()
    plt.grid('on')
    plt.show()

    #ww = np.asarray([used_w[i]/np.sum(used_w[i]) for i in range(10)])
    #np.save('ww',ww)