import tensorflow as tf
import numpy as np
from model_mnist import *
from pgd_attack import *
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

## parameters ##
batch_size=1
img_size=28
eps_bound = 0.3
res = 50

def get_model(input_images_pl, input_label_pl, case):
    if case in ['M_nat', 'M_adv']:
        model = Model_madry(input_images_pl, input_label_pl)
    elif case in ['C_nat', 'C_adv']:
        model = Model_crop(input_images_pl, input_label_pl)
    # model = Model_crop_insftmx(input_images_pl, input_label_pl)


    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    ## OPTIMIZER ##
    _lambda = 1.
    cls_score = tf.gather(model.pre_softmax[0], input_label_pl[0])
    loss = cls_score - _lambda * tf.nn.l2_loss(input_images_pl)
    grad = tf.gradients(loss, model.x_input)[0]

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(model.vars)
    if case == 'M_nat':
        saver.restore(sess,'../ckpt/madry/natural/checkpoint-24900')
    elif case == 'M_adv':
        saver.restore(sess,'../ckpt/madry/secret/checkpoint-99900')
    elif case == 'C_nat':
        saver.restore(sess,'../ckpt/crop64_20_nat_itr35k/bb_64crop_ckpt')
    elif case == 'C_adv':
        saver.restore(sess,'../ckpt/crop9_20_itr150k/crop_ckpt')
    ## Our model insftmx
    # saver.restore(sess, '../ckpt/crop9_20_insftmx_itr170k/insftmx_crop_ckpt')

    metric = {'cls_score': cls_score, 'xent': model.xent, 'loss': loss}
    return sess, metric, grad

def run_model(sess, metric, grad, x_batch):
    hist = {'xent': [], 'loss': [], "input": [], 'cls_score': []}
    for i in range(100):
        feed_dict = {input_images_pl:x_batch, input_label_pl:y_batch}
        xent_i, cls_score_i, loss_i = sess.run([metric['xent'], metric['cls_score'], metric['loss']], feed_dict)
        hist['cls_score'] += [cls_score_i]
        hist['xent'] += [xent_i]
        hist['input'] += [np.copy(x_batch)]
        hist['loss'] += [loss_i]
        g = sess.run(grad, feed_dict)
        x_batch += np.sign(g)*0.01
        x_batch = np.clip(x_batch, 0, 1)
    print('done')
    return hist

def viz_input(result):
    st_cls_score, ed_cls_score, st_img, ed_img = result['st_cls_score'], result['ed_cls_score'], result['st_img'], result['ed_img']
    ref_cls_score, ref_img = result['ref_cls_score'], result['ref_img']
    fig1 = plt.figure()
    sns.distplot(st_cls_score, label='st')
    sns.distplot(ed_cls_score, label='ed')
    sns.distplot(ref_cls_score, label='ref')
    plt.legend()

    fig2 = plt.figure()
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, i*10+j+1)
            plt.imshow(st_img[i*10+j][0, :, :, 0], vmax=1.0, vmin=0.0)
            plt.axis('off')

    fig3 = plt.figure()
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, i*10+j+1)
            plt.imshow(ed_img[i*10+j][0, :, :, 0], vmax=1.0, vmin=0.0)
            plt.axis('off')

    fig4 = plt.figure()
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, i*10+j+1)
            plt.imshow(ref_img[i*10+j][0, :, :, 0], vmax=1.0, vmin=0.0)
            plt.axis('off')

    figs = [fig1, fig2, fig3, fig4]
    return figs

if __name__ == '__main__':
    import os
    import seaborn as sns
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    input_images_pl = tf.placeholder(shape=(batch_size, img_size, img_size, 1), dtype=tf.float32, name='input_pl')
    input_label_pl = tf.placeholder(shape=(batch_size), dtype=tf.int64, name='label_pl')

    # sess, metric, grad = get_model(input_images_pl, input_label_pl, 'M_adv')
    sess, metric, grad = get_model(input_images_pl, input_label_pl, 'M_nat')
    # sess, metric, grad = get_model(input_images_pl, input_label_pl, 'C_nat')
    # sess, metric, grad = get_model(input_images_pl, input_label_pl, 'C_adv')

    # tf.reset_graph()
    result = {'st_cls_score': [],
              'ed_cls_score': [],
              'ref_cls_score': [],
              'st_img': [],
              'ed_img': [],
              'ref_img':[],
              }
    for i in range(300):
        x_batch, y_batch = mnist.test.next_batch(batch_size)
        x_batch = x_batch.reshape((batch_size,img_size,img_size,1))
        x_batch_rand = np.random.randn(batch_size,img_size,img_size,1)
        hist = run_model(sess, metric, grad, x_batch)
        ref_cls_score_i = sess.run(metric['cls_score'],  {input_images_pl:x_batch, input_label_pl:y_batch})
        result['st_cls_score'] += [hist['cls_score'][0]]
        result['ed_cls_score'] += [hist['cls_score'][-1]]
        result['ref_cls_score'] += [ref_cls_score_i]
        result['ref_img'] += [x_batch]
        result['st_img'] += [hist['input'][0]]
        result['ed_img'] += [hist['input'][-1]]
    viz_input(result)
    plt.show()
    print('done')
