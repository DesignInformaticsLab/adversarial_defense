import tensorflow as tf
import numpy as np
from model_mnist import *
from pgd_attack import *
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

## parameters ##
batch_size=32
img_size=28
eps_bound = 0.3
res = 50


def get_landscape(model, ckpt_path):
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    ## restore vanilla model and get blackbox gradient ##
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    attack = LinfPGDAttack(model, epsilon=eps_bound, k=40, a=0.01, random_start=False, loss_func='xent')
    adv_dict_test = {model.x_input: x_batch,
                     model.y_input: y_batch}

    grad_val = sess.run(attack.grad,adv_dict_test)
    orth_grad_val = np.zeros_like(grad_val)
    for i in range(grad_val.shape[0]):
        k = grad_val[i].flatten()
        x = np.random.randn(784)
        x -= x.dot(k) * k / np.linalg.norm(k)**2
        orth_grad_val[i] = np.sign(x.reshape(1,28,28,1))#x.reshape(1,28,28,1)/ np.max(np.abs(x))#
        grad_val[i] = np.sign(grad_val[i])#grad_val[i]/np.max(np.abs(grad_val[i]))#


    vanilla_xent_img = np.zeros((batch_size, 1, res,res)) # bs, crop#, res_i, res_j
    pred = np.zeros((batch_size, 1, res,res)) # bs, crop#, res_i, res_j
    for i in range(res):
        for j in range(res):
            x_nat = adv_dict_test[input_images]
            x_adv = x_nat + eps_bound*i/res*grad_val + eps_bound*j/res*orth_grad_val
            #x_adv = np.clip(x_adv, 0, 1)
            xent_i, pred_i = sess.run([model.y_xent, model.y_pred], {input_images:x_adv, input_label:adv_dict_test[input_label]})
            vanilla_xent_img[:, :, i, j] = np.expand_dims(xent_i,-1)
            pred[:,:,i,j] = np.expand_dims(pred_i==y_batch, -1)
    return vanilla_xent_img, pred

def plotting(xent_img):
    res_x, res_y = xent_img[0].shape
    X = [[eps_bound*i/res for i in range(res_x)] for j in range(res_x)]
    Y = [[eps_bound*j/res for i in range(res_y)] for j in range(res_y)]
    Z1, Z2, Z3, Z4= xent_img

    fig = plt.figure(figsize=(12,3))

    ax = fig.add_subplot(141, projection='3d')
    ax.plot_surface(X,Y,Z1.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(142, projection='3d')
    ax.plot_surface(X,Y,Z2.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(143, projection='3d')
    ax.plot_surface(X,Y,Z3.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(144, projection='3d')
    ax.plot_surface(X,Y,Z4.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')
    return fig

def plotting_crops(xent_img):
    fig = plt.figure(figsize=(10,7))
    X = [[eps_bound*i/res for i in range(res)] for j in range(res)]
    Y = [[eps_bound*j/res for i in range(res)] for j in range(res)]

    ax = fig.add_subplot(331, projection='3d')
    Z = xent_img[0,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(332, projection='3d')
    Z = xent_img[1,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(333, projection='3d')
    Z = xent_img[2,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(334, projection='3d')
    Z = xent_img[3,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(335, projection='3d')
    Z = xent_img[4,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(336, projection='3d')
    Z = xent_img[5,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(337, projection='3d')
    Z = xent_img[6,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(338, projection='3d')
    Z = xent_img[7,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')

    ax = fig.add_subplot(339, projection='3d')
    Z = xent_img[8,:,:]
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')
    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')
    # plt.show()
    return fig

if __name__ == '__main__':
    import os
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_batch, y_batch = mnist.test.next_batch(batch_size)
    x_batch = x_batch.reshape(batch_size, img_size, img_size, 1)
    input_images = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64, shape=(batch_size))

    if 0:
        model = Model_madry(input_images, input_label)
        ckpt_path = '../ckpt/madry/natural/checkpoint-24900'
        xent_img, acc_img = get_landscape(model, ckpt_path)
        np.save('mnat',{'xent_img':xent_img, 'acc_img':acc_img})

        ckpt_path = '../ckpt/madry/secret/checkpoint-99900'
        xent_img, acc_img = get_landscape(model, ckpt_path)
        np.save('madv',{'xent_img':xent_img, 'acc_img':acc_img})

    else:
        model = Model_crop(input_images, input_label)
        ckpt_path = '../ckpt/crop9_20_nat_itr50k/crop_ckpt'
        xent_img, acc_img = get_landscape(model, ckpt_path)
        np.save('cnat',{'xent_img':xent_img, 'acc_img':acc_img})

        ckpt_path = '../ckpt/crop9_20_itr150k/crop_ckpt'
        xent_img, acc_img = get_landscape(model, ckpt_path)
        np.save('cadv',{'xent_img':xent_img, 'acc_img':acc_img})

    import numpy as np
    dir = './sign_grad_clip_pix'
    mnat = np.load(dir+'/mnat.npy').item()
    madv = np.load(dir+'/madv.npy').item()
    cnat = np.load(dir+'/cnat.npy').item()
    cadv = np.load(dir+'/cadv.npy').item()
    n = 25
    for i in range(10):
        plotting([mnat['xent_img'][i, 0, :n, :n], madv['xent_img'][i, 0, :n, :n], cnat['xent_img'][i, 0, :n, :n],
                  cadv['xent_img'][i, 0, :n, :n]])
        #plotting([mnat['acc_img'][i, 0, :n, :n], madv['acc_img'][i, 0, :n, :n], cnat['acc_img'][i, 0, :n, :n],
        #         cadv['acc_img'][i, 0, :n, :n]])
    plt.show()
