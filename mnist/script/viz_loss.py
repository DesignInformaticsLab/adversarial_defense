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


def get_surrogate(model_type):
    input_images = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64, shape=(batch_size))
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    ## restore vanilla model and get blackbox gradient ##
    if 1:
        model = Model_madry(input_images, input_label)
        saver = tf.train.Saver()
        saver.restore(sess,'../ckpt/madry/natural/checkpoint-24900')
    else:
        model = Model_madry(input_images, input_label)
        saver = tf.train.Saver()
        saver.restore(sess,'../ckpt/madry/natural/checkpoint-24900')

    attack = LinfPGDAttack(model, epsilon=eps_bound, k=40, a=0.01, random_start=True, loss_func='xent')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_batch, y_batch = mnist.test.next_batch(batch_size)
    x_batch = x_batch.reshape(batch_size, img_size, img_size, 1)
    adv_dict_test = {model.x_input: x_batch,
                     model.y_input: y_batch}
    grad_val = sess.run(attack.grad,adv_dict_test)
    orth_grad_val = np.zeros_like(grad_val)
    for i in range(grad_val.shape[0]):
        k = grad_val[i].flatten()
        x = np.random.randn(784)
        x -= x.dot(k) * k / np.linalg.norm(k)**2
        orth_grad_val[i] = x.reshape(1,28,28,1) / np.max(np.abs(x))
        grad_val[i] = grad_val[i]/np.max(np.abs(grad_val[i]))

    return adv_dict_test, grad_val, orth_grad_val



def run_blackbox_attack(adv_dict_test, grad_val, orth_grad_val):
    input_images = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64, shape=(batch_size))
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    model = Model_madry(input_images, input_label)
    attack = LinfPGDAttack(model, epsilon=eps_bound, k=40, a=0.01, random_start=True, loss_func='xent')
    saver = tf.train.Saver()
    saver.restore(sess,'../ckpt/madry/natural/checkpoint-24900')
    vanilla_xent_img = np.zeros((batch_size, 1, res,res)) # bs, crop#, res_i, res_j
    for i in range(res):
        for j in range(res):
            x_adv = adv_dict_test[input_images] + eps_bound*i/res*grad_val + eps_bound*j/res*orth_grad_val
            xent_i = sess.run(model.y_xent, {input_images:x_adv, input_label:adv_dict_test[input_label]})
            vanilla_xent_img[:, :, i, j] = np.expand_dims(xent_i,-1)


    ## restore madry's model ##
    madry_xent_img = np.zeros((batch_size, 1, res,res)) # bs, crop#, res_i, res_j
    saver.restore(sess,'../ckpt/madry/secret/checkpoint-99900')
    for i in range(res):
        for j in range(res):
            x_adv = adv_dict_test[input_images] + eps_bound*i/res*grad_val + eps_bound*j/res*orth_grad_val
            xent_i = sess.run(model.y_xent, {input_images:x_adv, input_label:adv_dict_test[input_label]})
            madry_xent_img[:, :, i, j] = np.expand_dims(xent_i,-1)

    # restore our's model ##
    our_model = Model_crop(input_images, input_label)
    saver = tf.train.Saver(var_list=our_model.vars)
    saver.restore(sess,'../ckpt/crop9_20_itr150k/crop_ckpt')
    ours_xent_img = np.zeros((batch_size, 9, res,res)) # bs, crop#, res_i, res_j
    pred_list = np.zeros((batch_size, 9, res,res)) # bs, crop#, res_i, res_j
    for i in range(res):
        for j in range(res):
            x_adv = adv_dict_test[input_images] + eps_bound*i/res*grad_val + eps_bound*j/res*orth_grad_val
            xent_i, pred_i = sess.run([our_model.xent_indv,our_model.y_pred], {input_images:x_adv, input_label:adv_dict_test[input_label]})
            ours_xent_img[:, :, i, j] = np.asarray(xent_i).transpose((1,0))
            pred_list[:, :, i, j] = np.asarray(pred_i).transpose((1,0))

    ## restore our's model insftmx##
    tf.get_variable_scope().reuse_variables()
    our_model_insftmx = Model_crop_insftmx(input_images, input_label)
    saver = tf.train.Saver(var_list=our_model_insftmx.vars)
    saver.restore(sess,'../ckpt/crop9_20_insftmx_itr170k/insftmx_crop_ckpt')
    insftmx_xent_img = np.zeros((batch_size, 9, res,res)) # bs, crop#, res_i, res_j
    insftmx_pred_list = np.zeros((batch_size, 9, res,res)) # bs, crop#, res_i, res_j
    for i in range(res):
        for j in range(res):
            x_adv = adv_dict_test[input_images] + eps_bound*i/res*grad_val + eps_bound*j/res*orth_grad_val
            xent_i, pred_i = sess.run([our_model_insftmx.xent_indv,our_model_insftmx.y_pred], {input_images:x_adv, input_label:adv_dict_test[input_label]})
            insftmx_xent_img[:, :, i, j] = np.asarray(xent_i).transpose((1,0))
            insftmx_pred_list[:, :, i, j] = np.asarray(pred_i).transpose((1,0))

    landscape_dict = {'vanilla_xent_img':vanilla_xent_img,
                      'madry_xent_img':madry_xent_img,
                      'ours_xent_img':ours_xent_img,
                      'insftmx_xent_img':insftmx_xent_img}
    return landscape_dict

def plotting(xent_img):
    X = [[eps_bound*i/res for i in range(res)] for j in range(res)]
    Y = [[eps_bound*j/res for i in range(res)] for j in range(res)]
    Z = xent_img

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z.transpose((1,0)), rstride=2, cstride=2, cmap='seismic')

    ax.view_init(30, -120)
    ax.set_xlabel('$\epsilon_1$')
    ax.set_ylabel('$\epsilon_2$')
    # plt.show()
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
    if 1:
        model_type = 1
        adv_dict_test, grad_val, orth_grad_val = get_surrogate(model_type)
        landscape_dict = run_blackbox_attack(adv_dict_test, grad_val, orth_grad_val)
        np.save('../asset/blackbox_landscape',landscape_dict )
    else:
        landscape_dict = np.load('../asset/blackbox_landscape.npy').item()

    for idx in range(30):
        dir = os.path.join('../asset/','blackbox_landscape_{}'.format(idx))
        if not os.path.exists(dir): os.mkdir(dir)
        fig = plotting(np.median(landscape_dict['vanilla_xent_img'][idx,:],0))
        fig.savefig(os.path.join(dir,'vanilla.png'))
        fig = plotting(np.median(landscape_dict['madry_xent_img'][idx,:],0))
        fig.savefig(os.path.join(dir,'madrys.png'))
        fig = plotting(np.median(landscape_dict['ours_xent_img'][idx,:],0))
        fig.savefig(os.path.join(dir,'ours.png'))
        fig = plotting(np.median(landscape_dict['insftmx_xent_img'][idx,:],0))
        fig.savefig(os.path.join(dir,'insftmx.png'))

        fig = plotting_crops(landscape_dict['ours_xent_img'][idx])
        fig.savefig(os.path.join(dir,'crops.png'))
        plt.close('all')