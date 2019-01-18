import tensorflow as tf
import numpy as np
from pgd_attack import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

## parameters ##
batch_size=100
img_size=32
eps_bound = 8./255
res = 50


def get_landscape(model, ckpt_path):
    ## restore vanilla model and get blackbox gradient ##
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    attack = LinfPGDAttack(model, epsilon=eps_bound, num_steps=40, step_size=0.01, random_start=False, loss_func='xent')
    adv_dict_test = {model.x_input: x_batch,
                     model.y_input: y_batch}

    grad_val = sess.run(attack.grad,adv_dict_test)
    orth_grad_val = np.zeros_like(grad_val)
    for i in range(grad_val.shape[0]):
        k = grad_val[i].flatten()
        x = np.random.randn(32*32*3)
        x -= x.dot(k) * k / np.linalg.norm(k)**2
        orth_grad_val[i] = np.sign(x.reshape(1,32,32,3))#x.reshape(1,28,28,1)/ np.max(np.abs(x))#
        grad_val[i] = np.sign(grad_val[i])#grad_val[i]/np.max(np.abs(grad_val[i]))#


    vanilla_xent_img = np.zeros((batch_size, 1, res,res)) # bs, crop#, res_i, res_j
    pred = np.zeros((batch_size, 1, res,res)) # bs, crop#, res_i, res_j
    for i in range(res):
        for j in range(res):
            x_adv = adv_dict_test[model.x_input] + eps_bound*i/res*grad_val + eps_bound*j/res*orth_grad_val
            x_adv = np.clip(x_adv,0,1)
            xent_i, pred_i = sess.run([model.y_xent, model.y_pred], {model.x_input:x_adv, model.y_input:adv_dict_test[model.y_input]})
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
    import cifar10_input
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    raw_cifar = cifar10_input.CIFAR10Data('../cifar10_data')
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess)
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)
    x_batch = x_batch/255.
    if 1:
        from model_cifar import Model as Model_crop
        from model_cifar_madry import Model as Model_madry
        import json
        with open('config.json') as config_file:
            config = json.load(config_file)

        if 1:
            model = Model_madry(config, mode='eval')
            ckpt_path = '../ckpt/madry/nat_models/naturally_trained/checkpoint-70000'
            xent_img, acc_img = get_landscape(model, ckpt_path)
            np.save('mnat',{'xent_img':xent_img, 'acc_img':acc_img})

            ckpt_path = '../ckpt/madry/adv_models/adv_trained/checkpoint-70000'
            xent_img, acc_img = get_landscape(model, ckpt_path)
            np.save('madv',{'xent_img':xent_img, 'acc_img':acc_img})
        else:
            model = Model_madry(config, mode='eval')
            ckpt_path = '../ckpt/crop_8_28/a_very_robust_model_crop9_28_earlydrop/checkpoint-35002'
            xent_img, acc_img = get_landscape(model, ckpt_path)
            np.save('cnat',{'xent_img':xent_img, 'acc_img':acc_img})

            ckpt_path = '../ckpt/crop8_nat_models/crop9_28_nat/checkpoint-28001'
            xent_img, acc_img = get_landscape(model, ckpt_path)
            np.save('cadv',{'xent_img':xent_img, 'acc_img':acc_img})

        import numpy as np

dir = './sign_grad_noclip_pix'
mnat = np.load(dir + '/mnat.npy').item()
madv = np.load(dir + '/madv.npy').item()
cnat = np.load(dir + '/cnat.npy').item()
cadv = np.load(dir + '/cadv.npy').item()
n = 25
for i in range(10):
    plotting([mnat['xent_img'][i, 0, :n, :n], madv['xent_img'][i, 0, :n, :n], cnat['xent_img'][i, 0, :n, :n],
              cadv['xent_img'][i, 0, :n, :n]])
    # plotting([mnat['acc_img'][i, 0, :n, :n], madv['acc_img'][i, 0, :n, :n], cnat['acc_img'][i, 0, :n, :n],
    #         cadv['acc_img'][i, 0, :n, :n]])
    plt.show()