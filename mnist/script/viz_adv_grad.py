import tensorflow as tf
from model_mnist import *
from pgd_attack import *
import matplotlib.pyplot as plt

## parameters ##
batch_size=32
img_size=28
eps_bound = 0.3

def get_grad_val(x_batch, y_batch):
    input_images = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64, shape=(batch_size))
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    ## restore madry's model with nat training##
    model = Model_madry(input_images, input_label)
    attack = LinfPGDAttack(model, epsilon=eps_bound, k=40, a=0.01, random_start=True, loss_func='xent')
    saver = tf.train.Saver()
    saver.restore(sess,'../ckpt/madry/natural/checkpoint-24900')
    adv_grad = sess.run(attack.grad, {model.x_input:x_batch, model.y_input:y_batch})
    adv_grads_vanilla = adv_grad.reshape(batch_size,28,28)

    ## restore madry's model with adv training##
    saver.restore(sess,'../ckpt/madry/secret/checkpoint-99900')
    adv_grad = sess.run(attack.grad, {model.x_input:x_batch, model.y_input:y_batch})
    adv_grads_madry = adv_grad.reshape(batch_size,28,28)

    ## restore our's model with nat training##
    our_model = Model_crop(input_images, input_label)
    our_saver = tf.train.Saver(var_list=our_model.vars)
    our_attack = LinfPGDAttack(our_model, epsilon=eps_bound, k=40, a=0.01, random_start=True, loss_func='xent')
    our_saver.restore(sess,'../ckpt/crop64_20_nat_itr35k/bb_64crop_ckpt')
    adv_grad = sess.run(our_attack.grad, {model.x_input:x_batch, model.y_input:y_batch})
    adv_grads_ours_vanilla = adv_grad.reshape(batch_size,28,28)

    ## restore our's model with adv training##
    our_saver.restore(sess,'../ckpt/crop9_20_itr150k/crop_ckpt')
    adv_grad = sess.run(our_attack.grad, {model.x_input:x_batch, model.y_input:y_batch})
    adv_grads_ours = adv_grad.reshape(batch_size,28,28)

    adv_grad_dict = {'nat_img': np.squeeze(x_batch),
                     'nat_label': y_batch,
                     'adv_grads_vanilla': adv_grads_vanilla,
                     'adv_grads_madry': adv_grads_madry,
                     'adv_grads_ours_vanilla': adv_grads_ours_vanilla,
                     'adv_grads_ours': adv_grads_ours,
                     }
    return adv_grad_dict

def plot_adv_img(data, idx):
    N = 10
    fig = plt.figure(figsize=(12,6))
    for i in range(N):
        plt.subplot(5, N, i + 1)
        plt.imshow(data['nat_img'][10*idx:10*idx+10][i], cmap='gray')
        plt.axis('off')

        plt.subplot(5, N, N+i + 1)
        plt.imshow(data['adv_grads_vanilla'][10*idx:10*idx+10][i], cmap='seismic')
        plt.axis('off')

        plt.subplot(5, N, 2*N+i + 1)
        plt.imshow(data['adv_grads_madry'][10*idx:10*idx+10][i], cmap='seismic')
        plt.axis('off')

        plt.subplot(5, N, 3*N+i + 1)
        plt.imshow(data['adv_grads_ours_vanilla'][10*idx:10*idx+10][i], cmap='seismic')
        plt.axis('off')

        plt.subplot(5, N, 4 * N + i + 1)
        plt.imshow(data['adv_grads_ours'][10 * idx:10 * idx + 10][i], cmap='seismic')
        plt.axis('off')
    return fig

if __name__ == '__main__':
    import os
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_batch, y_batch = mnist.test.next_batch(batch_size)
    adv_grad_dict = get_grad_val(x_batch.reshape(batch_size,img_size,img_size,1), y_batch)

    for idx in range(3):
        fig_path = os.path.join('../asset/', 'adv_img_{}.png'.format(idx))
        fig = plot_adv_img(adv_grad_dict, idx)
        fig.savefig(fig_path)
