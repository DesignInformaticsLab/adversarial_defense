import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

plt.figure()
aa = sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/secret/mnist_net_madry.mat')
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_1crop_ckpt/itr160000/mnist_net_crop1.mat')
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_4crop_ckpt/itr250000/mnist_net_crop4.mat')
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_9crop_ckpt/itr150000/mnist_net_crop9.mat')
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_20crop_ckpt/itr100000/mnist_net_crop20.mat')
dd = 10
with tf.Session() as sess:
    conv2 = tf.nn.conv2d_transpose(aa['conv2_w'].transpose(3, 0, 1, 2), aa['conv1_w'], strides=[1, 2, 2, 1],
                                   output_shape=[64, dd, dd, 1])
    conv2 = sess.run(conv2)
w_image = np.zeros((dd * 8, dd * 8))
for i in range(8):
    for j in range(8):
        w_image[dd * i:dd * (i + 1), dd * j:dd * (j + 1)] = conv2[8 * i + j, :, :, 0]
plt.imshow(np.abs(w_image))
plt.colorbar()

w_image = np.zeros((5 * 4, 5 * 8))
for i in range(4):
    for j in range(8):
        w_image[5 * i:5 * (i + 1), 5 * j:5 * (j + 1)] = aa['conv1_w'][:, :, 0, 8 * i + j]
plt.imshow(w_image)
plt.colorbar()

# of different magnitude
for i in range(4):
    for j in range(8):
        plt.subplot(4, 8, 8 * i + j + 1)
        plt.imshow(aa['conv1_w'][:, :, 0, 8 * i + j])
        plt.axis('off')
        plt.colorbar()

# visualize activations
plt.figure()
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/secret/mnist_net_madry.mat')
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_1crop_ckpt/itr160000/mnist_net_crop1.mat')
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_4crop_ckpt/itr250000/mnist_net_crop4.mat')
aa = sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_9crop_ckpt/itr150000/mnist_net_crop9.mat')
# aa=sio.loadmat('/home/hope-yao/Documents/robust_attention/mnist_ckpt/adv_train_20crop_ckpt/itr100000/mnist_net_crop20.mat')
num_eval_examples = 10000
batch_size = 16
img_size = 28
if 1:
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    img = mnist.test.images[:10].reshape(10, 28, 28, 1)
else:
    data = np.load('/home/hope-yao/Documents/robust_attention/adversary_data/mnist_adversary_noatt.npy').item()
    img = np.asarray(data['x'][:10], 'float32')

with tf.Session() as sess:
    act1 = tf.nn.conv2d(img, aa['conv1_w'], strides=[1, 1, 1, 1], padding='SAME') + aa['conv1_b']
    act1 = tf.nn.relu(act1)
    act1 = tf.layers.max_pooling2d(act1, pool_size=[2, 2], strides=[2, 2])
    act2 = tf.nn.conv2d(act1, aa['conv2_w'], strides=[1, 1, 1, 1], padding='SAME') + aa['conv2_b']
    act2 = tf.nn.relu(act2)
    act2 = tf.layers.max_pooling2d(act2, pool_size=[2, 2], strides=[2, 2])
    act = sess.run(act1)
dd = 14
w_image = np.zeros((dd * 4, dd * 8))
idx = 0
for i in range(4):
    for j in range(8):
        w_image[dd * i:dd * (i + 1), dd * j:dd * (j + 1)] = act[idx, :, :, 8 * i + j]
plt.imshow(w_image, cmap='jet')  # , interpolation='bilinear')
plt.grid('off')
plt.colorbar()



