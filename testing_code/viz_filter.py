import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_net_weights(ckpt_path, flag):
    from tensorflow.python import pywrap_tensorflow
    import scipy.io as sio
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    # print(reader.debug_string().decode("utf-8"))
    if flag:
        net = {
        'conv1_w': reader.get_tensor('Variable'),
        'conv1_b': reader.get_tensor('Variable_1'),
        'conv2_w': reader.get_tensor('Variable_2'),
        'conv2_b': reader.get_tensor('Variable_3'),
        'fc1_w': reader.get_tensor('Variable_4'),
        'fc1_b': reader.get_tensor('Variable_5'),
        'fc2_w': reader.get_tensor('Variable_6'),
        'fc2_b': reader.get_tensor('Variable_7'),
        }
    else:
        net = {
        'conv1_b': reader.get_tensor('classifier/conv1/biases'),
        'conv1_w': reader.get_tensor('classifier/conv1/weights'),
        'conv2_b': reader.get_tensor('classifier/conv2/biases'),
        'conv2_w': reader.get_tensor('classifier/conv2/weights'),
        'fc1_b': reader.get_tensor('classifier/fc1/biases'),
        'fc1_w': reader.get_tensor('classifier/fc1/weights'),
        'fc2_b': reader.get_tensor('classifier/fc2/biases'),
        'fc2_w': reader.get_tensor('classifier/fc2/weights'),
        }
    # sio.savemat('mnist_net',net)
    return net

def get_viz(aa):

    # fig = plt.figure()
    # w_image = np.zeros((5 * 4 + 3, 5 * 8 + 7))
    # for i in range(4):
    #     for j in range(8):
    #         w_image[5 * i+i:5 * (i + 1)+i, 5 * j+j:5 * (j + 1)+j] = aa['conv1_w'][:, :, 0, 8 * i + j]
    # plt.imshow(w_image)
    # plt.colorbar()
    # plt.show()

    # of different magnitude
    fig = plt.figure(figsize=(6, 3))
    fig.tight_layout()
    for i in range(4):
        for j in range(8):
            plt.subplot(4, 8, 8 * i + j + 1)
            plt.imshow(aa['conv1_w'][:, :, 0, 8 * i + j], vmin=0, vmax=1)
            plt.axis('off')
            # plt.colorbar()
    # plt.show()
    #
    # dd = 10
    # with tf.Session() as sess:
    #     conv2 = tf.nn.conv2d_transpose(aa['conv2_w'].transpose(3, 0, 1, 2), aa['conv1_w'], strides=[1, 2, 2, 1],
    #                                    output_shape=[64, dd, dd, 1])
    #     conv2 = sess.run(conv2)
    # w_image = np.zeros((dd * 8, dd * 8))
    # for i in range(8):
    #     for j in range(8):
    #         w_image[dd * i:dd * (i + 1), dd * j:dd * (j + 1)] = conv2[8 * i + j, :, :, 0]
    # plt.imshow(np.abs(w_image))
    # plt.colorbar()
    #
    # print('done')
    return fig

def get_activation(ckpt_path):
    # visualize activations
    plt.figure()
    aa = get_net_weights(ckpt_path)
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


if __name__ == '__main__':
    # first layer filters
    ckpt_path = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/madry/natural/checkpoint-24900'
    net = get_net_weights(ckpt_path, 1)
    fig = get_viz(net)

    ckpt_path = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/madry/secret/checkpoint-99900'
    net = get_net_weights(ckpt_path, 1)
    fig = get_viz(net)

    ckpt_path = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/crop9_20_nat_itr50k/crop_ckpt'
    net = get_net_weights(ckpt_path, 0)
    fig = get_viz(net)

    ckpt_path = '/home/hope-yao/Documents/adversarial_defense/mnist/ckpt/crop9_20_itr150k/crop_ckpt'
    net = get_net_weights(ckpt_path, 0)
    fig = get_viz(net)
    plt.show()

    print('done')


