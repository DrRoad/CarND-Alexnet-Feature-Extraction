import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet


def fully_connected(input_layer, size):
    """
    Performs a single fully connected layer pass, e.g. returns `input * weights + bias`.
    """
    weights = tf.get_variable(
        'weights',
        shape=[input_layer.get_shape().as_list()[-1], size],
        initializer=tf.contrib.layers.xavier_initializer()
    )
    biases = tf.get_variable(
        'biases',
        shape=[size],
        initializer=tf.constant_initializer(0.0)
    )
    return tf.matmul(input_layer, weights) + biases


def fully_connected_relu(input_layer, size):
    return tf.nn.relu(fully_connected(input_layer, size))

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)

with tf.variable_scope('fc8'):
    fc8 = fully_connected_relu(fc7, nb_classes)

probs = tf.nn.softmax(fc8)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
