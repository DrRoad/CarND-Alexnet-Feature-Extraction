import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
from nolearn.lasagne import BatchIterator


def fully_connected(input_layer, size):
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
    return tf.nn.xw_plus_b(input_layer, weights, biases)

def load_pickled_data(file, columns):
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))

X_train, y_train = load_pickled_data('train.p', ['features', 'labels'])
nb_classes = 43

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)

tf_x_batch = tf.placeholder(tf.float32, (None, 32, 32, 3))
tf_y_batch = tf.placeholder(tf.int64, None)
tf_resized = tf.image.resize_images(tf_x_batch, [227, 227])

fc7 = AlexNet(tf_resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)

with tf.variable_scope('fc8'):
    logits = fully_connected(fc7, nb_classes)

predictions = tf.arg_max(tf.nn.softmax(logits), 1)
softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf_y_batch)
loss = tf.reduce_mean(softmax_cross_entropy)

with tf.variable_scope('fc8', reuse=True):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
        loss,
        var_list=[
            tf.get_variable('weights'),
            tf.get_variable('biases')
        ])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print('Training...')
    for epoch in range(10):
        # Train on whole randomised dataset in batches
        batch_iterator = BatchIterator(batch_size=128, shuffle=True)
        for x_batch, y_batch in batch_iterator(X_train, y_train):
            session.run([optimizer], feed_dict={
                tf_x_batch: x_batch,
                tf_y_batch: y_batch
            })

        p = []
        sce = []
        batch_iterator = BatchIterator(batch_size=128)
        for x_batch, y_batch in batch_iterator(X_valid, y_valid):
            [p_batch, sce_batch] = session.run([predictions, softmax_cross_entropy], feed_dict={
                tf_x_batch: x_batch,
                tf_y_batch: y_batch
            })
            p.extend(p_batch)
            sce.extend(sce_batch)
        p = np.array(p)
        sce = np.array(sce)
        accuracy = 100.0 * np.sum(p == y_valid) / p.shape[0]
        loss = np.mean(sce)
        print('Finished epoch {}: valid accuracy = {:.2f}%, loss = {:.4f}'.format(epoch, accuracy, loss))
