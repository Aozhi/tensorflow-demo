#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_integer('num_hidden1_units', 100,
                            'Number of hidden1 units')

tf.app.flags.DEFINE_integer('num_hidden2_units', 50,
                            'Number of hidden2 units')

tf.app.flags.DEFINE_float('learning_rate', 0.5,
                          'Initial learning rate')

tf.app.flags.DEFINE_float('end_learning_rate', 0.0001,
                          'The minimal end learning rate')

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'fixed',
                           'Specifies how the learning rate is decayed. One of "fixed", "exponential", "polynomial"')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                           'Learning decay factor')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', 100,
                            'The number of epochs for training')

FLAGS = tf.app.flags.FLAGS



def _configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized.', FLAGS.learning_rate_decay_type)


mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
images = mnist_data.train.images
labels = mnist_data.train.labels
test_images = mnist_data.test.images
test_labels = mnist_data.test.labels

def main(_):
    x = tf.placeholder(tf.float32, (None, 784))
    W = tf.Variable(tf.zeros((784, 10)))
    b = tf.Variable(tf.zeros((10)))
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, (None, 10))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    num_samples_per_epoch = labels.shape[0]
    num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)
    for epoch in range(FLAGS.num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * FLAGS.batch_size
            end_idx = (batch_num + 1) * FLAGS.batch_size
            images_train, labels_train = images[start_idx:end_idx], labels[start_idx:end_idx]
            print('labels:', labels_train)
#            print("images_train shape:", images_train.shape, ",  labels_train shape:", labels_train.shape)
            _, train_accuracy = sess.run([train_step, accuracy], feed_dict={x: images_train, y_: labels_train})
            if batch_num % 20 == 0:
                print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", TRAIN ACCURACY="  + "{:.3f}".format(100 * train_accuracy))
    print("test accuracy:" + "{:.3f}".format(100 * sess.run(accuracy, feed_dict={x:test_images, y_: test_labels})))

if __name__ == "__main__":
    tf.app.run()




