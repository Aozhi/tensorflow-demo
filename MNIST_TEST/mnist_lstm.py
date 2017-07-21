#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

tf.app.flags.DEFINE_string('train_dir', 'lstm_model/train_logs', 'train model store path')


tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          'Initial learning rate')

tf.app.flags.DEFINE_float('end_learning_rate', 0.0001,
                          'The minimal end learning rate')

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                           'Specifies how the learning rate is decayed. One of "fixed", "exponential", "polynomial"')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                           'Learning decay factor')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'Specifies the optimizer format')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', 10,
                            'The number of epochs for training')

tf.app.flags.DEFINE_integer('hidden_units', 128, 'number of lstm cell hidden untis')


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

def _configure_optimizer(learning_rate):
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate,
                    rho=FLAGS.adadelta_rho,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
                    learning_rate,
                    initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
                    learning_rate,
                    learning_rate_power=FLAGS.ftrl_learning_rate_power,
                    initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
                    l1_regularization_strength=FLAGS.ftrl_l1,
                    l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=FLAGS.momentum,
                    name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=FLAGS.rmsprop_decay,
                    momentum=FLAGS.rmsprop_momentum,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def RNN(X, weights, biases):
    x = tf.unstack(X, 28, 1)
    print("x shape:", x[0].shape, "num x:", len(x))
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_units, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    print("num outputs:", len(outputs), "outputs shape:", outputs[0].shape)
    return tf.matmul(outputs[-1], weights) + biases


mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
images = mnist_data.train.images
labels = mnist_data.train.labels
test_images = mnist_data.test.images
test_labels = mnist_data.test.labels

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        num_samples_per_epoch = labels.shape[0]
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        x = tf.placeholder(tf.float32, [None, 28, 28])
        y = tf.placeholder(tf.float32, [None, 10])
        weights = tf.Variable(tf.random_normal([FLAGS.hidden_units, 10]), dtype=tf.float32)
        biases = tf.Variable(tf.random_normal([10]), dtype=tf.float32)
        pred = RNN(x, weights, biases)
        opt = _configure_optimizer(learning_rate)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, name='loss')
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('train_op'):
            train_op = opt.minimize(loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        pred_prob = tf.nn.softmax(pred)
        pred_res_index = tf.argmax(pred, 1)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        summaries.add(tf.summary.scalar('global_step', global_step))
        summaries.add(tf.summary.scalar('eval/Loss', loss))
        summaries.add(tf.summary.scalar('accuracy', accuracy))
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model_path = FLAGS.train_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(model_path, graph=graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = 0
        for epoch in range(FLAGS.num_epochs):
            for batch_num in range(num_batches_per_epoch):
                step += 1
                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num + 1) * FLAGS.batch_size
                images_train, labels_train = images[start_idx:end_idx, :], labels[start_idx:end_idx]
                images_train = images_train.reshape(FLAGS.batch_size, 28, 28)
                _, loss_value, train_accuracy, summary, pred_res, pred_res_in, pred_out_prob = sess.run([train_op, loss, accuracy, summary_op, pred, pred_res_index, pred_prob],
                                                          feed_dict={x: images_train, y: labels_train})
                summary_writer.add_summary(summary, step)
                if batch_num % 100 == 0:
                    print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", Minibatch Loss=" + "{:.4f}".format(loss_value) + \
                        ", TRAIN ACCURACY=" + "{:.3f}".format(100 * train_accuracy))
                    print("pred 0 is ", pred_res[0,:], pred_res_in[0])
                    print("pred prob:", pred_out_prob[0])
            saver.save(sess, model_path, global_step=step)
            
        print(sess.run(accuracy, feed_dict={x: test_images[:].reshape((-1, 28, 28)), y: test_labels[:]}))


if __name__ == "__main__":
    tf.app.run()




