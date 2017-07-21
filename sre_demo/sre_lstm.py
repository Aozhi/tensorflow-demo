#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tables as tb
import tensorflow as tf
import numpy as np
import os

tf.app.flags.DEFINE_string('train_dir', 'lstm_model/train_logs', 'train model store path')


tf.app.flags.DEFINE_float('learning_rate', 0.0003,
                          'Initial learning rate')

tf.app.flags.DEFINE_float('end_learning_rate', 0.00003,
                          'The minimal end learning rate')

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                           'Specifies how the learning rate is decayed. One of "fixed", "exponential", "polynomial"')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                           'Learning decay factor')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'Specifies the optimizer format')

tf.app.flags.DEFINE_integer('batch_size', 64,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', 10,
                            'The number of epochs for training')

tf.app.flags.DEFINE_integer('hidden_units', 1024, 'number of lstm cell hidden untis')
tf.app.flags.DEFINE_integer('num_layers', 3, 'number of lstm layers')
tf.app.flags.DEFINE_integer('feature_dim', 40, 'dim of feature')
tf.app.flags.DEFINE_integer('left_context', 10, 'number of left context')
tf.app.flags.DEFINE_integer('right_context', 10, 'number of right context')


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


def get_lstm_model(x, weights, biases):
    lstm_cells = []
    for _ in range(FLAGS.num_layers):
        lstm_cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden_units)
        lstm_cells.append(lstm_cell)
    stack_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    outputs, _ = tf.nn.dynamic_rnn(stack_lstm, x, dtype=tf.float32)
    outputs = tf.transpose(outputs, [1,0,2])
    #  print("outputs shape:", outputs.shape, "outputs get shape 0:", outputs.get_shape()[0])
    #  last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    last = outputs[-1]
    logits = tf.matmul(last, weights) + biases
    return logits

file_train = tb.open_file('/home/hanyu/tensflow/sre-lstm/train.h5', 'r')
pdfids = file_train.root.pdfid[:].astype(np.int32)
utts = file_train.root.utterance[:]
speakerinfo = file_train.root.speakerinfo[:]
num_pdf = int(np.max(pdfids)) + 1
if int(np.sum(speakerinfo[:, 1])) != pdfids.shape[0]:
    print("error number between speakerinfo [%d] and pdf [%d]" % (int(np.sum(speakerinfo[:, 1])), pdfids.shape[0]))
    exit(1)

def reduce_lstm_data(data):
    data_size = data.shape[0]
    left_context = FLAGS.left_context
    right_context = FLAGS.right_context
    max_time = left_context + right_context + 1
    reduce_data = []
    for i in range(data_size):
        if i < left_context:
            data_stack = np.tile(data[0, :], (left_context - i, 1))
            data_stack = np.row_stack((data_stack, data[0:i, :]))
        else:
            data_stack = data[i - left_context:i, :]
        data_stack = np.row_stack((data_stack, data[i:i + right_context + 1, :]))
        if i + right_context + 1 > data_size:
            data_stack = np.row_stack((data_stack, np.tile(data[-1, :], (i + right_context + 1 - data_size,1))))
        if data_stack.shape != (max_time, FLAGS.feature_dim):
            print("error when make data %d" %(i), ", and data stack shape:", data_stack.shape)
        reduce_data.append(data_stack)
    return np.array(reduce_data)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        num_samples_per_epoch = pdfids.shape[0]
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        max_time = FLAGS.left_context + FLAGS.right_context + 1
        x = tf.placeholder(tf.float32, [None, None, FLAGS.feature_dim])
        y = tf.placeholder(tf.int32, [None])
        label_onehot = tf.one_hot(y - 1, depth=num_pdf)
        weights = tf.Variable(tf.random_normal([FLAGS.hidden_units, num_pdf]), dtype=tf.float32)
        biases = tf.Variable(tf.random_normal([num_pdf]), dtype=tf.float32)
        pred = get_lstm_model(x, weights, biases)
        #  logits, pred, loss = get_lstm_model(x, y)
        opt = _configure_optimizer(learning_rate)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_onehot, name='loss')
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('train_op'):
            train_op = opt.minimize(loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label_onehot, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

#        pred_prob = tf.nn.softmax(pred)
#        pred_res_index = tf.argmax(pred, 1)
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
                utts_origin, pdfs_train = utts[start_idx:end_idx, :], pdfids[start_idx:end_idx]
                utts_train = reduce_lstm_data(utts_origin)
                print("utts_train shape:", utts_train.shape)
                print("utts_train:", utts_train)
                exit(1)

                _, loss_value, train_accuracy, summary = sess.run([train_op, loss, accuracy, summary_op],
                                                          feed_dict={x: utts_train, y: pdfs_train})
                summary_writer.add_summary(summary, step)
                #  if batch_num % 100 == 0:
                print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", Minibatch Loss=" + "{:.4f}".format(loss_value) + \
                        ", TRAIN ACCURACY=" + "{:.3f}".format(100 * train_accuracy))
                    #  print("pred 0 is ", pred_res[0,:], pred_res_in[0])
                    #  print("pred prob:", pred_out_prob[0])
            saver.save(sess, model_path, global_step=step)
            
        print(sess.run(accuracy, feed_dict={x: test_images[:].reshape((-1, 28, 28)), y: test_labels[:]}))


if __name__ == "__main__":
    tf.app.run()




