#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tables as tb
import tensorflow as tf
import numpy as np
from utils.reduce_data import *
from model.model import *
from utils.tools import *
import os

tf.app.flags.DEFINE_string('train_dir', 'dvector_model/train_logs', 'train model store path')


tf.app.flags.DEFINE_float('learning_rate', 0.0005,
                          'Initial learning rate')

tf.app.flags.DEFINE_float('end_learning_rate', 0.00005,
                          'The minimal end learning rate')

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                           'Specifies how the learning rate is decayed. One of "fixed", "exponential", "polynomial"')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                           'Learning decay factor')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'Specifies the optimizer format')
tf.app.flags.DEFINE_float('momentum', 0.5, 'Specifies the momentum param')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'The number of samples in each batch. To simulate shuffling input data ')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', 10000,
                            'The number of epochs for training')

tf.app.flags.DEFINE_integer('lstm_hidden_units', 1024, 'number of lstm cell hidden untis')
tf.app.flags.DEFINE_integer('lstm_num_layers', 3, 'number of lstm layers')
tf.app.flags.DEFINE_integer('feature_dim', 40, 'dim of feature')
tf.app.flags.DEFINE_integer('left_context', 4, 'number of left context')
tf.app.flags.DEFINE_integer('right_context', 4, 'number of right context')
tf.app.flags.DEFINE_integer('lstm_time', 30, 'lstm max time')
tf.app.flags.DEFINE_integer('dvector_dim', 600, 'dvector dim')
tf.app.flags.DEFINE_float('cnn_dropout', 0.75, 'probability to keep units in cnn')
tf.app.flags.DEFINE_float('lstm_in_dropout', 0.5, 'probability to keep input units in lstm')
tf.app.flags.DEFINE_float('lstm_out_dropout', 0.5, 'probability to keep output units in lstm')
tf.app.flags.DEFINE_bool('training', True, 'training state')


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


model_path = FLAGS.train_dir
if not os.path.exists(model_path):
    os.makedirs(model_path)
else:
    empty_dir(model_path)



file_train = tb.open_file(os.path.join(os.getcwd(), 'train.h5'), 'r')
utts = file_train.root.utterance[:]
speakerinfo = file_train.root.speakerinfo[:]
utt_features, utt_labels, utt_num_samples = split_data_into_utt(utts, speakerinfo, FLAGS)
num_speakers = np.max(speakerinfo[:, 0])
print("utts feature length:", len(utt_features), ", utts labels length:", len(utt_labels), ", utt num samples list length:", len(utt_num_samples))
#  time.sleep(100)
#      exit(1)

def main(_):
    global utt_features, utt_labels, utt_num_samples
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        num_samples_per_epoch = np.sum(utt_num_samples)
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)
        print("number of batches:", num_batches_per_epoch)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #  learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        boundaries = [5000, 10000, 20000, 40000, 80000, 120000]
        values = [0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00003, 0.00001]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        neighbor_dim = FLAGS.left_context + FLAGS.right_context + 1
        lstm_time = FLAGS.lstm_time
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, lstm_time, neighbor_dim, FLAGS.feature_dim])
                labels = tf.placeholder(tf.int32, [FLAGS.batch_size])
                if FLAGS.training:
                    logits, _ = prepare_model(inputs, num_speakers, FLAGS)
                else:
                    logits, dvectors = prepare_model(inputs, num_speakers, FLAGS)
                softmax_result = tf.nn.softmax(logits)
                label_onehot = tf.one_hot(labels - 1, depth=num_speakers)
                opt = _configure_optimizer(learning_rate)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_onehot, name='loss')
                with tf.name_scope('loss'):
                    loss = tf.reduce_mean(cross_entropy)

                with tf.name_scope('result_print'):
                    judge = tf.argmax(logits, 1)
                    true_judge = tf.argmax(label_onehot, 1)
                    prob = tf.nn.softmax(logits)

                with tf.name_scope('train_op'):
                    train_op = opt.minimize(loss)
                with tf.name_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label_onehot, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        summaries.add(tf.summary.scalar('global_step', global_step))
        summaries.add(tf.summary.scalar('eval/Loss', loss))
        summaries.add(tf.summary.scalar('accuracy', accuracy))
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        summary_merged = tf.summary.merge_all()

    #  check_point_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'train_logs-32000')
    config = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':1})
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(model_path, graph=graph)
        sess.run(tf.global_variables_initializer())
        #  sess.run(tf.local_variables_initializer())
        #  saver.restore(sess, check_point_dir)
        step = 0
        #  print(batch_datas[:])
        #  print(batch_labels[:])
        #  print("utts_train:", utts_train)
        #  exit(1)

        # small data for test
        #  utt_features, utt_labels, utt_num_samples = shuffle_data_label(utt_features, utt_labels, utt_num_samples)
        #  advance_batch_datas = []
        #  advance_batch_labels = []
        #  c_utt_index = 0
        #  c_utt_used_samples = 0
        #  for _ in range(400):
        #      batch_datas, batch_labels, c_utt_index, c_utt_used_samples = reduce_batch_data(utt_features, utt_labels, utt_num_samples, FLAGS, c_utt_index, c_utt_used_samples)
        #      advance_batch_datas.extend(batch_datas)
        #      advance_batch_labels.extend(batch_labels)
        #  advance_batch_datas, advance_batch_labels, _ = shuffle_data_label(advance_batch_datas, advance_batch_labels)

        last_loss = -1.0
        for epoch in range(FLAGS.num_epochs):
            num_advance_batch_samples = 100
            c_utt_index = 0
            c_utt_used_samples = 0
            utt_features, utt_labels, utt_num_samples = shuffle_data_label(utt_features, utt_labels, utt_num_samples)
            advance_batch_datas = []
            advance_batch_labels = []
            for batch_num in range(num_batches_per_epoch):
                step += 1
                if len(advance_batch_labels) < FLAGS.batch_size and c_utt_index >= 0:
                    for _ in range(num_advance_batch_samples):
                        batch_datas, batch_labels, c_utt_index, c_utt_used_samples = reduce_batch_data(utt_features, utt_labels, utt_num_samples, FLAGS, c_utt_index, c_utt_used_samples)
                        advance_batch_datas.extend(batch_datas)
                        advance_batch_labels.extend(batch_labels)
                        if c_utt_index < 0:
                            break
                    advance_batch_datas, advance_batch_labels, _ = shuffle_data_label(advance_batch_datas, advance_batch_labels)
                    print("advance_batch_labels:", advance_batch_labels)
                if c_utt_index < 0:
                    break

                #  start_idx = batch_num * FLAGS.batch_size
                #  end_idx = (batch_num + 1) * FLAGS.batch_size
                batch_datas, batch_labels = advance_batch_datas[0:FLAGS.batch_size], advance_batch_labels[0:FLAGS.batch_size]
                #  batch_datas = advance_batch_datas[start_idx: end_idx]
                #  batch_labels = advance_batch_labels[start_idx: end_idx]
                #  print("batch label:", batch_labels)
                _, loss_value, train_accuracy, summary, out_judge, out_true_judge, out_prob, out_learning_rate, softmax_out = sess.run([train_op, loss, accuracy, summary_merged, judge, true_judge, prob, learning_rate, softmax_result],
                        feed_dict={inputs: batch_datas, labels: batch_labels, global_step: step})
                advance_batch_datas = advance_batch_datas[FLAGS.batch_size:]
                advance_batch_labels = advance_batch_labels[FLAGS.batch_size:]
                summary_writer.add_summary(summary, step)
                #  if batch_num % 100 == 0:
                print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", Minibatch Loss=" + "{:.4f}".format(loss_value) + \
                        ", TRAIN ACCURACY=" + "{:.3f}".format(100 * train_accuracy))
                #  print(batch_datas)
                #  print(batch_labels)
                #  print("current utt index:", c_utt_index)
                #  print("current utt used samples:", c_utt_used_samples)
                print("max softmax result:", np.max(softmax_out))
                print("program out labels:", out_judge + 1)
                print("true lables:", out_true_judge + 1)
                print("learning rate:", out_learning_rate)
                if last_loss < 0:
                    last_loss = loss_value
                else:
                    if float(10 * last_loss) < float(loss_value) or float(last_loss) > 1000000.0:
                        print("loss increase in error state")
                        exit(1)
                    else:
                        last_loss = loss_value
                #  print(out_prob)
                #      print("pred 0 is ", pred_res[0,:], pred_res_in[0])
                #      print("pred prob:", pred_out_prob[0])
                #  exit(1)
            saver.save(sess, model_path, global_step=step)

if __name__ == "__main__":
    tf.app.run()




