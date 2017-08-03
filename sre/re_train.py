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
from utils.dvector_tool import *
from model.model import *
from utils.tools import *
import os

np.set_printoptions(threshold='nan')

tf.app.flags.DEFINE_string('train_dir', 'dvector_model/train_logs', 'train model store path')

tf.app.flags.DEFINE_string('check_point', '200', 'train model store step')

tf.app.flags.DEFINE_string('train_file', os.path.join(os.getcwd(), 'train.h5'), 'train data file path')
tf.app.flags.DEFINE_string('test_file', os.path.join(os.getcwd(), 'test.h5'), 'test data file path')
tf.app.flags.DEFINE_integer('num_test_utt', 20, 'number of utterance in test.h5 dataset')

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
tf.app.flags.DEFINE_float('opt_epsilon', 0.1, 'a current good choice is 1.0 or 0.1 in ImageNet example')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('num_small_test_batch_size', 60,
                            'number of batch in small dataset test')
tf.app.flags.DEFINE_bool('small_dataset_test', False,
                         'whether doing small dataset test.')

tf.app.flags.DEFINE_integer('num_advance_batch_samples', 200,
                            'The number of batches that are shuffled advance. To simulate shuffling input data ')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', 50,
                            'The number of epochs for training')

tf.app.flags.DEFINE_integer('lstm_hidden_units', 1024, 'number of lstm cell hidden untis')
tf.app.flags.DEFINE_integer('lstm_num_layers', 3, 'number of lstm layers')
tf.app.flags.DEFINE_integer('feature_dim', 40, 'dim of feature')
tf.app.flags.DEFINE_integer('left_context', 4, 'number of left context')
tf.app.flags.DEFINE_integer('right_context', 4, 'number of right context')
tf.app.flags.DEFINE_integer('lstm_time', 120, 'lstm max time')
tf.app.flags.DEFINE_integer('cnn_num_filter', 4, 'define number of cnn filter, lstm_time must be divided exactly of this number, using in conv2d')
tf.app.flags.DEFINE_integer('cnn_shift_time', 3, 'cnn depth stride time, using in conv3d')
tf.app.flags.DEFINE_integer('dvector_dim', 600, 'dvector dim')
tf.app.flags.DEFINE_float('cnn_dropout', 0.75, 'probability to keep units in cnn')
tf.app.flags.DEFINE_float('lstm_in_dropout', 0.5, 'probability to keep input units in lstm')
tf.app.flags.DEFINE_float('lstm_out_dropout', 0.5, 'probability to keep output units in lstm')
tf.app.flags.DEFINE_bool('training', True, 'batch norm training model')
tf.app.flags.DEFINE_bool('batch_norm', True, 'doing batch normalization')


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
        optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    epsilon=FLAGS.opt_epsilon)
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


#  file_train = tb.open_file(os.path.join(os.getcwd(), 'train.h5'), 'r')
file_train = tb.open_file(FLAGS.train_file, 'r')
utts = file_train.root.utterance[:]
speakerinfo = file_train.root.speakerinfo[:]
utt_features, utt_labels, utt_num_samples = split_data_into_utt(utts, speakerinfo, FLAGS)

file_test = tb.open_file(FLAGS.test_file, 'r')
test_utts = file_test.root.utterance[:]
test_speakerinfo = file_test.root.speakerinfo[:]
test_features, test_labels, test_num_samples = split_data_into_utt(test_utts, test_speakerinfo, FLAGS)

num_test_utt = int(FLAGS.num_test_utt)
test_features, test_labels, test_num_samples = test_features[0:num_test_utt], test_labels[0:num_test_utt], test_num_samples[0:num_test_utt]

num_speakers = np.max(utt_labels)
print("utts feature length:", len(utt_features), ", utts labels length:", len(utt_labels), ", utt num samples list length:", len(utt_num_samples))
print("utt num samples:", np.sum(utt_num_samples))
#  time.sleep(100)
#      exit(1)
check_point_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'train_logs-' + FLAGS.check_point)

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
        # custom learning rate
        boundaries = [2000, 5000, 8000, 12000, 15000, 18000, 20000, 25000]
        values = [0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00003, 0.00001, 0.000005]
        # small dataset learning rate
        #  boundaries = [300, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
        #  values = [0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00003, 0.00001, 0.00005, 0.000005]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        neighbor_dim = FLAGS.left_context + FLAGS.right_context + 1
        lstm_time = FLAGS.lstm_time
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, lstm_time, neighbor_dim, FLAGS.feature_dim])
                labels = tf.placeholder(tf.int32, [FLAGS.batch_size])
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
                    train_op = opt.minimize(loss, global_step=global_step)
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

    config = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':1})
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(model_path, graph=graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, check_point_dir)

        # small data for test
        if FLAGS.small_dataset_test:
            print("It's in small dataset test")
            advance_batch_datas = []
            advance_batch_labels = []
            c_utt_index = 0
            c_utt_used_samples = 0
            for _ in range(70):
                batch_datas, batch_labels, c_utt_index, c_utt_used_samples = reduce_batch_data(utt_features, utt_labels, utt_num_samples, FLAGS, c_utt_index, c_utt_used_samples)
                advance_batch_datas.extend(batch_datas)
                advance_batch_labels.extend(batch_labels)
                if c_utt_index < 0:
                    print("out of utts")
                    break
            if len(advance_batch_datas) != len(advance_batch_labels):
                print("error dim between datas and labels")
            print("number of advance dataset:", len(advance_batch_labels))
            advance_batch_datas, advance_batch_labels, _ = shuffle_data_label(advance_batch_datas, advance_batch_labels)

            for epoch in range(FLAGS.num_epochs):
                advance_batch_datas, advance_batch_labels, _ = shuffle_data_label(advance_batch_datas, advance_batch_labels)
                for batch_num in range(FLAGS.num_small_test_batch_size):
                    start_idx = batch_num * FLAGS.batch_size
                    end_idx = (batch_num + 1) * FLAGS.batch_size
                    batch_datas = advance_batch_datas[start_idx: end_idx]
                    batch_labels = advance_batch_labels[start_idx: end_idx]
                    _, loss_value, train_accuracy, summary, out_judge, \
                    out_true_judge, out_prob, out_learning_rate, softmax_out, step, dvec = sess.run(
                        [train_op, loss, accuracy, summary_merged, \
                        judge, true_judge, prob, learning_rate, softmax_result, global_step, dvectors], \
                        feed_dict={inputs: batch_datas, labels: batch_labels})
                    summary_writer.add_summary(summary, step)
                    print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", Minibatch Loss=" + "{:.4f}".format(loss_value) + \
                        ", TRAIN ACCURACY=" + "{:.3f}".format(100 * train_accuracy))
                    print("current utt index:", c_utt_index)
                    print("current utt used samples:", c_utt_used_samples)
                    print("global step:", step)
                    #  print("softmax result:", softmax_out[:])
                    print("mean softmax result:", np.mean(np.max(softmax_out, axis=1)))
                    print("program out labels:", out_judge + 1)
                    print("true lables:", out_true_judge + 1)
                    print("learning rate:", out_learning_rate)
                    if step % 100 == 0:
                        print("dvectors:", dvec[:])
                        saver.save(sess, model_path, global_step=step)
            exit(1)

        last_loss = -1.0
        num_advance_batch_samples = FLAGS.num_advance_batch_samples
        for epoch in range(FLAGS.num_epochs):
            c_utt_index = 0
            c_utt_used_samples = 0
            utt_features, utt_labels, utt_num_samples = shuffle_data_label(utt_features, utt_labels, utt_num_samples)
            advance_batch_datas = []
            advance_batch_labels = []
            for batch_num in range(num_batches_per_epoch):
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

                batch_datas, batch_labels = advance_batch_datas[0:FLAGS.batch_size], advance_batch_labels[0:FLAGS.batch_size]
                _, loss_value, train_accuracy, summary, out_judge, \
                out_true_judge, out_prob, out_learning_rate, softmax_out, step = sess.run(
                        [train_op, loss, accuracy, summary_merged, \
                        judge, true_judge, prob, learning_rate, softmax_result, global_step], \
                        feed_dict={inputs: batch_datas, labels: batch_labels})

                advance_batch_datas = advance_batch_datas[FLAGS.batch_size:]
                advance_batch_labels = advance_batch_labels[FLAGS.batch_size:]

                summary_writer.add_summary(summary, step)
                #  if batch_num % 100 == 0:
                print("Epoch " + str(epoch + 1) + ", Minibatch " + str(batch_num + 1) + \
                        " of %d " % num_batches_per_epoch + ", Minibatch Loss=" + "{:.4f}".format(loss_value) + \
                        ", TRAIN ACCURACY=" + "{:.3f}".format(100 * train_accuracy))
                #  print(batch_datas)
                #  print(batch_labels)
                print("current utt index:", c_utt_index)
                print("current utt used samples:", c_utt_used_samples)
                print("global step:", step)
                #  print("softmax result:", softmax_out[:])
                print("mean softmax result:", np.mean(np.max(softmax_out, axis=1)))
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
                if step % 500 == 0:
                    saver.save(sess, model_path, global_step=step)
                    # testing small test dataset
                    test_utt_index = 0
                    test_utt_used_samples = 0
                    test_batch_datas = []
                    test_batch_labels = []
                    test_dvectors = {}
                    for _ in range(int(np.sum(test_num_samples) / FLAGS.batch_size)):
                        batch_datas, batch_labels, test_utt_index, test_utt_used_samples = reduce_batch_data(test_features, test_labels, test_num_samples, FLAGS, test_utt_index, test_utt_used_samples)
                        test_batch_datas.extend(batch_datas)
                        test_batch_labels.extend(batch_labels)
                        if test_utt_index < 0:
                            break
                    print("test batch data length:", len(test_batch_labels))
                    for index in range(int(len(batch_labels) / FLAGS.batch_size)):
                        start_idx = index * FLAGS.batch_size
                        end_idx = (index + 1) * FLAGS.batch_size
                        sample_data = test_batch_datas[start_idx: end_idx]
                        sample_label = test_batch_labels[start_idx: end_idx]
                        spk_dvectors = sess.run([dvectors], feed_dict={inputs: sample_data, labels:sample_label})[0]
                        for i in range(spk_dvectors.shape[0]):
                            spk_dvector = dvector_normalize_length(spk_dvectors[i], FLAGS)
                            speaker_label = str(sample_label[i]) + '_' + str(i) + '_' + str(index)
                            test_dvectors[speaker_label] = spk_dvector
                    eer, eer_th = compute_eer(test_dvectors, test_dvectors)
                    print("eer: %.2f, threshold: %.2f" % (eer, eer_th))

if __name__ == "__main__":
    tf.app.run()




