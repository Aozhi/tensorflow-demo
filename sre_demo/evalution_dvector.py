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

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'The number of samples in each batch. To simulate shuffling input data ')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', 1000,
                            'The number of epochs for training')

tf.app.flags.DEFINE_integer('lstm_hidden_units', 1024, 'number of lstm cell hidden untis')
tf.app.flags.DEFINE_integer('lstm_num_layers', 3, 'number of lstm layers')
tf.app.flags.DEFINE_integer('feature_dim', 40, 'dim of feature')
tf.app.flags.DEFINE_integer('left_context', 4, 'number of left context')
tf.app.flags.DEFINE_integer('right_context', 4, 'number of right context')
tf.app.flags.DEFINE_integer('lstm_time', 30, 'lstm max time')
tf.app.flags.DEFINE_integer('dvector_dim', 600, 'dvector dim')
tf.app.flags.DEFINE_float('cnn_dropout', 0.75, 'probability to keep units in cnn')
tf.app.flags.DEFINE_float('lstm_in_dropout', 0.75, 'probability to keep input units in lstm')
tf.app.flags.DEFINE_float('lstm_out_dropout', 0.75, 'probability to keep output units in lstm')
tf.app.flags.DEFINE_bool('training', False, 'training state')


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



file_enroll = tb.open_file(os.path.join(os.getcwd(), 'enroll.h5'), 'r')
file_test = tb.open_file(os.path.join(os.getcwd(), 'test.h5'), 'r')
enroll_utts = file_enroll.root.utterance[:]
enroll_speakerinfo = file_enroll.root.speakerinfo[:]
enroll_features, enroll_labels, enroll_num_samples = split_data_into_utt(enroll_utts, enroll_speakerinfo, FLAGS)
test_utts = file_test.root.utterance[:]
test_speakerinfo = file_test.root.speakerinfo[:]
test_features, test_labels, test_num_samples = split_data_into_utt(test_utts, test_speakerinfo, FLAGS)

#model file path
check_point_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'train_logs-32000')

def main(_):
    global utt_features, utt_labels, utt_num_samples
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        num_samples_per_epoch = np.sum(utt_num_samples)
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)
        print("number of batches:", num_batches_per_epoch)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        neighbor_dim = FLAGS.left_context + FLAGS.right_context + 1
        lstm_time = FLAGS.lstm_time
        #  cnn_inputs = tf.placeholder(tf.float32, [30, neighbor_dim, FLAGS.feature_dim, 1])
        #  cnn = get_cnn_net(cnn_inputs, FLAGS)
        #  print("cnn shape:", cnn.shape)
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, lstm_time, neighbor_dim, FLAGS.feature_dim])
                labels = tf.placeholder(tf.int32, [FLAGS.batch_size])
                if FLAGS.training:
                    logits, _ = prepare_model(inputs, num_speakers, FLAGS)
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
                else:
                    _, dvector = prepare_model(inputs, num_speakers, FLAGS)
        if FLAGS.training:
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))
            summaries.add(tf.summary.scalar('global_step', global_step))
            summaries.add(tf.summary.scalar('eval/Loss', loss))
            summaries.add(tf.summary.scalar('accuracy', accuracy))
            summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            summary_op = tf.summary.merge(list(summaries), name='summary_op')
            summary_merged = tf.summary.merge_all()


    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':1})) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, check_point_dir)
        step = 0

        # make enroll data dvectors, key as speakerid, value as dvector
        enroll_dvectors = {}
        for index in range(len(enroll_labels)):
            speaker_label = enroll_labels[index]
            speaker_dvectors = []
            for sample in range(enroll_num_samples[index]):
                sample_data, sample_label, _, _ = reduce_batch_data(enroll_features, enroll_labels, enroll_num_samples, FLAGS, index, sample)
                if sample_label[0] != speaker_label:
                    break
                spk_dvector = sess.run([dvector], feed_dict={inputs: sample_data, labels:sample_label})
                spk_dvector = dvector_normalize_length(spk_dvector, FLAGS)
                speaker_dvectors.append(spk_dvector)
            # use mean of dvectors as speaker dvector
            speaker_average_dvector = np.mean(speaker_dvectors, axis=0)
            if speaker_label in enroll_dvectors.keys():
                enroll_dvectors[speaker_label] = np.mean([enroll_dvectors[speaker_label], speaker_average_dvector], axis=0)
            else:
                enroll_dvectors[speaker_label] = speaker_average_dvector

        test_dvectors = {}
        for index in range(len(test_labels)):
            speaker_label = test_labels[index]
            speaker_dvectors = []
            for sample in range(test_num_samples[index]):
                sample_data, sample_label, _, _ = reduce_batch_data(test_features, test_labels, test_num_samples, FLAGS, index, sample)
                if sample_label != speaker_label:
                    break
                spk_dvector = sess.run([dvector], feed_dict={inputs: sample_data, labels:sample_label})
                spk_dvector = dvector_normalize_length(spk_dvector, FLAGS)
                speaker_dvectors.append(spk_dvector)
            # use mean of dvectors as speaker dvector
            speaker_average_dvector = np.mean(speaker_dvectors, axis=0)
            if speaker_label in test_dvectors.keys():
                test_dvectors[speaker_label] = np.mean([test_dvectors[speaker_label], speaker_average_dvector], axis=0)
            else:
                test_dvectors[speaker_label] = speaker_average_dvector

        eer, eer_th = compute_eer(enroll_dvectors, test_dvectors)
        print("eer: %.2f, threshold: %.2f" % (eer, eer_th))



if __name__ == "__main__":
    tf.app.run()




