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

tf.app.flags.DEFINE_string('train_dir', 'dvector_model_test/train_logs', 'train model store path')
tf.app.flags.DEFINE_string('train_file', os.path.join(os.getcwd(), 'train.h5'), 'train data file path')
tf.app.flags.DEFINE_string('check_point', '9600', 'train model store step')
tf.app.flags.DEFINE_string('num_speakers', 1972, 'equal to number of speaker in train dataset, which is used to restore model')


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

tf.app.flags.DEFINE_integer('num_small_test_batch_size', 32,
                            'number of batch in small dataset test')
tf.app.flags.DEFINE_bool('small_dataset_test', True,
                         'whether doing small dataset test.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer('num_epochs', 1000,
                            'The number of epochs for training')

tf.app.flags.DEFINE_integer('lstm_hidden_units', 1024, 'number of lstm cell hidden untis')
tf.app.flags.DEFINE_integer('lstm_num_layers', 3, 'number of lstm layers')
tf.app.flags.DEFINE_integer('feature_dim', 40, 'dim of feature')
tf.app.flags.DEFINE_integer('left_context', 4, 'number of left context')
tf.app.flags.DEFINE_integer('right_context', 4, 'number of right context')
tf.app.flags.DEFINE_integer('lstm_time', 120, 'lstm max time')
tf.app.flags.DEFINE_integer('cnn_num_filter', 4, 'define number of cnn filter, lstm_time must be divided exactly of this number')
tf.app.flags.DEFINE_integer('cnn_shift_time', 3, 'cnn depth stride time')
tf.app.flags.DEFINE_integer('dvector_dim', 600, 'dvector dim')
tf.app.flags.DEFINE_float('cnn_dropout', 0.75, 'probability to keep units in cnn')
tf.app.flags.DEFINE_float('lstm_in_dropout', 0.75, 'probability to keep input units in lstm')
tf.app.flags.DEFINE_float('lstm_out_dropout', 0.75, 'probability to keep output units in lstm')
tf.app.flags.DEFINE_bool('training', False, 'training state')
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



file_train = tb.open_file(FLAGS.train_file, 'r')
utts = file_train.root.utterance[:]
speakerinfo = file_train.root.speakerinfo[:]
utt_features, utt_labels, utt_num_samples = split_data_into_utt(utts, speakerinfo, FLAGS)
utt_features = utt_features[0:200]
utt_labels = utt_labels[0:200]
utt_num_samples = utt_num_samples[0:200]
print("num samples:", np.sum(utt_num_samples))
num_speakers = FLAGS.num_speakers

#model file path
check_point_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'train_logs-' + FLAGS.check_point)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        neighbor_dim = FLAGS.left_context + FLAGS.right_context + 1
        lstm_time = FLAGS.lstm_time
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, lstm_time, neighbor_dim, FLAGS.feature_dim])
                labels = tf.placeholder(tf.int32, [FLAGS.batch_size])
                label_onehot = tf.one_hot(labels - 1, depth=num_speakers)
                logits, dvector = prepare_model(inputs, num_speakers, FLAGS)
                judge = tf.argmax(logits, 1)
                true_judge = tf.argmax(label_onehot, 1)
                softmax_result = tf.nn.softmax(logits)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':1})) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
#        sess.run(tf.local_variables_initializer())
        saver.restore(sess, check_point_dir)
        global utt_features, utt_labels, utt_num_samples
        #  utt_features, utt_labels, utt_num_samples = shuffle_data_label(utt_features, utt_labels, utt_num_samples)
        test_features = utt_features[0:20]
        test_labels = utt_labels[0:20]
        test_num_samples = utt_num_samples[0:20]
        print("test_labels:", test_labels)

        # make test data dvectors, key as speakerid, value as dvector
        test_dvectors = {}
        for index in range(len(test_labels)):
            speaker_label = test_labels[index]
            speaker_dvectors = []
            for sample in range(test_num_samples[index]):
                sample_data, sample_label, c_utt, _ = reduce_batch_data(test_features, test_labels, test_num_samples, FLAGS, index, sample)
                if c_utt < 0:
                    break
                spk_dvector, out_label, out_true_label = sess.run([dvector, judge, true_judge], feed_dict={inputs: sample_data, labels:sample_label})
                spk_dvector = np.reshape(spk_dvector, [FLAGS.dvector_dim])
                spk_dvector = dvector_normalize_length(spk_dvector, FLAGS)
                speaker_dvectors.append(spk_dvector)
                print("program out labels:", out_label + 1)
                print("true lables:", out_true_label + 1)
            #speaker_label = "speakerid"_"utt_index"
            speaker_label = str(test_labels[index]) + '_' + str(index)
            speaker_average_dvector = np.mean(speaker_dvectors, axis=0)

            print("finish test speaker [%s] utt" % (speaker_label))
            test_dvectors[speaker_label] = speaker_average_dvector

        #  print("test dvectors:", test_dvectors )
        print("test dvectors length:", len(test_dvectors))
        eer, eer_th = compute_eer(test_dvectors, test_dvectors)
        print("eer: %.4f, threshold: %.4f" % (eer, eer_th))



if __name__ == "__main__":
    tf.app.run()




