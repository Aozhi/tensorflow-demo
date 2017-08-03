#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tables as tb
import tensorflow as tf
import numpy as np
import os

def PReLU(inputs, scope):
    alphas = tf.get_variable(scope, inputs.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    return tf.nn.relu(inputs) + alphas * (inputs - abs(inputs)) * 0.5



def my_conv2d(FLAGS, x, W, b, scope, strides = 1, padding = "VALID"):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    if FLAGS.batch_norm:
        batch_norm_scope = scope + '_bn'
        print("using batch norm:" + batch_norm_scope)
        x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=FLAGS.training, scope=batch_norm_scope)
    return PReLU(x, scope)

def my_conv3d(FLAGS, x, W, b, scope, strides = [1, 1, 1, 1, 1], padding = "VALID"):
    x = tf.nn.conv3d(x, W, strides=strides, padding=padding)
    x = tf.nn.bias_add(x, b)
    if FLAGS.batch_norm:
        batch_norm_scope = scope + '_bn'
        print("using batch norm:" + batch_norm_scope)
        x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=FLAGS.training, scope=batch_norm_scope)
    return PReLU(x, scope)

def weight_variable(shape, name='weights'):
    initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def bias_variable(shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def get_cnn_net(inputs, cnn_scope, reuse_symbol, FLAGS):
    """
    # inputs = [batch_size * lstm_time, neighbor_dim, feature_dim, 1]
    """
    with tf.variable_scope(cnn_scope, reuse=reuse_symbol) as scope:
        if int(inputs.shape[0]) != int(FLAGS.batch_size):
            print("cnn inputs shape error in lstm_time:", inputs.shape)
            exit(1)
        # CNN define
        num_inchannel = FLAGS.lstm_time / FLAGS.cnn_num_filter
        weights = {
            #'wc1': weight_variable([5, 5, FLAGS.lstm_time, 128], 'wc1'),
            'wc1': weight_variable([5, 5, num_inchannel, 128], 'wc1'),
            'wc2': weight_variable([1, 3, 128, 256], 'wc2'),
            'wc3': weight_variable([2, 4, 256, 512], 'wc3'),
#            'wd' : weight_variable([1 * 7 * 256, 1024], 'wd'),
        }

        biases = {
            'bc1': bias_variable([128], 'bc1'),
            'bc2': bias_variable([256], 'bc2'),
            'bc3': bias_variable([512], 'bc3'),
#            'bd' : bias_variable([1024], 'bd'),
        }
        if not reuse_symbol:
            inputs_hist = tf.summary.histogram('inputs', inputs)
            wc1_hist = tf.summary.histogram('conv1/weights', weights['wc1'])
            bc1_hist = tf.summary.histogram('conv1/biases', biases['bc1'])
            wc2_hist = tf.summary.histogram('conv2/weights', weights['wc2'])
            bc2_hist = tf.summary.histogram('conv2/biases', biases['bc2'])
            wc3_hist = tf.summary.histogram('conv3/weights', weights['wc3'])
            bc3_hist = tf.summary.histogram('conv3/biases', biases['bc3'])
            #  wd_hist = tf.summary.histogram('cnn_fc/weights', weights['wd'])
            #  bd_hist = tf.summary.histogram('cnn_fc/biases', biases['bd'])

        #conv1
        tf.to_float(inputs)
        if not reuse_symbol:
            print("cnn inputs shape:", inputs.shape)
        #Couv-1
        conv1 = my_conv2d(FLAGS, inputs, weights['wc1'], biases['bc1'], 'conv1_layer', 2)
        if not reuse_symbol:
            print("conv1 shape:", conv1.shape)
            conv1_hist = tf.summary.histogram('conv1_out', conv1)
        #max pool
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool1')
        if not reuse_symbol:
            conv1_maxpool_hist = tf.summary.histogram('conv1_pool_out', conv1)
            print("conv1 pool shape:", conv1.shape)
        #Conv-2
        conv2 = my_conv2d(FLAGS, conv1, weights['wc2'], biases['bc2'], 'conv2_layer', 1)
        if not reuse_symbol:
            print("conv2 shape:", conv2.shape)
            conv2_hist = tf.summary.histogram('conv2_out', conv2)
        #max pool
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME', name='max_pool2')
        if not reuse_symbol:
            conv2_maxpool_hist = tf.summary.histogram('conv2_pool_out', conv2)
            print("conv2 pool shape:", conv2.shape)
        conv3 = my_conv2d(FLAGS, conv2, weights['wc3'], biases['bc3'], 'conv3_layer', 1)
        print("conv3 shape:", conv3.shape)
        conv3 = tf.reshape(conv3, [FLAGS.batch_size, 512])
        if not reuse_symbol:
            conv3_hist = tf.summary.histogram('fc_out', conv3)
        return conv3


def get_cnn_net3d(inputs, cnn_scope, reuse_symbol, FLAGS):
    """
    # inputs = [batch_size , lstm_time, neighbor_dim, feature_dim, 1]
    """
    with tf.variable_scope(cnn_scope, reuse=reuse_symbol) as scope:
        if int(inputs.shape[0]) != int(FLAGS.batch_size):
            print("cnn inputs shape error in lstm_time:", inputs.shape)
            exit(1)
        # CNN define
        num_inchannel = FLAGS.lstm_time / FLAGS.cnn_num_filter
        weights = {
            #'wc1': weight_variable([5, 5, FLAGS.lstm_time, 128], 'wc1'),
            'wc1': weight_variable([1, 5, 5, 1, 128], 'wc1'),
            'wc2': weight_variable([1, 1, 3, 128, 256], 'wc2'),
            'wc3': weight_variable([1, 2, 4, 256, 512], 'wc3'),
#            'wd' : weight_variable([1 * 7 * 256, 1024], 'wd'),
        }

        biases = {
            'bc1': bias_variable([128], 'bc1'),
            'bc2': bias_variable([256], 'bc2'),
            'bc3': bias_variable([512], 'bc3'),
#            'bd' : bias_variable([1024], 'bd'),
        }

        strides = {
            'stride1': [1, FLAGS.cnn_shift_time, 2, 2, 1],
            'stride2': [1, 1, 1, 1, 1],
            'stride2': [1, 1, 1, 1, 1],
        }
        if not reuse_symbol:
            inputs_hist = tf.summary.histogram('inputs', inputs)
            wc1_hist = tf.summary.histogram('conv1/weights', weights['wc1'])
            bc1_hist = tf.summary.histogram('conv1/biases', biases['bc1'])
            wc2_hist = tf.summary.histogram('conv2/weights', weights['wc2'])
            bc2_hist = tf.summary.histogram('conv2/biases', biases['bc2'])
            wc3_hist = tf.summary.histogram('conv3/weights', weights['wc3'])
            bc3_hist = tf.summary.histogram('conv3/biases', biases['bc3'])
            #  wd_hist = tf.summary.histogram('cnn_fc/weights', weights['wd'])
            #  bd_hist = tf.summary.histogram('cnn_fc/biases', biases['bd'])

        #conv1
        tf.to_float(inputs)
        if not reuse_symbol:
            print("cnn inputs shape:", inputs.shape)
        #Couv-1
        conv1 = my_conv3d(FLAGS, inputs, weights['wc1'], biases['bc1'], 'conv3d1_layer', strides['stride1'])
        if not reuse_symbol:
            print("conv3d1 shape:", conv1.shape)
            conv1_hist = tf.summary.histogram('conv3d1_out', conv1)
        #max pool
        conv1 = tf.nn.max_pool3d(conv1, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='max_pool3d1')
        if not reuse_symbol:
            conv1_maxpool_hist = tf.summary.histogram('conv3d1_pool_out', conv1)
            print("conv3d1 pool shape:", conv1.shape)
        #Conv-2
        conv2 = my_conv3d(FLAGS, conv1, weights['wc2'], biases['bc2'], 'conv3d2_layer')
        if not reuse_symbol:
            print("conv3d2 shape:", conv2.shape)
            conv2_hist = tf.summary.histogram('conv3d2_out', conv2)
        #max pool
        conv2 = tf.nn.max_pool3d(conv2, ksize=[1, 1, 1, 2, 1], strides=[1, 1, 1, 2, 1], padding='SAME', name='max_pool3d2')
        if not reuse_symbol:
            conv2_maxpool_hist = tf.summary.histogram('conv3d2_pool_out', conv2)
            print("conv3d2 pool shape:", conv2.shape)
        conv3 = my_conv3d(FLAGS, conv2, weights['wc3'], biases['bc3'], 'conv3d3_layer')
        print("conv3d3 shape:", conv3.shape)
        conv3 = tf.reshape(conv3, [FLAGS.batch_size, int(FLAGS.lstm_time / FLAGS.cnn_shift_time), 512])
        if not reuse_symbol:
            conv3_hist = tf.summary.histogram('conv3d3_out', conv3)
        return conv3




def get_lstm_net(inputs, lstm_scope, reuse_symbol, FLAGS):
    #inputs shape = [batch_size, lstm_time, cnn_out]
    #max_time = left_context + 1(current_frame) + right_context
    #define lstm
    with tf.variable_scope(lstm_scope, reuse=reuse_symbol) as scope:
        if inputs.shape[0] != FLAGS.batch_size:
            print("lstm inputs error shape in lstm_time:", inputs.shape)
            exit(1)
        weights = {
            'wd': weight_variable([1024, FLAGS.dvector_dim], 'wd'),
            #  'fc': tf.get_variable("fc", tf.random_normal([1024, 600])),
        }
        biases = {
            'bd': bias_variable([FLAGS.dvector_dim], 'bd'),
            #  'fc': tf.get_variable("fc", tf.random_normal([600]))
        }
        if not reuse_symbol:
            inputs_hist = tf.summary.histogram('inputs', inputs)
            w_hist = tf.summary.histogram('lstm_fc/weights', weights['wd'])
            b_hist = tf.summary.histogram('lstm_fc/biases', biases['bd'])
        tf.to_float(inputs)
        if not reuse_symbol:
            print("lstm inputs shape:", inputs.shape)
        lstm_cells = []
        for _ in range(FLAGS.lstm_num_layers):
            lstm_cell = tf.contrib.rnn.GRUCell(FLAGS.lstm_hidden_units)
            lstm_cells.append(lstm_cell)
        stack_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        initial_state = stack_lstm.zero_state(FLAGS.batch_size, tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(stack_lstm, inputs, dtype=tf.float32, initial_state=initial_state)
        outputs = tf.transpose(outputs, [1,0,2])
        last = outputs[-1]
        if FLAGS.batch_norm:
            last = tf.contrib.layers.batch_norm(last, decay=0.9, center=True, scale=True, updates_collections=None, is_training=FLAGS.training, scope='lstm_output_bn')
            print("using batch norm: lstm_output_bn")
        last = PReLU(last, 'LSTM_out')
        if not reuse_symbol:
            print("lstm last shape:", last.shape)
            last_hist = tf.summary.histogram('lstm_out', last)
        fc = tf.add(tf.matmul(last, weights['wd']), biases['bd'])
        fc = PReLU(fc, 'lstm_fc')
        if not reuse_symbol:
            print("lstm out shape:", fc.shape)
        #Add hitogram summary
            fc_hist = tf.summary.histogram('fc_out', fc)
        return fc

def prepare_model(inputs, num_speakers, FLAGS):
    #inputs shape = [batch_size, lstm_time, neighbor_dim, feature_dim]
    batch_size = int(FLAGS.batch_size)
    lstm_time = int(FLAGS.lstm_time)
    neighbor_dim = int(FLAGS.left_context + FLAGS.right_context + 1)
    feature_dim = int(FLAGS.feature_dim)
    if int(batch_size) != int(inputs.shape[0]):
        print("error inputs shape[0] != batch_size (%d)" % (batch_size), inputs.shape)
        exit(1)
    with tf.variable_scope('sre_cnn_net') as scope:
        print("inputs shape:", inputs.shape)
        #make CNN input
        #  print("cnn_num_filter:", FLAGS.cnn_num_filter)
        #  cnn_inputs = tf.split(inputs, int(FLAGS.cnn_num_filter), axis=1)
        #  cnn_outputs = []
        #  for i in range(FLAGS.cnn_num_filter):
        #      #CNN input shape = [batch_size, neighbor_dim, feature_dim, inchannel(lstm_time / cnn_num_filter)]
        #      cnn_input = cnn_inputs[i]
        #      cnn_input = tf.transpose(cnn_input, [0, 2, 3, 1])
        #      cnn_output = get_cnn_net(cnn_input, 'cnn_' + str(i), False, FLAGS)
        #      cnn_outputs.append(cnn_output)
        #  cnn_outputs = tf.stack(cnn_outputs)
        cnn_inputs = tf.reshape(inputs, [batch_size, lstm_time, neighbor_dim, feature_dim, 1])
        cnn_outputs = get_cnn_net3d(cnn_inputs, 'cnn3d', False, FLAGS)
        print("cnn_outputs shape:", cnn_outputs.shape)
    with tf.variable_scope('sre_lstm_net') as scope:
        weights = weight_variable([FLAGS.dvector_dim, num_speakers], 'out_weights')
        biases = bias_variable([num_speakers], 'out_biases')
        w_hist = tf.summary.histogram('dvector_out/weights', weights)
        b_hist = tf.summary.histogram('dvector_out/biases', biases)
        #  lstm_inputs = tf.transpose(cnn_outputs, [1, 0, 2])
        #  print("lstm_inputs shape:", lstm_inputs.shape)
        #  out = get_lstm_net(lstm_inputs, 'lstm', False, FLAGS)
        out = get_lstm_net(cnn_outputs, 'lstm', False, FLAGS)
        print("out shape:", out.shape)
        logits = tf.add(tf.matmul(out, weights), biases)
        print("logits shape:", logits.shape)
        logits_hist = tf.summary.histogram('logits', logits)
        return logits, out
        #  if FLAGS.training:
        #      print("out shape:", out.shape)
        #      logits = tf.add(tf.matmul(out, weights), biases)
        #      print("logits shape:", logits.shape)
        #      logits_hist = tf.summary.histogram('logits', logits)
        #      return logits, out
        #  else:
        #      dvector = tf.reshape(out, [FLAGS.dvector_dim])
        #      logits = tf.add(tf.matmul(out, weights), biases)
        #      print("dvector shape:", dvector.shape)
        #      #  return logits, dvector
        #      return logits, out




