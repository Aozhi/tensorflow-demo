#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
import random

def neighbor_data(data, left_context, right_context, start_idx, end_idx):
    #data format: [num_frames, feature] for one utt
    #  print("param data shape:", data.shape, "left_context:", left_context, "right_context:", right_context, "start_idx:", start_idx, "end_idx:", end_idx)
    data_size = data.shape[0]
#    print("data size:", data_size)
    total_time = left_context + right_context + 1
    feature_dim = data.shape[1]
    split_datas = []
    i = start_idx
    while i < end_idx:
        if i < left_context:
            data_stack = np.tile(data[0, :], (left_context - i, 1))
            data_stack = np.row_stack((data_stack, data[0:i, :]))
        else:
            data_stack = data[i - left_context:i, :]
        data_stack = np.row_stack((data_stack, data[i:i + right_context + 1, :]))
        if i + right_context + 1 > data_size:
            data_stack = np.row_stack((data_stack, np.tile(data[-1, :], (i + right_context + 1 - data_size, 1))))
#            print("i + right_context + 1 - data_size:", i + right_context + 1 - data_size)
        if data_stack.shape != (total_time, feature_dim):
            print("data stack shape:", data_stack.shape)
            print(data_stack[-1, :])
            print(data[i + right_context, :])
            print("error when make data, data_stack shape:", data_stack.shape, "total_time:", total_time, "feature_dim:", feature_dim)
            exit(1)
        split_datas.append(data_stack)
        i += 1
    ret_data = np.array(split_datas)
    return ret_data

def split_data_into_utt(data, data_info, FLAGS):
    #data format: [num_frames, feature_dim]
    #data_info format: [num_utts, [speaker_id, num_frames_of_utt]]
    #one sample format: [lstm_time, neighbor_dim, feature_dim], neighbor_dim = left_context + right_context + 1
    #num_utt_samples = int(utt_num_frames / lstm_time)
    utt_features = []
    utt_labels = []
    utt_num_samples = []
    lstm_time = FLAGS.lstm_time
    for i in range(data_info.shape[0]):
        num_frames_before = np.sum(data_info[0:i, 1])
        num_utt_frames = data_info[i][1]
        utt_feature = data[num_frames_before:(num_frames_before + num_utt_frames), :]
        num_sample = int(num_utt_frames / lstm_time)
        utt_features.append(utt_feature)
        utt_labels.append(data_info[i, 0])
        utt_num_samples.append(num_sample)
    #  print("utt_features[0] shape:", utt_features[0].shape, ", utt_labels[0] shape:", utt_labels[0].shape, ", utt_num_samples[0]:", utt_num_samples[0])
    return utt_features, utt_labels, utt_num_samples

def reduce_batch_data(utt_features, utt_labels, utt_num_samples, FLAGS, c_utt_index, c_utt_used_samples):
    #data format: [num_frames, feature_dim]
    #data_info format: [num_utts, [speaker_id, num_frames_of_utt]]
    #batch_datas format: [batch_size, lstm_time, neighbor_dim, feature_dim]
    #batch_labels format: [batch_size, 1 (speaker_id)] for lstm labels
    left_context = FLAGS.left_context
    right_context = FLAGS.right_context
    lstm_time = FLAGS.lstm_time
    batch_size = FLAGS.batch_size
    batch_datas = []
    batch_labels = []
    while len(batch_labels) != batch_size:
        if c_utt_used_samples >= utt_num_samples[c_utt_index]:
            c_utt_index += 1
            c_utt_used_samples = 0
        if c_utt_index >= len(utt_labels):
            return np.array([]), np.array([]), -1, 0
        one_label = utt_labels[int(c_utt_index)]
        one_feature = neighbor_data(utt_features[int(c_utt_index)], left_context, right_context, int(c_utt_used_samples * lstm_time), int((c_utt_used_samples + 1) * lstm_time))
        #  print("one feature shape:", one_feature.shape)
        batch_datas.append(one_feature)
        batch_labels.append(one_label)
        c_utt_used_samples += 1
    return np.array(batch_datas), np.array(batch_labels), c_utt_index, c_utt_used_samples

def shuffle_data_label(feats, labels, samples=None):
    if len(feats) != len(labels):
        print("error dim between feats(%d) and labels(%d)"%(len(feats), len(labels)))
        exit(1)
    index_shuf = range(len(feats))
    random.shuffle(index_shuf)
    ret_feats = []
    ret_labels = []
    ret_samples = []
    for i in index_shuf:
        ret_feats.append(feats[i])
        ret_labels.append(labels[i])
        if samples != None:
            ret_samples.append(samples[i])
    return ret_feats, ret_labels, ret_samples

