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

def dvector_normalize_length(dvector, FLAGS):
    if int(FLAGS.dvector_dim) != int(dvector.shape[0]):
        print("dvector dim error")
        exit(1)
    l2_norm = np.sum(np.square(dvector))
    scale = l2_norm / np.sqrt(l2_norm)
    ret_dvector = dvector / scale
    return ret_dvector

def cosine_similarity(X, Y):
    if X.shape != Y.shape:
        print("error shape between X and Y, X shape:", X.shape, ", Y shape:", Y.shape)
        exit(1)
    return np.dot(X, Y)

def compute_eer(enroll_dvectors, test_dvectors):
    target_scores = []
    nontarget_scores = []
    for (enroll_k, enroll_v) in enroll_dvectors.items():
        for (test_k, test_v) in test_dvectors.items():
            score = cosine_similarity(enroll_v, test_v)
            if enroll_k == test_k:
                target_scores.append(score)
            else:
                nontarget_scores.append(score)
    target_scores.sort()
    nontarget_scores.sort(reversed=True)
    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    eer = 0.0
    eer_th = 0.0
    for i in range(0, target_size, 1):
        FR = i
        FA = 0
        threshold = target_scores[i]
        for score in nontarget_scores:
            if float(score) < float(threshold):
                break
            else:
                FA += 1
        FA_R = float(FA) / float(nontarget_size)
        FR_R = float(FR) / float(target_size)
        if abs(FA_R - FR_R) < 0.0001:
            eer = FA_R
            eer_th = threshold
            break
        if FA_R <= 0:
            break
    return eer, eer_th


