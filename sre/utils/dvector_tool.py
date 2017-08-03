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
        print("dvector dim error shape:", dvector.shape)
        exit(1)
    l2_norm = np.sqrt(np.sum(np.square(dvector)))
    scale = l2_norm / np.sqrt(FLAGS.dvector_dim)
    #  print("scale:", scale)
    ret_dvector = np.divide(dvector, scale)
    return ret_dvector

def cosine_similarity(X, Y):
    if X.shape != Y.shape:
        print("error shape between X and Y, X shape:", X.shape, ", Y shape:", Y.shape)
        exit(1)
    x = np.sqrt(np.sum(np.square(X)))
    y = np.sqrt(np.sum(np.square(Y)))
    ret = np.dot(X,Y) / x / y
    return ret

def compute_eer(enroll_dvectors, test_dvectors):
    print("enroll length: %d, test length: %d" % (len(enroll_dvectors), len(test_dvectors)))
    target_scores = []
    nontarget_scores = []
    target_trials_temp = []
    nontarget_trials_temp = []
    for (enroll_k, enroll_v) in enroll_dvectors.items():
        for (test_k, test_v) in test_dvectors.items():
            enroll_speakerid = enroll_k.split('_')[0]
            test_speakerid = test_k.split('_')[0]
            score = cosine_similarity(enroll_v, test_v)
            if enroll_speakerid == test_speakerid:
                target_scores.append(score)
                target_trials_temp.append(enroll_k + ',' + test_k)
            else:
                nontarget_scores.append(score)
                nontarget_trials_temp.append(enroll_k + ',' + test_k)
    target_index = [target_scores.index(x) for x in sorted(target_scores)]
    nontarget_index = [nontarget_scores.index(x) for x in sorted(nontarget_scores)]
    target_trials = []
    nontarget_trials = []
    for index in target_index:
        target_trials.append(target_trials_temp[index])
    for index in nontarget_index:
        nontarget_trials.append(nontarget_trials_temp[index])
    target_scores.sort()
    nontarget_scores.sort(reverse=True)

    print("target_scores:",target_scores)
    print("nontarget_scores:",nontarget_scores)
    print("target_trials:", target_trials)
    print("nontarget_trials:", nontarget_trials)
    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    eer = 0.0
    eer_th = 0.0
    for i in range(0, target_size):
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
        print("FAR: %.2f, FRR:%.2f, threshold=%.4f" % (FA_R, FR_R, threshold))
        if abs(FA_R - FR_R) <= 0.01 or FA_R < FR_R:
            eer = FR_R
            eer_th = threshold
            break
        if FA_R <= 0:
            break
    return eer, eer_th


