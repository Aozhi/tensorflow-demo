#!/usr/bin/env python2.7
#coding=utf-8

import sys
import numpy as np

ivec_dim = 600

def cosine_similarity(X, Y):
    if X.shape != Y.shape:
        print("error shape between X and Y, X shape:", X.shape, ", Y shape:", Y.shape)
        exit(1)
    x = np.sqrt(np.sum(np.square(X)))
    y = np.sqrt(np.sum(np.square(Y)))
    ret = np.dot(X,Y) / x / y
    return ret

def get_ivec(filename):
    ivec = []
    with open(filename) as f:
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            for item in items:
                try:
                    float(item)
                    ivec.append(float(item))
                except:
                    pass
            line = f.readline()
    if len(ivec) != ivec_dim:
        print "error ivector dim:", len(ivec)
        exit(1)
    return np.array(ivec, dtype=float)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(1)
    ivec1 = sys.argv[1]
    ivec2 = sys.argv[2]
    ivector1 = get_ivec(ivec1)
    ivector2 = get_ivec(ivec2)
    ret_score = cosine_similarity(ivector1, ivector2)
    print "cosine score:%.2f" % (ret_score)
