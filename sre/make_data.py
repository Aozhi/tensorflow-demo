#!/usr/bin/env python2.7
#coding=utf-8


import scipy.io.wavfile as wav
import numpy as np
import speechpy
import tables as tb
import os
import sys

sample_rate = 16000

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: " + sys.argv[0] + " <string:wav.scp> <string:data_file_name_without_suffix>"
        exit(1)
    file_name = sys.argv[1]
    data_file = sys.argv[2]
    wav_scp = open(file_name, 'r').readlines()
    filew = tb.open_file(data_file.rstrip('.h5') + ".h5", mode='w')
    filters = tb.Filters(complevel=5, complib='blosc')
    utterance_storage = filew.create_earray(filew.root, 'utterance', tb.Float32Atom(shape=(), dflt=0.0),
                                                shape=(0, 40), filters=filters)
    speaker_storage = filew.create_earray(filew.root, 'speakerinfo', tb.IntAtom(shape=(), dflt=0.0),
                                                shape=(0, 2), filters=filters)
    num_wav = 0
    spk_ids = {}
    c_spk_id = 1
    for line in wav_scp:
        wav_info = line.strip().split(' ')
        utt_id = wav_info[0]
        spk_id = utt_id.strip().split('_')[0]
        utt_file = wav_info[1].strip()
        fs, signal = wav.read(utt_file)
        if int(fs) != int(sample_rate):
            print "WARNING: file %s has error sample_rate %d" % (utt_file, int(fs))
        lmfe = speechpy.lmfe(signal, sampling_frequency=fs, frame_length=0.02, frame_stride=0.01,
                    num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        if not spk_id in spk_ids.keys():
            spk_ids[spk_id] = c_spk_id
            c_spk_id += 1
        if lmfe.shape[0] < 100:
            print("the [%s] wav file is no use" % (spk_id))
            continue
        utterance_storage.append(lmfe)
        speaker_storage.append(np.array([[int(spk_ids[spk_id]), int(lmfe.shape[0])]], dtype=np.int32))
        print "finish %s[%d] wav %s " % (spk_id, spk_ids[spk_id], utt_id), ",utterance shape:", lmfe.shape
        num_wav += 1
    print "finish handle %d wav files" % (num_wav)
    filew.close()
