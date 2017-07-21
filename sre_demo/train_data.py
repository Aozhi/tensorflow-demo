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
    if len(sys.argv) != 4:
        print "Usage: " + sys.argv[0] + " <string:wav.scp> <string:ali-pdf.txt> <string:data_file_name_without_suffix>"
        exit(1)
    file_name = sys.argv[1]
    pdf_file_name = sys.argv[2]
    data_file = sys.argv[3]
    wav_scp = open(file_name, 'r').readlines()
    pdf_scp = open(pdf_file_name, 'r').readlines()
    if (len(pdf_scp) > len(wav_scp)):
        print "lines between pdf_scp and wav_scp error"
        exit(1)
    filew = tb.open_file(data_file.rstrip('.h5') + ".h5", mode='w')
    filters = tb.Filters(complevel=5, complib='blosc')
    utterance_storage = filew.create_earray(filew.root, 'utterance', tb.Float32Atom(shape=(), dflt=0.0),
                                                shape=(0, 40), filters=filters)
    pdf_storage = filew.create_earray(filew.root, 'pdfid', tb.IntAtom(shape=(), dflt=0.0),
                                                shape=(0, 0, ), filters=filters)
    speaker_storage = filew.create_earray(filew.root, 'speakerinfo', tb.IntAtom(shape=(), dflt=0.0),
                                                shape=(0, 2), filters=filters)
    num_wav = 0
    wav_dict = {}
    for line in wav_scp:
        wav_info = line.strip().split(' ')
        utt_id = wav_info[0]
        utt_wav = wav_info[1]
        wav_dict[utt_id] = utt_wav
    print "wav_len:", len(wav_dict)
    spk_ids = {}
    c_spk_id = 1
    for line in pdf_scp:
        num_wav += 1
        pdf_info = line.strip().split(' ')
        utt_id = pdf_info[0]
        spk_id = utt_id.strip().split('_')[0]
        pdf_labels = pdf_info[1:]
        if not utt_id in wav_dict.keys():
            print "can not find %s wav file" % (utt_id)
            continue
        utt_file = wav_dict[utt_id]
        fs, signal = wav.read(utt_file)
        if int(fs) != int(sample_rate):
            print "WARNING: file %s has error sample_rate %d" % (wav_file, int(fs))
        lmfe = speechpy.lmfe(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,
                    num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        if int(lmfe.shape[0]) != len(pdf_labels):
            print "error in compute fbank feature lmfe shape:", lmfe.shape, " number pdf_labels:", len(pdf_labels)
            continue
        if not spk_id in spk_ids.keys():
            spk_ids[spk_id] = c_spk_id
            c_spk_id += 1
        utterance_storage.append(lmfe)
        pdf_storage.append(np.array(pdf_labels, dtype=np.int32))
        speaker_storage.append(np.array([[int(spk_ids[spk_id]), len(pdf_labels)]], dtype=np.int32))
        print "finish %s[%d] wav %s " % (spk_id, spk_ids[spk_id], utt_id), ",utterance shape:", lmfe.shape, ", pdf shape:", len(pdf_labels)
    print "finish handle %d wav files" % (num_wav)
    filew.close()
