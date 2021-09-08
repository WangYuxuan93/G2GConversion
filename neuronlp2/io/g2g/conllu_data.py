__author__ = 'max'

import os.path
import numpy as np
from collections import defaultdict, OrderedDict
import torch
import re

from neuronlp2.io.g2g.reader import CoNLLUReaderG2G
from neuronlp2.io.alphabet import Alphabet
from neuronlp2.io.logger import get_logger
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io.common import ROOT, END, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [PAD, ROOT, END]
NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 140]


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<
def read_bucketed_data(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
                       pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
                       mask_out_root=False, pos_idx=4):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    src_words = [[] for _ in _buckets]

    reader = CoNLLUReaderG2G(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                             pre_alphabet=pre_alphabet, pos_idx=pos_idx)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end) # Jeffrey: sentence will be transformed to a instance
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, sent.pre_ids, inst.src_heads, inst.src_type_ids]) #Jeffrey: bucket principle
                src_words[bucket_id].append(sent.words)
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])  # Jeffrey: record the max sen length in every bucket
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len  # Jeffrey: record the max char length in every bucket
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))] # Jeffrey: sample size in evrey bucket
    data_tensors = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensors.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id])
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.zeros([bucket_size, bucket_length, bucket_length], dtype=np.int64)
        tid_inputs = np.zeros([bucket_size, bucket_length, bucket_length], dtype=np.int64)
        preid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        # source graph
        src_hid_inputs = np.zeros([bucket_size, bucket_length, bucket_length], dtype=np.int64)
        src_tid_inputs = np.zeros([bucket_size, bucket_length, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids, preids, src_hids, src_tids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            if pre_alphabet:
                preid_inputs[i, :inst_size] = preids
                preid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG

            # heads,type ids
            for h, hid in enumerate(hids):
                for kk, x in enumerate(hid):
                    hid_inputs[i, h, x] = 1
                    tid_inputs[i, h, x] = tids[h][kk]
                hid_inputs[i, h, inst_size:] = PAD_ID_TAG
                tid_inputs[i, h, inst_size:] = PAD_ID_TAG

            # souce graph
            for h, hid in enumerate(src_hids):
                for kk, x in enumerate(hid):
                    src_hid_inputs[i, h, x] = 1
                    src_tid_inputs[i, h, x] = src_tids[h][kk]
                src_hid_inputs[i, h, inst_size:] = PAD_ID_TAG
                src_tid_inputs[i, h, inst_size:] = PAD_ID_TAG

            # masks
            if symbolic_end:
                # mask out the end token
                masks[i, :inst_size-1] = 1.0
            else:
                masks[i, :inst_size] = 1.0   # mask the padding
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1
        if mask_out_root:
            masks[:,0] = 0

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks = torch.from_numpy(masks)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)
        pres = torch.from_numpy(preid_inputs)
        # source graph
        src_heads = torch.from_numpy(src_hid_inputs)
        src_types = torch.from_numpy(src_tid_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'PRETRAINED': pres,
                       'SRC': np.array(src_words[bucket_id],dtype=object), 
                       'SRC_HEAD': src_heads, 'SRC_TYPE': src_types }
        data_tensors.append(data_tensor)
    return data_tensors, bucket_sizes

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<
def read_data(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
              pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
              mask_out_root=False, pos_idx=4):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLUReaderG2G(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                          pre_alphabet=pre_alphabet, pos_idx=pos_idx)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    src_words = []
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        #print (inst.sentence.words)
        data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, sent.pre_ids, inst.src_heads, inst.src_type_ids])
        src_words.append(sent.words)
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    hid_inputs = np.zeros([data_size, max_length,max_length], dtype=np.int64)  # Jeffrey: 由empty 改成zeros
    tid_inputs = np.zeros([data_size, max_length,max_length], dtype=np.int64)
    # source graph
    src_hid_inputs = np.zeros([data_size, max_length,max_length], dtype=np.int64) 
    src_tid_inputs = np.zeros([data_size, max_length,max_length], dtype=np.int64)

    preid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    masks = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    for i, inst in enumerate(data):
        wids, cid_seqs, pids, hids, tids, preids, src_hids, src_tids = inst
        inst_size = len(wids)
        lengths[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        if pre_alphabet:
            preid_inputs[i, :inst_size] = preids
            preid_inputs[i, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[i, c, :len(cids)] = cids
            cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[i, :inst_size] = pids
        pid_inputs[i, inst_size:] = PAD_ID_TAG
        # type ids ,heads
        for h, hid in enumerate(hids):
            for kk, x in enumerate(hid):
                hid_inputs[i, h, x] = 1
                tid_inputs[i, h, x] = tids[h][kk]
            hid_inputs[i, h, inst_size:] = PAD_ID_TAG
            tid_inputs[i, h, inst_size:] = PAD_ID_TAG

        for h, hid in enumerate(src_hids):
            for kk, x in enumerate(hid):
                src_hid_inputs[i, h, x] = 1
                src_tid_inputs[i, h, x] = src_tids[h][kk]
            src_hid_inputs[i, h, inst_size:] = PAD_ID_TAG
            src_tid_inputs[i, h, inst_size:] = PAD_ID_TAG

        # masks
        if symbolic_end:
            # mask out the end token
            masks[i, :inst_size-1] = 1.0
        else:
            masks[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1
    if mask_out_root:
        masks[:,0] = 0

    words = torch.from_numpy(wid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks = torch.from_numpy(masks)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)
    pres = torch.from_numpy(preid_inputs)
    # source graph
    src_heads = torch.from_numpy(src_hid_inputs)
    src_types = torch.from_numpy(src_tid_inputs)

    data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                   'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'SRC': src_words,
                   'PRETRAINED': pres, 'SRC_HEAD': src_heads, 'SRC_TYPE': src_types}
    return data_tensor, data_size