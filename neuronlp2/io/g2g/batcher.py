# For batching g2g data

import numpy as np
import torch
import random

def iterate_data_g2g(data, batch_size, bucketed=False, unk_replace=0., shuffle=False, task_type="dp", **kwargs):
    if bucketed:
        return iterate_bucketed_batch(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
    else:
        return iterate_batch(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)


def iterate_bucketed_batch(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC', 'HEAD', 'TYPE',"SRC_HEAD", "SRC_TYPE","info2","info1"] + stack_keys)
    sdp_keys = ["HEAD", "TYPE", "SRC_HEAD", "SRC_TYPE","info2"]
    long_keys = ["info1"]
    stack_keys = set(stack_keys)
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths, "SRC":data["SRC"][excerpt]}#, 'SRC': data['SRC'][excerpt]}
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
            batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
            batch.update({key: field[excerpt, :batch_length,:batch_length] for key, field in data.items() if key in sdp_keys})
            batch.update({key: field[excerpt, :batch_length,:batch_length,:batch_length] for key, field in data.items() if key in long_keys})
            yield batch

def iterate_batch(data, batch_size, unk_replace=0., shuffle=False):
    data, data_size = data
    words = data['WORD']
    single = data['SINGLE']
    max_length = words.size(1)

    if unk_replace:
        ones = single.new_ones(data_size, max_length)
        noise = single.new_empty(data_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    indices = None
    if shuffle:
        indices = torch.randperm(data_size).long()
        indices = indices.to(words.device)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC', 'HEAD', 'TYPE',"SRC_HEAD", "SRC_TYPE","info2","info1"] + stack_keys)
    stack_keys = set(stack_keys)
    sdp_keys = ['HEAD', 'TYPE', "SRC_HEAD", "SRC_TYPE","info2"]
    long_keys = ['info1']
    for start_idx in range(0, data_size, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        lengths = data['LENGTH'][excerpt]
        batch_length = lengths.max().item()
        batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths, 'SRC': data['SRC'][excerpt]}
        batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
        batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
        batch.update({key: field[excerpt, :batch_length,:batch_length] for key, field in data.items() if key  in sdp_keys})
        batch.update({key: field[excerpt, :batch_length, :batch_length, :batch_length] for key, field in data.items() if key in long_keys})
        yield batch


# if __name__ == '__main__':
#     easyfirst_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
#     batch = {'WORD':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'MASK':[[1,1,1,1,1,0],[1,1,1,1,1,1]],
#              'POS':[[0,1,2,3,4,5],[0,6,7,8,0,0]], 'LENGTH':np.array([6,5]),
#              'CHAR':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'HEAD':[[0,3,1,0,3,5],[0,6,7,8,0,0]],'TYPE':[[0,1,2,3,4,5],[0,6,7,8,0,0]]}
#     lengths = batch['LENGTH']
#     sample_generate_order(batch, lengths, n_recomp=3)
