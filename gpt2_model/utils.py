"""
Implement useful classes and functions.
"""

import numpy as np
import torch
import math
import os

def load_tokens_from_npy(file_name):
    np_tokens = np.load(file_name).astype(np.int32)
    pt_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return pt_tokens

class DataLoaderLite:
    """ Data loader for data batch loading """

    def __init__(self, B, T, split, master_process=True, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val', 'original_gpt2')

        data_folder = 'data'
        shards = os.listdir(data_folder)
        shards = sorted([s for s in shards if split in s])
        shards = [os.path.join(data_folder, s) for s in shards]
        self.shards = shards

        assert len(self.shards) > 0, 'no shards found'
        if master_process:
            print(f'Found {len(self.shards)} shards for {split} split.')
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens_from_npy(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        full_batch = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]
        input_ids = full_batch[:-1].view(self.B, self.T)                    # (B=B, T=T)
        targets = full_batch[1:].view(self.B, self.T)                       # (B=B, T=T)

        self.current_position += self.B * self.T * self.num_processes 

        # switch shard when next batch is out of bound of current shard:
        if self.current_position + self.B * self.T * self.num_processes + 1 > len(self.tokens):
            self.current_shard += 1
            self.current_shard %= len(self.shards)      # loop over shards for multiple epochs

            self.tokens = load_tokens_from_npy(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        
        return input_ids, targets


def get_lr(step, warmup_steps, max_steps, min_lr, max_lr):
    # linear warmup
    if step < warmup_steps:
        return max_lr * ((step + 1) / warmup_steps)
    
    # revert to min lr after optim
    if step > max_steps:
        return min_lr

    # in between, cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1. + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def flatten_list(l):
    result = []
    for i in l:
        if isinstance(i, list):
            result.extend(flatten_list(i))
        else:
            result.append(i)
    return result