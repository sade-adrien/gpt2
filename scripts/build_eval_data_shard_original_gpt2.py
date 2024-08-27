"""
This script evaluates the original GPT2 model on the validation dataset we have built.
Because the tokenizer is not the same it requires re-building the exact same val dataset with
the original tokenizer.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from transformers import GPT2Tokenizer
import gpt2_model as gpt2
import numpy as np
import torch

def load_tokens_from_npy(file_name):
    np_tokens = np.load(file_name).astype(np.int32)
    return np_tokens

val_file = 'data/val_data.npy'
dataset = load_tokens_from_npy(val_file)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=False)
custom_tokenizer = gpt2.GPT2Tokenizer.from_pretrained('weights/gpt2tokenizer_fineweb-edu.model')

text = custom_tokenizer.decode(dataset)

tokens = tokenizer.encode(text)

tokens = np.array(tokens, dtype=np.uint16)
assert (tokens < 2**16).all()

np.save('data/shard_original_gpt2.npy', tokens)