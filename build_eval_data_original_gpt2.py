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

tokenizer_small = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=False)
tokenizer_xl = GPT2Tokenizer.from_pretrained('gpt2-xl', use_auth_token=False)
new_tokenizer = gpt2.GPT2Tokenizer.from_pretrained('weights/gpt2tokenizer_slimpajama.model')

text = new_tokenizer.decode(dataset)

tokens_small = tokenizer_small.encode(text)
tokens_xl = tokenizer_xl.encode(text)

tokens_small, tokens_xl = np.array(tokens_small, dtype=np.uint16), np.array(tokens_xl, dtype=np.uint16)
assert (tokens_small < 2**16).all()
assert (tokens_xl < 2**16).all()

np.save('data/val_data_original_gpt2_small.npy', tokens_small)
np.save('data/val_data_original_gpt2_xl.npy', tokens_xl)