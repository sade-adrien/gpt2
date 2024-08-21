"""
This script evaluates the original GPT2 (small and xl) on our SlimPajama eval shard (re-tokenized accordingly).
We'll use these losses to compare with our custom GPT2 training.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from gpt2_model import DataLoaderLite
from transformers import GPT2LMHeadModel
from tqdm import tqdm
import numpy as np
import torch

B = 32
T = 1024
max_val_steps = len(np.load('data/shard_original_gpt2.npy')) // (B * T)
device = 'cuda:0'

def run_micro_step(model, dataloader, device):
    input_ids, targets = dataloader.next_batch()
    input_ids, targets = input_ids.to(device), targets.to(device)
    # mixed-precision training
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model(input_ids, labels=targets).loss
    return loss

def run_eval(model, dataloader, device):
    dataloader.reset()
    with torch.no_grad():
        loss_accumulation = .0
        for _ in tqdm(range(max_val_steps)):
            loss = run_micro_step(model, dataloader, device)
            loss_accumulation += loss.detach() / max_val_steps
    return loss_accumulation

dataloader = DataLoaderLite(B=B, T=T, split='original_gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2', device_map=device)
model = GPT2LMHeadModel.from_pretrained('gpt2-xl', device_map=device)

model.eval()
eval_loss = run_eval(model, dataloader, device)

print(f'Original GPT2-{sum(p.numel() for p in model.parameters()):.2e} scores on our SlimPajam eval subset a loss={eval_loss:.6f}.')

# GPT2-124M --> eval_loss=9.507551
# GPT2-1.5B --> eval_loss=