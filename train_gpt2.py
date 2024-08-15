import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.nn as nn
from gpt2 import *
import torch
import time

##################################################################################################

gpt2config = GPT2Config(vocab_size=50_304)
global_batch = 524_288          # global batch size to fit gpt2 batch for 125M params (B=.5M) with a dense-power-of-2 number
B = 16
T = 1024
assert global_batch % (B * T) == 0, 'ensure coherence between micro batch and global batch'
gradient_accumulation_steps = global_batch // (B * T)
max_steps = 50
warmup_steps = 5
max_lr = 6e-4
min_lr = max_lr / 10
betas = (.9, .95)
eps = 1e-8
weight_decay = 0.1

##################################################################################################

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps:0'
print(f'{device=}')

##################################################################################################

model = GPT2(gpt2config).to(device)         #50_304 is dense-power-of-2 number
# model = torch.compile(model)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2tokenizer_c4.model')
dataloader = DataLoaderLite(B=B , T=T, tokenizer=tokenizer)
optimizer = model.configure_optimizer(weight_decay=weight_decay, learning_rate=max_lr, betas=betas, eps=eps, device=device, verbose=True)

# using tf32 matmul to speed up - use `highest` for fp32 and `medium` for bf16
torch.set_float32_matmul_precision('high')         # we notice no consistent speed up when combined with mixed-precision on A100, autocast probably overides this                      

for step in range(max_steps):
    start = time.time()

    optimizer.zero_grad()

    loss_accumulation = .0
    for _ in range(gradient_accumulation_steps):
        input_ids, targets = dataloader.next_batch()
        input_ids, targets = input_ids.to(device), targets.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output, loss = model(input_ids, targets)

        loss /= gradient_accumulation_steps
        loss_accumulation += loss.detach()
        loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)

    lr = get_lr(step, warmup_steps, max_steps, min_lr, max_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    if 'cuda' in device:
        torch.cuda.synchronize()
    elif 'mps' in device:
        torch.mps.synchronize()
    end = time.time()
    dt = (end - start) * 1000
    tokens_per_sec = (dataloader.B * dataloader.T * gradient_accumulation_steps / dt * 1000)

    print(f'step {step}: {lr=:.6f}, loss={loss_accumulation.item():.8f}, {norm=:.3f}, {dt=:.2f}ms, {tokens_per_sec=:.2f}')
