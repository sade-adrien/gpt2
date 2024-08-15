import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from gpt2 import *
import torch
import time

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps:0'
print(f'{device=}')

model = GPT2().to(device)
model = torch.compile(model)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2tokenizer_c4.model')
dataloader = DataLoaderLite(B=16 , T=1024, tokenizer=tokenizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# using tf32 matmul to speed up - use `highest` for fp32 and `medium` for bf16
torch.set_float32_matmul_precision('high')         # we notice no consistent speed up when combined with mixed-precision on A100, autocast probably overides this                      

for i in range(50):
    start = time.time()
    input_ids, targets = dataloader.next_batch()
    input_ids, targets = input_ids.to(device), targets.to(device)

    optimizer.zero_grad()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        output, loss = model(input_ids, targets)
    loss.backward()
    optimizer.step()

    if 'cuda' in device:
        torch.cuda.synchronize()
    elif 'mps' in device:
        torch.mps.synchronize()
    end = time.time()
    dt = (end - start) * 1000
    tokens_per_sec = (dataloader.B * dataloader.T / dt * 1000)

    print(f'step {i}: {loss=}, {dt=:.2f}ms, {tokens_per_sec=:.2f}')
