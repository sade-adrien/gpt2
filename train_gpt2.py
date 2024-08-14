from gpt2 import *
import torch
import time

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = 'mps'

model = GPT2().to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2tokenizer_c4.model')
dataloader = DataLoaderLite(B=4 , T=32, tokenizer=tokenizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


for i in range(50):
    start = time.time()
    input_ids, targets = dataloader.next_batch()
    input_ids, targets = input_ids.to(device), targets.to(device)

    optimizer.zero_grad()
    output, loss = model(input_ids, targets)
    loss.backward()
    optimizer.step()

    torch.mps.synchronize()
    end = time.time()
    dt = (end - start) * 1000
    tokens_per_sec = (dataloader.B * dataloader.T / dt * 1000)

    print(f'step {i}: {loss=}, {dt=:.2f}ms, {tokens_per_sec=:.2f}')
