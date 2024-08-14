from gpt2 import *
import tiktoken
import torch
import time

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
device = 'cpu'
print(f'{device=}')

m = GPT2.from_pretrained('gpt2').to(device)
t = tiktoken.get_encoding('gpt2')

tokens = t.encode("Hello, I am a language model,")

start = time.time()
gen = m.generate(tokens, do_sample=True, max_new_tokens=1000)[0]
end = time.time()

print(t.decode(gen.tolist()))
print(f"time = {end-start: .2f}s")