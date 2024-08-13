from gpt2 import *
import tiktoken
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)

m = GPT2.from_pretrained('gpt2')
t = tiktoken.get_encoding('gpt2')

tokens = t.encode("Hello, I am a language model,")

gen = m.generate(tokens, do_sample=True)[0]

print(gen)

print(t.decode(gen.tolist()))