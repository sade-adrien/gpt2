import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from gpt2_model import GPT2, GPT2Config

model = GPT2(GPT2Config(vocab_size=50_304, n_layer=64, n_head=24, n_embd=4096)).to('cuda:0')
model2 = GPT2(GPT2Config(vocab_size=50_304, n_layer=64, n_head=24, n_embd=4096)).to('cuda:1')

while True:
    pass