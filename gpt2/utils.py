"""
Implement useful classes and functions.
"""

from .tokenizer import GPT2Tokenizer
import math

class DataLoaderLite:
    """ Data loader for data batch loading """

    def __init__(self, B, T, tokenizer):
        self.B = B
        self.T = T

        with open('data/shakespear_data.txt', 'r') as file:
            data = file.read()
        
        self.tokens = tokenizer.encode(data, return_tensors=True)[0]        # (B=1, T=whole_file_tokenized)

        print(f'Loaded {self.tokens.shape[-1]} tokens.')

        self.current_position = 0
    
    def next_batch(self):
        full_batch = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]
        input_ids = full_batch[:-1].view(self.B, self.T)                    # (B=B, T=T)
        targets = full_batch[1:].view(self.B, self.T)                       # (B=B, T=T)

        self.current_position += self.B * self.T

        # loop when next batch is out of bound of file:
        if self.current_position + self.B * self.T + 1 >= len(self.tokens):
            self.current_position = 0
        
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