"""
GPT model implementation.
We adopt a `GPT2` nomenclature as this will be configured to follow GPT2 paper even though
this architecture is an almost perfectly direct legacy of the original GPT.
"""

from dataclasses import dataclass
from torch.nn import functional as F
import torch.nn as nn

@dataclass                      # automatically creates base methods (such as __init__, or __repr__)
class GPT2Config:
    block_size: int = 256       # context window
    vocab_size: int = 65        # vocabulary size
    n_layer: int = 6            # nb of transformer blocks
    n_head: int = 6             # nb of attention heads in MHA
    n_embd: int = 384           # hidden size of transformers


class CausalSelfAttention(nn.Module):
    """ Attention block class """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = self.config.n_head
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)             # q,k,v for all heads
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)                 # output proj

        self.register_buffer(
                        'mask_tril',                                                         
                        torch.tril(
                            torch.ones(
                                self.config.block_size, self.config.block_size
                            )
                        ).view(1, 1, self.config.block_size, self.config.block_size)    # (B, H, T, T) 
                    )       

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)                                                            # (B, T, C*3)
        q, k, v = qvk.split(self.config.n_embd, dim=2)                                  # (B, T, C)*3
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)   # (B, nH, T, Hs)   w/ C = nH * Hs
        k = k.view(B,  T, self.config.n_head, C // self.config.n_head).transpose(1, 2)   # (B, nH, T, Hs)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)   # (B, nH, T, Hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))                # (B, nH, T, T)
        att = att.masked_fill(self.mask_tril[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v                                                                     # (B, nH, T, Hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                                # (B, T, C) re-assemble heads
        
        y = self.c_proj(y)

        return y 


class MLP(nn.Module):
    """ MLP block class """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(self.config.n_embd, 4 * self.config.n_embd)
        self.gelu = nn.GELU()           # original GPT2 used the tanh approx (nn.GELU(approximate='tanh'))
        self.c_proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """ Transformer block class """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(self.config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

class GPT2(nn.Module):
    """ GPT2 model class """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                'wte': nn.Embedding(self.config.vocab_size, self.config.n_embd),
                'wpe': nn.Embedding(self.config.block_size, self.config.n_embd),
                'h': nn.ModuleList(
                    [Block(self.config) for _ in range(self.config.n_layer)]
                ),
                'ln_f': nn.LayerNorm(self.config.n_embd),            
            }
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

