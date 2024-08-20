"""
GPT model implementation.
We adopt a `GPT2` nomenclature as this will be configured to follow GPT2 paper even though
this architecture is an almost perfectly direct legacy of the original GPT.
"""

from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn
import inspect
import torch
import math

@dataclass                       # automatically creates base methods (such as __init__, or __repr__)
class GPT2Config:
    block_size: int = 1024       # context window
    vocab_size: int = 50_257     # vocabulary size
    n_layer: int = 12            # nb of transformer blocks
    n_head: int = 12             # nb of attention heads in MHA
    n_embd: int = 768            # hidden size of transformers


class CausalSelfAttention(nn.Module):
    """ Attention block class """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = self.config.n_head
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)             # q,k,v for all heads
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)                 # output proj
        self.c_proj.SPECIFIC_SCALE_INIT = (2 * self.config.n_layer) ** -0.5

        self.register_buffer(
                        'bias',                                                         # not a bias, name is ill-chosen (`mask_tril` would be better but we keep original for name matching)                                                         
                        torch.tril(
                            torch.ones(
                                self.config.block_size, self.config.block_size
                            )
                        ).view(1, 1, self.config.block_size, self.config.block_size)    # (B, H, T, T) 
                    )       

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)                                                            # (B, T, C*3)
        q, k, v = qkv.split(self.config.n_embd, dim=2)                                  # (B, T, C)*3
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)   # (B, nH, T, Hs)   w/ C = nH * Hs
        k = k.view(B,  T, self.config.n_head, C // self.config.n_head).transpose(1, 2)   # (B, nH, T, Hs)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)   # (B, nH, T, Hs)

        # we replace the manual attention implementation by pytorch's to enable flash-attention's kernel fusion when compiling
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))                # (B, nH, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                                                                           # (B, nH, T, Hs)
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

    def __init__(self, config=None):
        super().__init__()
        self.config = GPT2Config() if config is None else config
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

        # weight sharing scheme, research has since then adopted different weights, but for small model and training we hypothesis that this simplifies training
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights ) 

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"Input sequence is too large ({T} > {self.config.block_size})."

        pos_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.transformer.wpe(pos_ids)
        tok_emb = self.transformer.wte(input_ids)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))      # flatten on batch/seq logits before evaluation, targets is only tokens

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if hasattr(module, 'SPECIFIC_SCALE_INIT'):
                std *= module.SPECIFIC_SCALE_INIT
            torch.nn.init.normal_(module.weight, mean=.0, std=std)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias) 

    def generate(self, input_ids, max_new_tokens=100, do_sample=False, topk=50):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.transformer.wte.weight.device).view(1, -1)

        if not do_sample:
            topk = 1

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(input_ids)
                logits = logits[:, -1, :]

                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
                indices = torch.multinomial(topk_probs, 1)
                new_tokens = torch.gather(topk_indices, -1, indices)

                input_ids = torch.concat((input_ids, new_tokens), dim=-1)
            
        return input_ids
    
    @classmethod        # decorator for method to be called directly on the class rather than the object
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        
        print("Loading weights from pretrained GPT: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained('openai-community/' + model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, device, betas=(0.9, 0.999), eps=1e-8, verbose=False):
        params_dict = {n: p for n,p in self.named_parameters() if p.requires_grad}

        # we flag weight tensors in matmuls and embeddings as they have more than 1 dim
        decay_params = [p for n,p in params_dict.items() if p.dim() >= 2]
        # conversely, bias and layernorm have a max dim of 1
        nodecay_params = [p for n,p in params_dict.items() if p.dim() < 2] 

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': .0},
        ]

        # fuse optimizer if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)

        if verbose:
            print(f'nb decay parameter = {sum(p.numel() for p in decay_params):.3e}')
            print(f'nb non-decay parameter = {sum(p.numel() for p in nodecay_params):.3e}')
            print(f"using fused AdamW: {use_fused}")
        
        return optimizer