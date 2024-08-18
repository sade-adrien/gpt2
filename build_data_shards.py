"""
This script downloads the data (streamed) and tokenize it using a
pretrained tokenizer. It then saves the data using multiple shards.
"""

from datasets import load_dataset
from gpt2 import GPT2Tokenizer
import numpy as np

local_dir = 'data/'
shard_size = 100_000_000    # 100M tokens/shard
total_tokens = 1_000_000_000

dataset = load_dataset('cerebras/SlimPajama-627B', split='train', streaming=True)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2tokenizer_slimpajama.model')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2tokenizer_slimpajama.model')

# register end of sentence/doc token (overide last one to no change the vocab size that was carefully selected)
# we need those two lines bc my stupid-ass forgot to include it when training the tokenizer
tokenizer.register_special_tokens({'<|endoftext|>': 50_303})        
tokenizer.vocab = tokenizer._build_vocab()
eot = 50_303

def tokenize(doc):
    tokens = [eot]
    tokens.extend(tokenizer.encode(doc, allowed_special='none')[0])
    tokens_np = np.array(tokens, dtype=np.uint16)
    assert (tokens_np < 2**16).all(), 'vocab size too large for np.uint16'
    return tokens_np

def save_shard(file_name, tokens_np):
    np.save('data/' + file_name, tokens_np)


nb_shards = (total_tokens // shard_size) if (total_tokens % shard_size == 0) else (total_tokens // shard_size + 1)
current_shard = 0
tokens_count_within_shard = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)

for data in dataset:
    print(f'shard: {current_shard}/{nb_shards-1}+1 | tokens_in_shard: {tokens_count_within_shard:,} / {shard_size:,}')

    tokens = tokenize(data['text'])
    if tokens_count_within_shard + len(tokens) <= shard_size:
        all_tokens_np[tokens_count_within_shard : tokens_count_within_shard + len(tokens)] = tokens
        tokens_count_within_shard += len(tokens)

    else:
        remainder = shard_size - tokens_count_within_shard
        all_tokens_np[tokens_count_within_shard : tokens_count_within_shard + remainder] = tokens[:remainder]

        print('saving shard...')
        file_name = f'train_data_{current_shard}' if current_shard < nb_shards else f'val_data'
        save_shard(file_name, all_tokens_np)

        if current_shard == nb_shards:
            break

        current_shard += 1
        tokens_count_within_shard = len(tokens) - remainder
        # if too long we just skip to avoid anoying issues
        if tokens_count_within_shard > shard_size:
            continue
        all_tokens_np[0 : tokens_count_within_shard] = tokens[remainder:]

