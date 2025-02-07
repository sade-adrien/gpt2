from gpt2_model import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

vocab_size = 50_304         # GPT2 vocab size (rounded up from 50257 to a `dense-power-of-2` number for computation optimization)

n_samples = 1_000
# dataset = load_dataset('cerebras/SlimPajama-627B', split='train', streaming=True)
dataset = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
dataset = dataset.take(n_samples)

data = ""
for d in tqdm(dataset):
    data += d['text'] + ' '

start = time.time()

tokenizer = GPT2Tokenizer()
tokenizer.train(data, vocab_size, verbose=True)
tokenizer.save('weights/gpt2tokenizer_fineweb-edu')

end = time.time()

print(f'Training tokenizer took {end-start:.2f}s.')
