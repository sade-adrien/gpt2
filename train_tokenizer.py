from tokenizer import GPT2Tokenizer
from datasets import load_dataset

en = load_dataset("cerebras/SlimPajama-627B", "train", streaming=True)

