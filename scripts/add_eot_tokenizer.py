"""
Register end of sentence/doc token (overide last one to no change the vocab size that was carefully selected)
we need this script bc it was not included in the original training_tokenizer file...
"""

from gpt2_model import GPT2Tokenizer

# tokenizer = GPT2Tokenizer.from_pretrained('weights/gpt2tokenizer_slimpajama.model')
tokenizer = GPT2Tokenizer.from_pretrained('weights/gpt2tokenizer_fineweb-edu.model')

tokenizer.register_special_tokens({'<|endoftext|>': 50_303})

removed = tokenizer.merges.popitem()        
print(f'Removed last merge: {removed}')

tokenizer.vocab = tokenizer._build_vocab()

tokenizer.save('weights/gpt2tokenizer_fineweb-edu_')