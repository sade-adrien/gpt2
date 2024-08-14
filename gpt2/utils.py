"""
Implement useful classes.
"""

from .tokenizer import GPT2Tokenizer

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