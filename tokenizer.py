"""
Define Tokenizer class (base) and the GPT2Tokenizer class wrapper to implement specific behavior of GPT2 Tokenizer.
"""

import regex as re
import unicodedata

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_stats(bytes_ids, stats=None):
    """
    Returns a dictionnary of consecutive pairs count.
    bytes_ids = [1, 4, 6, 3, 1, 4, 6]
    return {(1,4): 2, (4,6): 2, (6,3): 1, (3,1): 1}
    """

    stats = {} if stats is None else stats
    for p0, p1 in zip(bytes_ids, bytes_ids[1:]):
        stats[(p0,p1)] = stats.get((p0,p1), 0) + 1
    return stats

def merge(bytes_ids, pair, new_id):
    """
    Merges all consecutive ids that are pair and replace them by new_id.
    bytes_ids = [1, 4, 6, 3, 1, 4, 6], pair = (1,4), new_id=7
    return [7, 6, 3, 7, 6]
    """

    i = 0
    while i < len(bytes_ids) - 1:
        if (bytes_ids[i], bytes_ids[i+1]) == pair:
            bytes_ids[i] = new_id
            bytes_ids.pop(i+1)
        i += 1

    return bytes_ids

def clean_control_char(string):
    """
    Removes control character like `\n` in a string - usefull for pretty printing.
    """

    chars = []
    for char in string:
        if unicodedata.category(char)[0] == 'C':
            chars.append(f"\\u{ord(char):04x}")           # unknown utf-8 char
        else:
            chars.append(char)
    return "".join(chars)

def render_token(token):
    """
    Prints nicely a token.
    """

    string = token.decode('utf-8', errors='replace')
    string = clean_control_char(string)
    return string

class Tokenizer:
    """ Base Tokenizer Class """

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, ids):
        raise NotImplementedError
    
    def _build_vocab(self):
        """
        Builds vocabulary from merges
        """

        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')
        
        return vocab
    
    def save(self, file_name):
        """
        Saves file_name.vocab and file_name.model.
        The first one is a pretty print to analyze the vocab.
        The second one contains the actual model that will be loaded with self.load().
        """

        with open(file_name+'.model', 'w') as file:
            file.write(self.__class__.__name__ + '\n')              # name/version of tokenizer
            file.write(self.pattern + '\n')                         # write the pattern
            file.write(str(len(self.special_tokens)) + '\n')        # write the nb of special tokens
            for special, idx in self.special_tokens.items():
                file.write(special + ' ' + str(idx) + '\n')         # write the special tokens
            for idx1, idx2 in self.merges:
                file.write(str(idx1) + ' ' + str(idx2) + '\n')      # write the merges
            
        merges_revert = {idx: pair for pair, idx in self.merges.items()}
        with open(file_name+'.vocab', 'w', encoding='utf-8') as file:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in merges_revert:
                    idx0, idx1 = merges_revert[idx]
                    s0, s1 = render_token(self.vocab[idx0]), render_token(self.vocab[idx1])
                    file.write(f'[{s0}][{s1}] -> [{s}] {idx}\n')
                else:
                    file.write(f'[{s}] {idx}\n')


    def load(self, file_name):
        """
        Loads file_name.model and builds vocab, ie load tokenizer.
        """

        assert file_name.endswith(".model")
        merges = {}
        special_tokens = {}
        pattern = ""
        idx = 256

        with open(file_name, 'r', encoding='utf-8') as file:
            name_version = file.readline().strip()
            assert 'GPT2' in name_version

            pattern = file.readline().strip()
            num_special_tokens = int(file.readline().strip())

            for  _ in range(num_special_tokens):
                special, special_idx = file.readline().strip().split()
                special_tokens[special] = int(special_idx)
            
            for line in file:
                idx1, idx2 = line.split()
                idx1, idx2 = int(idx1), int(idx2)
                merges[(idx1, idx2)] = idx
                idx += 1
        
        self.merges = merges
        self.special_tokens = special_tokens
        self.pattern = pattern
        self.vocab = self._build_vocab()

class GPT2Tokenizer(Tokenizer):
    """ GPT2 Tokenizer Class, infant Tokenizer class to reproduce GPT2 """

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.special_tokens = {}
        self.invert_special_tokens = {}
    
    def train(self, text, vocab_size, verbose=False):
        """
        Trains tokenizer on a text document. Will erase exising vocab and merges.
        """

        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_chunks = re.findall(self.pattern, text)
        text_ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk_ids in text_ids:
                stats = get_stats(chunk_ids, stats)
            max_pair = max(stats, key=stats.get)
            new_id = 256 + i

            text_ids = [merge(tid, max_pair, new_id) for tid in text_ids]
            merges[max_pair] = new_id
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]

            if verbose:
                print(f"Merge {i+1}/{num_merges}: {vocab[max_pair[0]]} [{max_pair[0]}] + {vocab[max_pair[1]]} [{max_pair[1]}] --> {vocab[new_id]} [{new_id}]")
            
            self.merges = merges
            self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        """
        Registers all special tokens at once.
        """

        self.special_tokens = special_tokens
        self.invert_special_tokens = {v: k for k,v in self.special_tokens.items()}
    
    def decode(self, ids):
        """
        Returns string from integer ids.
        """

        bytes_ids = []
        for idx in ids:
            if idx in self.vocab:
                bytes_ids.append(self.vocab[idx])
            elif idx in self.special_tokens:
                bytes_ids.append(self.inverse_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f'Invalid token id: {idx}.')
            
        bytes_text = b"".join(bytes_ids)
        text = bytes_text.decode('utf-8', errors='replace')
        return text

    def _encode_chunks(self, bytes_text):
        """
        Encodes bytes.
        """

        bytes_ids = list(bytes_text)
        while len(bytes_ids) >= 2:
            stats = get_stats(bytes_ids)
            min_pair = min(stats, key=lambda p: stats.get(p, float('inf')))     # select the pair with lowest merging opportunities
            if min_pair not in self.merges:
                break                                                           # stop merging if no more merging opportunities

            idx = self.merges[min_pair]
            bytes_ids = merge(bytes_ids, min_pair, idx)
        return bytes_ids
    
    def encode_ordinary(self, text):
        """
        Encodes text while ignoring all special tokens.
        """
        text_chunks =re.findall(self.pattern, text)
        text_ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            text_ids.extend(self._encode_chunks(chunk_bytes))
        return text_ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Encodes text with handling of special tokens.
        allowed_special = "none_raise" | "all" | "none"
        "none_raise" raises an error if a special token is encountered. 
        "all" accounts for all special tokens.
        "none" ignores all special tokens (similar to self.encode_ordinary).
        """

        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            assert all(token not in text for token in self.special_tokens)
        else:
            raise ValueError(f"`{allowed_special=}` not recognized, select [`none_raise`, `all`, `none`]")
        
        if special == {}:
            return self.encode_ordinary(text)
        
        special_pattern = '(' + "|".join(re.escape(special_token) for special_token in special) + ')'      # finding all special tokens in the text based on a matching - re.escape helps handling special char such as '['
        split_chunks = re.split(special_pattern, text)                                                     # text is split such as to separate special tokens from regular

        text_ids = []
        for chunk in split_chunks:
            if chunk in special:
                text_ids.append(special[chunk])
            else:
                text_ids.extend(self.encode_ordinary(chunk))
        
        return text_ids


        


