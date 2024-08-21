"""
Evaluate a model on the hellaswag task (text prediction).
"""

from torch.nn import functional as F
import torch

val_file = 'data/hellaswag_val.jsonl'

def flatten_list(l):
    result = []
    for i in l:
        if isinstance(i, list):
            result.extend(flatten_list(i))
        else:
            result.append(i)
    return result

def render_example(example, tokenizer):
    """
    Given the example of type dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN - 4 candidates)
    - mask (1 where we evaluate the logl, 0 elsewhere)
    - label (index of the correct completion)
    """

    context = example['ctx']
    label = example['label']
    endings = example['endings']

    tokens_list = []
    mask_list = []

    context_tokens = flatten_list(tokenizer.encode(context))

    for end in endings:
        end_tokens = flatten_list(tokenizer.encode(' ' + end))
        tokens_list.append(context_tokens + end_tokens)
        mask_list.append([0] * len(context_tokens) + [1] * len(end_tokens))

    tokens = torch.zeros((4, max([len(t) for t in tokens_list])), dtype=torch.long)
    mask = torch.zeros((4, max([len(t) for t in tokens_list])), dtype=torch.long)

    for i in range(4):
        tokens[i, :len(tokens_list[i])] = torch.tensor(tokens_list[i])
        mask[i, :len(mask_list[i])] = torch.tensor(mask_list[i])
    
    return tokens, mask, label


def iterate_data(file_name):
    with open(file_name) as file:
        for line in file:
            data = json.loads(line)
            yield data

def hellaswag_evaluation(model, tokenizer, device):
    num_correct_norm, num_correct, num_total = 0, 0, 0

    for i, example in enumerate(iterate_data(val_file)):
        if i % 100 == 0:
            print(f'{i}/{10_042}')
        tokens, mask, label = render_example(example, tokenizer)
        tokens, mask = tokens.to(device), mask.to(device)
        
        with torch.no_grad():
            outputs = model(tokens)

        try:
            logits = outputs.logits
        except:
            logits = outputs[0]

        # compute the loss (while accounting for the mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flatten_shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        flatten_shift_tokens = shift_tokens.view(-1)

        shift_losses = F.cross_entropy(flatten_shift_logits, flatten_shift_tokens, reduction='none').view(tokens.shape[0], -1)      # prevent reduction as we first have to account for the mask

        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # now we predict the label based on the loss
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)


    return num_correct/num_total, num_correct_norm/num_total





import os
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from gpt2_model import GPT2Tokenizer, GPT2, GPT2Config
import json
import torch

torch.set_float32_matmul_precision('high') # use tf32
tok = GPT2Tokenizer.from_pretrained('gpt2')
mod = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.bfloat16, device_map='cuda:0')
mod = torch.compile(mod)
acc, acc_norm = hellaswag_evaluation(mod,tok,'cuda:0')

print(f'GPT2 small - custom hellaswag: {acc=:.3f}, {acc_norm:.3f}')