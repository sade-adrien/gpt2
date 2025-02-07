"""
Script to train a GPT2 model using a pre-trained GPT2Tokenizer (re-implemantation).
Script can be run with `python train_gpt2.py` if not using DDP (ensure use_DDP=False).
If using DDP, run with `torchrun --standalone --nproc_per_node=2 train_gpt2.py` (and change the nb of gpu/node accordingly).
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.hellaswag_eval import hellaswag_evaluation
import torch.distributed as dist
from gpt2_model import *
import torch.nn as nn
from tqdm import tqdm
import torch
import time
import json

##################################################################################################
gpt2config = GPT2Config(vocab_size=50_304)      # round-up vocab_size to a dense-power-of-2 number for efficient computations
global_batch = 2**19                            # global batch size to fit gpt2 batch for 125M params (B=.5M) with a dense-power-of-2 number
B = 64
T = 1024
max_steps = int(1.59e10 / global_batch)          # ~1 epoch for the above global batch and a dataset of 15.9B tokens
max_val_steps = int(1e8 / global_batch)         # evaluation steps to perform (eval file is 100M tokens)
val_steps = 300                                 # frequency of evaluation
save_steps = int(max_steps / 10)                # frequency of checkpoint saving (~10 saves)
save_dir = 'weights/'                           # directory for model/log saving
log_steps = 1                                   # frequency of logs
log_file = save_dir + 'logs.json'               # json for easy parsing
warmup_steps = int(3.75e8 / global_batch)       # linear warmup over the first 375M tokens as in GPT3 training
max_lr = 6e-4
min_lr = max_lr * .1
betas = (.9, .95)
eps = 1e-8
weight_decay = 0.1
##################################################################################################
# to run with DDP, use command: torchrun --standalone --nproc_per_node=2 train_gpt2.py
use_DDP = True

# seed is absolutely needed when using DDP, to init similarly the model on all GPUs
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

if use_DDP:
    assert torch.cuda.is_available(), "to use data distributed parallel scheme, you need cuda GPUs"
    init_process_group(backend='nccl')
    DDP_rank = int(os.environ['RANK'])                  # GPU global rank
    DDP_local_rank = int(os.environ['LOCAL_RANK'])      # GPU rank within node
    DDP_world_size = int(os.environ['WORLD_SIZE'])      # nb of GPUs available
    device = f'cuda:{DDP_local_rank}'
    torch.cuda.set_device(device)

    # flag gpu0 as master process to handle one-time actions
    master_process = (DDP_rank == 0)

else:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    DDP_rank, DDP_local_rank, DDP_world_size, master_process = 0, 0, 1, True

if master_process:
    print(f'Using device: {device}')

# preferably those 2 lines would have been placed one section up with the hyperparams
# however grad_acc_steps depends of the DDP_world_size defined here
assert global_batch % (B * T * DDP_world_size) == 0, 'ensure coherence between micro batch and global batch'
gradient_accumulation_steps = global_batch // (B * T * DDP_world_size)

# open for writing to clear the log file
with open(log_file, "w") as file:
    pass
##################################################################################################


def main():
    model = GPT2(gpt2config).to(device)                # 50_304 is dense-power-of-2 number
    # model = torch.compile(model)                     # for unkown reasons, compilation breaks the hellaswag eval...
    if use_DDP:
        model = DDP(model, device_ids=[DDP_local_rank])
    raw_model = model.module if use_DDP else model

    if master_process:
        print(f'Training a GPT2-{sum(p.numel() for p in raw_model.parameters()):.3e}.')


    tokenizer = GPT2Tokenizer.from_pretrained('weights/gpt2tokenizer_fineweb-edu.model') 
    train_dataloader = DataLoaderLite(B=B , T=T, split='train', master_process=master_process, process_rank=DDP_rank, num_processes=DDP_world_size)
    val_dataloader = DataLoaderLite(B=B , T=T, split='val', master_process=master_process, process_rank=DDP_rank, num_processes=DDP_world_size)
    optimizer = raw_model.configure_optimizer(weight_decay=weight_decay, learning_rate=max_lr, betas=betas, eps=eps, device=device)

    # using tf32 matmul to speed up - use `highest` for fp32 and `medium` for bf16
    torch.set_float32_matmul_precision('high')         # we notice no consistent speed up when combined with mixed-precision on A100, autocast probably overides this                      

    for step in tqdm(range(max_steps)):

        # eval loop
        val_loss, hellaswag_acc, hellaswag_acc_norm = None, None, None
        if (step % val_steps == 0) or (step == max_steps - 1):
            val_loss = run_eval(model, val_dataloader, device)
            hellaswag_acc, hellaswag_acc_norm = run_hellaswag_eval(model, tokenizer, device)
            if master_process:
                print(f'Step {step}: Validation Loss={val_loss:.6f}, hellaswag accuracy={hellaswag_acc:.3f}, hellaswag accuracy_norm={hellaswag_acc_norm:.3f}')

        # run 1 training step
        start = time.time()
        optimizer.zero_grad()
        loss_accumulation = .0
        
        if use_DDP:
            # in case of DDP the no_sunc manager avoid sharing gradients at all micro steps, we do it only once at the end to save time
            with model.no_sync():   
                for micro_step in range(gradient_accumulation_steps - 1):           # -1 because we perform the last on out of the no_sync context manager
                    loss = run_micro_step(model, train_dataloader, device)
                    loss_accumulation += micro_step_loss_backward(loss, gradient_accumulation_steps)
            # one micro step out of the no_sync context to share gradients between GPUs when using DDP
            loss = run_micro_step(model, train_dataloader, device)
            loss_accumulation += micro_step_loss_backward(loss, gradient_accumulation_steps)

            # all_reduce on loss_accumulation to avg the loss from all GPUs
            dist.all_reduce(loss_accumulation, op=dist.ReduceOp.AVG)
        
        else:
            for micro_step in range(gradient_accumulation_steps):
                loss = run_micro_step(model, train_dataloader, device)
                loss_accumulation += micro_step_loss_backward(loss, gradient_accumulation_steps)

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)

        lr = get_lr(step, warmup_steps, max_steps, min_lr, max_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if 'cuda' in device:
            torch.cuda.synchronize()
        elif 'mps' in device:
            torch.mps.synchronize()
        end = time.time()
        dt = (end - start)
        tokens_per_sec = (train_dataloader.B * train_dataloader.T * gradient_accumulation_steps * DDP_world_size / dt)

        if master_process:
            print(f'step {step}: {lr=:.6f}, loss={loss_accumulation.item():.6f}, {dt=:.2f}s, {tokens_per_sec=:,.0f}')
        
        # update logs
        if master_process and (step % log_steps == 0):
            save_log(step, lr, norm, loss_accumulation, val_loss, hellaswag_acc, hellaswag_acc_norm)
        
        # save checkpoint
        if master_process and step > 0 and (step % save_steps == 0 or step == max_steps - 1):
            save_checkpoint(raw_model, optimizer, step, val_loss, lr)

    # killing DDP processes cleanly
    if use_DDP:
        destroy_process_group()


def run_micro_step(model, dataloader, device):
    input_ids, targets = dataloader.next_batch()
    input_ids, targets = input_ids.to(device), targets.to(device)

    # mixed-precision training
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(input_ids, targets)
        
    return loss

def micro_step_loss_backward(loss, gradient_accumulation_steps):
    loss /= gradient_accumulation_steps
    loss_acc = loss.detach()
    loss.backward()
    return loss_acc

def run_eval(model, dataloader, device):
    model.eval()
    dataloader.reset()

    with torch.no_grad():
        loss_accumulation = .0
        for _ in range(max_val_steps):
            loss = run_micro_step(model, dataloader, device)
            loss_accumulation += loss.detach() / max_val_steps
        
    if use_DDP:
        dist.all_reduce(loss_accumulation, op=dist.ReduceOp.AVG)

    model.train()
    return loss_accumulation

def run_hellaswag_eval(model, tokenizer, device):
    model.eval()

    acc, acc_norm = hellaswag_evaluation(model, tokenizer, device, DDP_rank, DDP_world_size)

    if use_DDP:
        acc, acc_norm = torch.tensor(acc, dtype=torch.float32, device=device), torch.tensor(acc_norm, dtype=torch.float32, device=device)       # need tensor to share accross GPUs
        dist.all_reduce(acc, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc_norm, op=dist.ReduceOp.AVG)        
        acc, acc_norm = acc.item(), acc_norm.item()

    model.train()

    return acc, acc_norm

def save_log(step, lr, norm, train_loss, val_loss, hellaswag_acc, hellaswag_acc_norm):
    log = {
        'step': step,
        'tokens': step * global_batch,
        'learning_rate': lr,
        'gradient_norm': norm.item(),
        'train_loss': train_loss.item(),
        'val_loss': val_loss.item() if val_loss else None,
        'hellaswag_acc': hellaswag_acc,
        'hellaswag_acc_norm': hellaswag_acc_norm,
    }

    with open(log_file, 'r') as file:
        try:
            all_logs = json.load(file)
        except (json.JSONDecodeError, ValueError):
            all_logs = []
    
    all_logs.append(log)

    with open(log_file, 'w') as file:
        json.dump(all_logs, file, indent=4)

def save_checkpoint(model, optimizer, step, val_loss, lr):
    checkpoint_path = save_dir + f'model_{step:05d}.pt'
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'step': step,
        'val_loss': val_loss.item() if val_loss else None,
        'optimizer': optimizer.state_dict(),
        'learning_rate': {
                        'current_lr': lr,
                        'max_lr': max_lr,
                        'min_lr': min_lr,
                        'warmup_steps': warmup_steps,
                        'max_steps': max_steps,
                    },
        'batch': {
            'B': B,
            'T': T,
            'global_batch': global_batch,
        }
    }

    torch.save(checkpoint, checkpoint_path)


main()