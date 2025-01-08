import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import argparse
import math

from tutel import system 
from mix_moe.transformer import MixMOETransformerMLP 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()


parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_tokens', type=int, default=512)
parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--num_local_experts', type=int, default=64)
parser.add_argument('--dtype', type=str, default='float16')
parser.add_argument('--fp32_gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
parser.add_argument('--a2a_ffn_overlap_degree', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--name', type=str, default='mix_moe')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument("--lr",default=5e-5, type=float, required=False, help="learning rate")
parser.add_argument("--seed",default=42, type=int, required=False, help="seed to replicate results")
parser.add_argument("--n_gpu",default=1, type=int, required=False, help="no of gpu available")
parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=False, help="gradient_accumulation_steps")
parser.add_argument("--num_workers",default=4, type=int, required=False, help="num of cpus available")
parser.add_argument("--num_train_epochs",default=1, type=int, required=False, help="no of epochs of training")
parser.add_argument("--output_dir",default='./output', type=str, required=False, help="path to save evaluation results")
parser.add_argument("--model_dir",default='./weights', type=str, required=False, help="path to save trained model")
parser.add_argument("--fp16",default=True, type=bool, required=False, help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument("--fp16_opt_level",default='O0', type=str, required=False, help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
parser.add_argument("--max_grad_norm",default=1.0, type=float, help="max gradient norm.")
parser.add_argument("--root_dir",default='./CNN/gpt2_1024_data', type=str, help="location of json dataset.")
parser.add_argument("--ids_file",default='./CNN/ids.json', type=str, help="location of train, valid and test file indexes")
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')

parser.add_argument('--moe', action='store_true',
                    help='replace position-wise ffn with moe position-wise ffn')
args = parser.parse_args()



parallel_env = system.init_data_model_parallel(backend='nccl' if args.device == 'cuda' else 'gloo')
dist_rank, dist_world_size, dist_print = parallel_env.global_rank, parallel_env.global_size, parallel_env.dist_print
args.local_rank = parallel_env.local_device.index

batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
hidden_size = args.hidden_size
num_local_experts = args.num_local_experts
top_value = args.top
a2a_ffn_overlap_degree = args.a2a_ffn_overlap_degree
device = parallel_env.local_device


if args.dtype == 'float32':
    torch.set_default_dtype(torch.float32)
elif args.dtype == 'float64':
    torch.set_default_dtype(torch.float64)
elif args.dtype == 'float16':
    torch.set_default_dtype(torch.float16)
elif args.dtype == 'bfloat16':
    torch.set_default_dtype(torch.bfloat16)
else:
    raise Exception('Unrecognized data type specified: %s' % args.dtype)


import argparse
from datetime import datetime
import os
import time

import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm
from moe_gpt import GPT2LMHeadModel
from transformers import GPT2Tokenizer

def add_special_tokens():
	""" Returns GPT2 tokenizer after adding separator and padding tokens """
	tokenizer = GPT2Tokenizer.from_pretrained('/root/models/gpt2')
	special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
	num_add_toks = tokenizer.add_special_tokens(special_tokens)
	return tokenizer

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

tokenizer = add_special_tokens()
model = GPT2LMHeadModel.from_pretrained('/root/models/gpt2')
model.resize_token_embeddings(len(tokenizer))
ignore_idx = tokenizer.pad_token_id
loss_fct = CrossEntropyLoss(ignore_index=ignore_idx) #ignores padding token for loss calculation
optimizer = AdamW(model.parameters(),lr=args.lr)




#-------------------------------------------------------------------------------------------------------
tokenizer = add_special_tokens()
model = GPT2LMHeadModel.from_pretrained('/root/models/gpt2')
model.resize_token_embeddings(len(tokenizer))
ignore_idx = tokenizer.pad_token_id
loss_fct = CrossEntropyLoss(ignore_index=ignore_idx) #ignores padding token for loss calculation
optimizer = AdamW(model.parameters(),lr=args.lr)


model = model.half().to(device)
model.train()

if args.multi_gpu:
    para_model = nn.parallel.DistributedDataParallel(model, device_ids=[dist_rank]).to(device)
else:
    para_model = model.to(device)


dist_print(para_model)

forward_start_time = torch.cuda.Event(enable_timing=True)
forward_end_time = torch.cuda.Event(enable_timing=True)
backward_start_time = torch.cuda.Event(enable_timing=True)
backward_end_time = torch.cuda.Event(enable_timing=True)
average_total_time, average_forward_time, average_backward_time, num_steps = 0, 0, 0, args.num_steps


batch = torch.load('./gpt2_data.pt')

c_batch = {}
for k in batch.keys():
    c_batch[k] = batch[k][0:args.batch_size].to(device)


inputs, labels = torch.tensor(c_batch['article']), torch.tensor(c_batch['article'])
inputs = inputs.to(device)
labels = labels.to(device)
para_model.train()


for i in range(num_steps):
    optimizer.zero_grad()

    forward_start_time.record()
   
    logits = para_model(inputs)[0]

    forward_end_time.record()
    torch.cuda.synchronize()

    idx = c_batch['sum_idx'] # index of separator token
    losses = []
    for j in range(args.batch_size):
        idx = batch['sum_idx'][j].item()
        shift_logits = logits[j][..., idx:-1, :].contiguous()
        shift_labels = labels[j][..., idx+1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(1)
        losses.append(loss)
    losses = torch.cat(losses, 0)
    loss = losses.sum()/args.gradient_accumulation_steps

    backward_start_time.record()
    loss.backward()
    optimizer.step()

    backward_end_time.record()
    torch.cuda.synchronize()
    forward_elapsed_time_ms = forward_start_time.elapsed_time(forward_end_time) 
    backward_elapsed_time_ms = backward_start_time.elapsed_time(backward_end_time)
    total_time_ms = forward_start_time.elapsed_time(backward_end_time)

    dist_print('STEP-%s: step_time = %.6f ms, foward_time = %.6f, backward_time = %.6f' % (i, total_time_ms, forward_elapsed_time_ms, backward_elapsed_time_ms))


    if i + 60 >= num_steps:
        average_total_time += total_time_ms

average_total_time /= 60
average_forward_time /= 60
average_backward_time /= 60
dist_print('\n naive gpt2 %s [Summary] Average synchronized step_time = %s ms,' % (num_local_experts, average_total_time,))

#-------------------------------------------------------------------------------------------------------



if args.moe:
    from fmoe.transformer import FMoETransformerMLP
    for i in [11]:
        moe_layer = FMoETransformerMLP(
            num_expert=num_local_experts,
            d_model=model_dim,
            d_hidden=hidden_size,
            world_size=dist_world_size,
            top_k=2,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True))
        _set_module(model, 'transformer.h.{}.mlp'.format(i), moe_layer)

model = model.half().to(device)
model.train()

for name, param in model.named_parameters():
    if hasattr(param, 'skip_allreduce'):
        model.add_param_to_skip_allreduce(name)

if args.multi_gpu:
    para_model = nn.parallel.DistributedDataParallel(model, device_ids=[dist_rank]).to(device)
    print("ddp")
else:
    para_model = model.to(device)


dist_print(para_model)

forward_start_time = torch.cuda.Event(enable_timing=True)
forward_end_time = torch.cuda.Event(enable_timing=True)
backward_start_time = torch.cuda.Event(enable_timing=True)
backward_end_time = torch.cuda.Event(enable_timing=True)
average_total_time, average_forward_time, average_backward_time, num_steps = 0, 0, 0, args.num_steps


batch = torch.load('./gpt2_data.pt')

c_batch = {}
for k in batch.keys():
    c_batch[k] = batch[k][0:args.batch_size].to(device)


inputs, labels = torch.tensor(c_batch['article']), torch.tensor(c_batch['article'])
inputs = inputs.to(args.device)
labels = labels.to(args.device)
para_model.train()


for i in range(num_steps):
    optimizer.zero_grad()

    forward_start_time.record()
   
    logits = para_model(inputs)[0]

    forward_end_time.record()
    torch.cuda.synchronize()

    idx = c_batch['sum_idx'] # index of separator token
    losses = []
    for j in range(args.batch_size):
        idx = batch['sum_idx'][j].item()
        shift_logits = logits[j][..., idx:-1, :].contiguous()
        shift_labels = labels[j][..., idx+1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(1)
        losses.append(loss)
    losses = torch.cat(losses, 0)
    loss = losses.sum()/args.gradient_accumulation_steps

    backward_start_time.record()
    loss.backward()
    optimizer.step()

    backward_end_time.record()
    torch.cuda.synchronize()
    forward_elapsed_time_ms = forward_start_time.elapsed_time(forward_end_time) 
    backward_elapsed_time_ms = backward_start_time.elapsed_time(backward_end_time)
    total_time_ms = forward_start_time.elapsed_time(backward_end_time)

    dist_print('STEP-%s: step_time = %.6f ms, foward_time = %.6f, backward_time = %.6f' % (i, total_time_ms, forward_elapsed_time_ms, backward_elapsed_time_ms))


    if i + 60 >= num_steps:
        average_total_time += total_time_ms

average_total_time /= 60
average_forward_time /= 60
average_backward_time /= 60
dist_print('\n fastermoe gpt2 %s [Summary] Average synchronized step_time = %s ms,' % ( num_local_experts, average_total_time,))



#----------------------------------------------------------------
tokenizer = add_special_tokens()
model = GPT2LMHeadModel.from_pretrained('/root/models/gpt2')
model.resize_token_embeddings(len(tokenizer))
ignore_idx = tokenizer.pad_token_id
loss_fct = CrossEntropyLoss(ignore_index=ignore_idx) #ignores padding token for loss calculation
optimizer = AdamW(model.parameters(),lr=args.lr)




if args.moe:
    for i in [11]:
        moe_layer = MixMOETransformerMLP(
            gate_type = {'type': 'top', 'k': top_value, 'fp32_gate': args.fp32_gate},
            experts = {'type': 'ffn', 'num_local_expert': num_local_experts, 'hidden_size': hidden_size, 'activation_fn': torch.nn.ReLU()},
            model_dim = model_dim,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
            total_batch_size = batch_size * num_tokens,
            world_size = dist_world_size,
            expert_rank = dist_rank,)
        _set_module(model, 'bert.encoder.layer.{}.ffn'.format(i), moe_layer)

model = model.half().to(device)
model.train()

for name, param in model.named_parameters():
    if hasattr(param, 'skip_allreduce'):
        model.add_param_to_skip_allreduce(name)

if args.multi_gpu:
    para_model = nn.parallel.DistributedDataParallel(model, device_ids=[dist_rank]).to(device)
    print("ddp")
else:
    para_model = model.to(device)


dist_print(para_model)

forward_start_time = torch.cuda.Event(enable_timing=True)
forward_end_time = torch.cuda.Event(enable_timing=True)
backward_start_time = torch.cuda.Event(enable_timing=True)
backward_end_time = torch.cuda.Event(enable_timing=True)
average_total_time, average_forward_time, average_backward_time, num_steps = 0, 0, 0, args.num_steps


batch = torch.load('./gpt2_data.pt')

c_batch = {}
for k in batch.keys():
    c_batch[k] = batch[k][0:args.batch_size].to(device)


inputs, labels = torch.tensor(c_batch['article']), torch.tensor(c_batch['article'])
inputs = inputs.to(args.device)
labels = labels.to(args.device)
para_model.train()


for i in range(num_steps):
    optimizer.zero_grad()

    forward_start_time.record()
   
    logits = para_model(inputs)[0]

    forward_end_time.record()
    torch.cuda.synchronize()

    idx = c_batch['sum_idx'] # index of separator token
    losses = []
    for j in range(args.batch_size):
        idx = batch['sum_idx'][j].item()
        shift_logits = logits[j][..., idx:-1, :].contiguous()
        shift_labels = labels[j][..., idx+1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(1)
        losses.append(loss)
    losses = torch.cat(losses, 0)
    loss = losses.sum()/args.gradient_accumulation_steps

    backward_start_time.record()
    loss.backward()
    optimizer.step()

    backward_end_time.record()
    torch.cuda.synchronize()
    forward_elapsed_time_ms = forward_start_time.elapsed_time(forward_end_time) 
    backward_elapsed_time_ms = backward_start_time.elapsed_time(backward_end_time)
    total_time_ms = forward_start_time.elapsed_time(backward_end_time)

    dist_print('STEP-%s: step_time = %.6f ms, foward_time = %.6f, backward_time = %.6f' % (i, total_time_ms, forward_elapsed_time_ms, backward_elapsed_time_ms))


    if i + 60 >= num_steps:
        average_total_time += total_time_ms

average_total_time /= 60
average_forward_time /= 60
average_backward_time /= 60
dist_print('\n mix_moe gpt2 %s [Summary] Average synchronized step_time = %s ms,' % (num_local_experts, average_total_time,))


exit()
