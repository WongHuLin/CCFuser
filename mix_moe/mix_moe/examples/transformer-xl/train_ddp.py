import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import argparse

from tutel import system 
from fmoe.transformer import FMoETransformerMLP
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

parser = argparse.ArgumentParser()


parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--num_tokens', type=int, default=512)
parser.add_argument('--model_dim', type=int, default=1024)
parser.add_argument('--hidden_size', type=int, default=4096)
parser.add_argument('--num_local_experts', type=int, default=64)
parser.add_argument('--dtype', type=str, default='float16')
parser.add_argument('--fp32_gate', default=False, action='store_true')
parser.add_argument('--top', type=int, default=2)
parser.add_argument('--a2a_ffn_overlap_degree', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--name', type=str, default='mix_moe')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--data', type=str, default='../data/enwik8/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='enwik8',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=8,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=64,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=512,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1024,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=10000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=512,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=128,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=512,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=5,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='/workspace/fastmoe/examples/transformer-xl/scripts/run_enwik8_base_moe', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')
parser.add_argument('--moe', action='store_true',
                    help='replace position-wise ffn with moe position-wise ffn')
parser.add_argument('--moe-num-expert', type=int, default=64,
                    help='number of experts in MoE')
parser.add_argument('--moe-top-k', type=int, default=2,
                    help='top_k experts in hard gate of moe')
args = parser.parse_args()
args.tied = not args.not_tied
assert args.moe_num_expert >= args.moe_top_k, "must have moe-num-expert >= moe-top_k"


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

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

device = parallel_env.local_device
device = torch.device(str(device))

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



###############################################################################
# Build the model
###############################################################################

ntokens = 100
args.n_token = ntokens

cutoffs, tie_projs = [], [False]
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    if not args.fp16:
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    if args.name == 'mix_moe':
        from mem_transformer_mix_moe import MemTransformerLM
        model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,
            moe=args.moe, moe_num_expert=args.moe_num_expert, moe_top_k=args.moe_top_k,
            world_size_=dist_world_size, expert_rank_=dist_rank)
    elif args.name == 'fastermoe':
        from mem_transformer import MemTransformerLM
        model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,
            moe=args.moe, moe_num_expert=args.moe_num_expert, moe_top_k=args.moe_top_k,
            world_size_=dist_world_size, expert_rank_=dist_rank)
    else:
        from mem_transformer import MemTransformerLM
        model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
                args.d_head, args.d_inner, args.dropout, args.dropatt,
                tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                same_length=args.same_length, attn_type=args.attn_type,
                clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,
                moe=args.moe, moe_num_expert=args.moe_num_expert, moe_top_k=args.moe_top_k,
                world_size_=dist_world_size, expert_rank_=dist_rank)
        model.apply(weights_init)
        model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

model = model.half().to(device)

model.train()

for name, param in model.named_parameters():
    if hasattr(param, 'skip_allreduce'):
        model.add_param_to_skip_allreduce(name)

if args.multi_gpu:
    para_model = nn.parallel.DistributedDataParallel(model, device_ids=[dist_rank]).to(device)
else:
    para_model = model.to(device)


# data_loader = create_data_loader(dataset, dist_rank, dist_world_size)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


forward_start_time = torch.cuda.Event(enable_timing=True)
forward_end_time = torch.cuda.Event(enable_timing=True)
backward_start_time = torch.cuda.Event(enable_timing=True)
backward_end_time = torch.cuda.Event(enable_timing=True)


x = torch.LongTensor(batch_size, args.mem_len).random_(1, 50).to(device)
y = torch.LongTensor(batch_size, args.mem_len).random_(1, 50).to(device)
average_total_time, average_forward_time, average_backward_time, num_steps = 0, 0, 0, args.num_steps

for i in range(num_steps):

    forward_start_time.record()

    optimizer.zero_grad()
    ret = para_model(x, y)
    
    forward_end_time.record()
    backward_start_time.record()
    
    loss, mems = ret[0], ret[1:]
    loss = loss.float().mean().type_as(loss)
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
dist_print('\n %s transformer-xl %s [Summary] Average synchronized step_time = %s ms,' % (args.name, num_local_experts, average_total_time,))







# #--------------------------------------------------------------------------
# args.moe = False

# if args.restart:
#     with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
#         model = torch.load(f)
#     if not args.fp16:
#         model = model.float()
#     model.apply(update_dropout)
#     model.apply(update_dropatt)
# else:
#     from mem_transformer import MemTransformerLM
#     model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
#             args.d_head, args.d_inner, args.dropout, args.dropatt,
#             tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
#             tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
#             ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
#             same_length=args.same_length, attn_type=args.attn_type,
#             clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,
#             moe=args.moe, moe_num_expert=args.moe_num_expert, moe_top_k=args.moe_top_k,
#             world_size_=dist_world_size, expert_rank_=dist_rank)
#     model.apply(weights_init)
#     model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
# args.n_all_param = sum([p.nelement() for p in model.parameters()])
# args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

# model = model.half().to(device)

# model.train()

# for name, param in model.named_parameters():
#     if hasattr(param, 'skip_allreduce'):
#         model.add_param_to_skip_allreduce(name)

# if args.multi_gpu:
#     para_model = nn.parallel.DistributedDataParallel(model, device_ids=[dist_rank]).to(device)
# else:
#     para_model = model.to(device)


# # data_loader = create_data_loader(dataset, dist_rank, dist_world_size)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


# forward_start_time = torch.cuda.Event(enable_timing=True)
# forward_end_time = torch.cuda.Event(enable_timing=True)
# backward_start_time = torch.cuda.Event(enable_timing=True)
# backward_end_time = torch.cuda.Event(enable_timing=True)


# x = torch.LongTensor(batch_size, args.mem_len).random_(1, 50).to(device)
# y = torch.LongTensor(batch_size, args.mem_len).random_(1, 50).to(device)
# average_total_time, average_forward_time, average_backward_time, num_steps = 0, 0, 0, args.num_steps

# for i in range(num_steps):

#     forward_start_time.record()

#     optimizer.zero_grad()
#     ret = para_model(x, y)
    
#     forward_end_time.record()
#     backward_start_time.record()
    
#     loss, mems = ret[0], ret[1:]
#     loss = loss.float().mean().type_as(loss)
#     loss.backward()

    
#     optimizer.step()
#     backward_end_time.record()
#     torch.cuda.synchronize()
#     forward_elapsed_time_ms = forward_start_time.elapsed_time(forward_end_time) 
#     backward_elapsed_time_ms = backward_start_time.elapsed_time(backward_end_time)
#     total_time_ms = forward_start_time.elapsed_time(backward_end_time)
#     dist_print('STEP-%s: step_time = %.6f ms, foward_time = %.6f, backward_time = %.6f' % (i, total_time_ms, forward_elapsed_time_ms, backward_elapsed_time_ms))


#     if i + 60 >= num_steps:
#         average_total_time += total_time_ms

# average_total_time /= 60
# average_forward_time /= 60
# average_backward_time /= 60
# dist_print('\n %s transformer-xl %s [Summary] Average synchronized step_time = %s ms,' % (args.name, num_local_experts, average_total_time,))

# #--------------------------------------------------------------------------






