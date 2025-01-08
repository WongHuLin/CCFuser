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
parser.add_argument('--batch_size', type=int, default=16)
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

print(device)

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


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self.linear_o = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

        out = self.linear_o(att)
        return out

class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = MultiHeadSelfAttention(model_dim, model_dim, model_dim, 8)

        self._moe_layer = MixMOETransformerMLP(
            gate_type = {'type': 'top', 'k': top_value, 'fp32_gate': args.fp32_gate},
            experts = {'type': 'ffn', 'num_local_expert': num_local_experts, 'hidden_size': hidden_size, 'activation_fn': torch.nn.ReLU()},
            model_dim = model_dim,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
            total_batch_size = batch_size * num_tokens,
            world_size = dist_world_size,
            expert_rank = dist_rank,
        )
        self.attn.requires_grad_(False)
        self.attn_start_time = torch.cuda.Event(enable_timing=True)
        self.attn_end_time = torch.cuda.Event(enable_timing=True)


        # Summary of different parameter types: gate, local_experts

    def forward(self, input):

        self._moe_layer.async_expert_assignment(input)

        self.attn_start_time.record()
        attn_out = self.attn(input)
        self.attn_end_time.record()

        torch.cuda.synchronize()
        attn_time = self.attn_start_time.elapsed_time(self.attn_end_time)

        result = self._moe_layer(attn_out)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result, attn_time

    # Important setting 1: skip handling expert parameters by Pytorch DDP
    def add_param_to_skip_allreduce(self, param_name):
        if not hasattr(self, '_ddp_params_and_buffers_to_ignore'):
          self._ddp_params_and_buffers_to_ignore = list()
        self._ddp_params_and_buffers_to_ignore.append(param_name)

print(parallel_env.global_rank)
model = ExampleModel().half().to(device)

for name, param in model.named_parameters():
    if hasattr(param, 'skip_allreduce'):
        model.add_param_to_skip_allreduce(name)

if torch.distributed.is_initialized():
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.local_rank])

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

torch.manual_seed(0)

x = torch.rand(batch_size, num_tokens, model_dim).half().to(device)
y = torch.LongTensor(batch_size).random_(1).to(device)


tuples = (dist_world_size, args.dtype, model_dim, hidden_size, batch_size * num_tokens, num_local_experts, top_value, a2a_ffn_overlap_degree, device)
dist_print('[Benchmark] world_size = %s, dtype = %s, model_dim = %s, hidden_size = %s, samples = %s, num_local_experts = %s, topK = %s, a2a_ffn_overlap_degree = %s, device = `%s`' % tuples)

average_total_time, average_forward_time, average_backward_time, num_steps = 0, 0, 0, args.num_steps

forward_start_time = torch.cuda.Event(enable_timing=True)
forward_end_time = torch.cuda.Event(enable_timing=True)
backward_start_time = torch.cuda.Event(enable_timing=True)
backward_end_time = torch.cuda.Event(enable_timing=True)


for i in range(num_steps):

    optimizer.zero_grad()
    dist.barrier()

    forward_start_time.record()
    output, attn_time = model(x)
    forward_end_time.record()

    torch.cuda.synchronize()

    loss = F.nll_loss(output, y)

    backward_start_time.record()
    loss.backward()
    backward_end_time.record()

    torch.cuda.synchronize()

    forward_elapsed_time_ms = forward_start_time.elapsed_time(forward_end_time) - attn_time
    backward_elapsed_time_ms = backward_start_time.elapsed_time(backward_end_time)
    total_time_ms = forward_start_time.elapsed_time(backward_end_time) - attn_time



    tflops = (batch_size * num_tokens * model_dim * hidden_size) * 4 * args.top * 3 * 1e-12 / (total_time_ms/1000)
    dist_print('STEP-%s: step_time = %.6f ms, foward_time = %.6f, backward_time = %.6f, perf = %.2f tflops.' % (i, total_time_ms, forward_elapsed_time_ms, backward_elapsed_time_ms , tflops))

    if i + 10 >= num_steps:
        average_total_time += total_time_ms
        average_forward_time += forward_elapsed_time_ms
        average_backward_time += backward_elapsed_time_ms

average_total_time /= 10
average_forward_time /= 10
average_backward_time /= 10

dist_print('\n %s %s [Summary] Average synchronized step_time = %s ms, forward = %s , backward = %s .\n\n' % (args.name, num_local_experts, average_total_time, average_forward_time, average_backward_time))


model.module._moe_layer.close()