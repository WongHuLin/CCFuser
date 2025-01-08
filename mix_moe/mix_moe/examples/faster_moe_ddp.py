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

print("before barrier")
dist.barrier()
print("after barrier")

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

class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self._moe_layer = FMoETransformerMLP(
            num_expert=num_local_experts,
            d_model=model_dim,
            d_hidden=hidden_size,
            world_size=dist_world_size,
            top_k=2,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
        )

        # Summary of different parameter types: gate, local_experts

    def forward(self, input):
        result = self._moe_layer(input)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result

    # Important setting 1: skip handling expert parameters by Pytorch DDP
    def add_param_to_skip_allreduce(self, param_name):
        if not hasattr(self, '_ddp_params_and_buffers_to_ignore'):
          self._ddp_params_and_buffers_to_ignore = list()
        self._ddp_params_and_buffers_to_ignore.append(param_name)

model = ExampleModel().half().to(device)
model.train()


for name, param in model.named_parameters():
    if hasattr(param, 'skip_allreduce'):
        model.add_param_to_skip_allreduce(name)

if torch.distributed.is_initialized():
    print(args.local_rank, dist.get_rank(), "init")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
print("afsfsa")

x = torch.rand(batch_size, num_tokens, model_dim).half().to(device).requires_grad_(True)
y = torch.LongTensor(batch_size).random_(1).to(device)



# x = torch.tensor(torch.randn([batch_size, num_tokens, model_dim], dtype=torch.float32, device='cpu').detach().numpy(), dtype=torch.get_default_dtype(), requires_grad=False, device=device)


tuples = (dist_world_size, args.dtype, model_dim, hidden_size, batch_size * num_tokens, num_local_experts, top_value, a2a_ffn_overlap_degree, device)
dist_print('[Benchmark] world_size = %s, dtype = %s, model_dim = %s, hidden_size = %s, samples = %s, num_local_experts = %s, topK = %s, a2a_ffn_overlap_degree = %s, device = `%s`' % tuples)

average_total_time, average_forward_time, average_backward_time, num_steps = 0, 0, 0, args.num_steps



forward_start_time = torch.cuda.Event(enable_timing=True)
forward_end_time = torch.cuda.Event(enable_timing=True)
backward_start_time = torch.cuda.Event(enable_timing=True)
backward_end_time = torch.cuda.Event(enable_timing=True)



for i in range(num_steps):

    optimizer.zero_grad()

    forward_start_time.record()
    output = model(x)
    torch.cuda.synchronize()

    forward_end_time.record()

    loss = F.nll_loss(output, y)

    backward_start_time.record()
    loss.backward()
    backward_end_time.record()

    torch.cuda.synchronize()


    forward_elapsed_time_ms = forward_start_time.elapsed_time(forward_end_time) 
    backward_elapsed_time_ms = backward_start_time.elapsed_time(backward_end_time)
    total_time_ms = forward_start_time.elapsed_time(backward_end_time)



    tflops = (batch_size * num_tokens * model_dim * hidden_size) * 4 * args.top * 3 * 1e-12 / (total_time_ms/1000)
    dist_print('STEP-%s: step_time = %.6f ms, foward_time = %.6f, backward_time = %.6f, perf = %.2f tflops.' % (i, total_time_ms, forward_elapsed_time_ms, backward_elapsed_time_ms , tflops))

    if i + 60 >= num_steps:
        average_total_time += total_time_ms
        average_forward_time += forward_elapsed_time_ms
        average_backward_time += backward_elapsed_time_ms

average_total_time /= 60
average_forward_time /= 60
average_backward_time /= 60
dist_print('\n %s %s [Summary] Average synchronized step_time = %s ms, forward = %s , backward = %s .\n\n' % (args.name, num_local_experts, average_total_time, average_forward_time, average_backward_time))
