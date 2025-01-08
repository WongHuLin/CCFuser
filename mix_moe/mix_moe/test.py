import torch.distributed
from mix_moe.transformer import MixMOETransformerMLP
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch


LOCAL_RANK = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
RANK = int(os.getenv('OMPI_COMM_WORLD_RANK'))
WORLD_SIZE = int(os.getenv('OMPI_COMM_WORLD_SIZE'))

import mix_moe_cuda
from linear import MixMOEInstance


def run(backend):
    # os.environ["RANK"] = str(proc_id)
    # os.environ["WORLD_SIZE"] = str(nprocs)
    # if int(os.environ["WORLD_SIZE"]) > 1:
    #     torch.distributed.init_process_group(backend="nccl")
    #     rank = torch.distributed.get_rank()
    #     world_size = torch.distributed.get_world_size()
    #     os.environ['OMPI_COMM_WORLD_SIZE'] = str(nprocs)
    #     os.environ['OMPI_COMM_WORLD_RANK'] = str(proc_id)

    # print(os.getenv('OMPI_COMM_WORLD_SIZE'))

    torch.distributed.init_process_group(backend=backend, rank=RANK, world_size=WORLD_SIZE)
    rank = RANK
    world_size = WORLD_SIZE

    print(rank, world_size)



    batch_size = 4096
    in_feat = 1024
    num_expert = 16
    top_k = 2
    block_size = 64

    # MixMOEInstance.instance(batch_size, num_expert, top_k, 
        # in_feat, in_feat*4, block_size)


    device = torch.device('cuda:'+str(rank))
    inp = torch.rand(batch_size, in_feat).half().to(device)
    inp.requires_grad = True

           
    mix_mlp = MixMOETransformerMLP(gate_type = {'type': 'top', 'k': 2},
            experts = {'type': 'ffn', 'num_local_expert': 16, 'hidden_size': in_feat*4, 'activation_fn': torch.nn.ReLU()},
            model_dim = in_feat,
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
            world_size = world_size, expert_rank=rank,
            total_batch_size= batch_size).to(device)
    mix_mlp.train()

    mix_mlp = mix_mlp.half()
    # _ = mix_mlp(inp)



        # warm up
    for _ in range(10):
        mix_mlp.async_expert_assignment(inp)
        o = mix_mlp.forward(inp)
        loss = o.sum()
        loss.backward()
    


    # for i in range(3):
    #     mix_mlp.forward(inp)

    # for i in range(10):
    #     mix_mlp.forward(inp)
    import time
    n_runs = 100
    tott = 0.0
    backt = 0.0
    maxt = 0.0
    sqtot = 0.0
    for i in range(n_runs):
        ts = time.time()
        mix_mlp.async_expert_assignment(inp)

        o = mix_mlp(inp)
        te = time.time()

        loss = o.sum()
        torch.cuda.synchronize()

        bts = time.time()
        loss.backward()
        bte = time.time()

        tott += te - ts
        sqtot += (te - ts) ** 2
        maxt = max(maxt, te - ts)
        backt += bte - bts
        torch.cuda.synchronize()

    gflops = (
        2e-9
        * n_runs
        * (
            in_feat * in_feat * 4 * batch_size * top_k * 2
            + batch_size * in_feat * num_expert
        )
        / tott
    )
    print(
        "Time mean/max/stdev/back {:.3f} {:.3f} {:.3f} {:.3f} ms, {:.3f} GFLOPs".format(
            tott * 1e3 / n_runs,
            maxt * 1e3,
            (sqtot / n_runs - (tott / n_runs) ** 2) * 1e3 * top_k / n_runs,
            backt * 1e3 / n_runs,
            gflops,
        )
    )

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument('--backend', type=str, help='backend')

    args = parser.parse_args()
    print(args.batch_size)

    run(args.backend)



# import torch
# import torch.nn as nn
# from fmoe import FMoETransformerMLP
# from fmoe.gates import NaiveGate
# from moe import BruteForceMoELinear
# import time
# import sys
# import os


# rank = None
# world_size = None
# dev_name_default = "cuda:0"


# class BruteForceMoE(nn.Module):
#     def __init__(
#         self,
#         num_expert=32,
#         d_model=1024,
#         d_hidden=4096,
#         world_size=1,
#         mp_group=None,
#         activation=torch.nn.functional.gelu,
#         gate=NaiveGate,
#         top_k=1,
#         pre_lnorm=False,
#     ):
#         assert world_size == 1, "Distributed brute force is not supported"
#         super().__init__()
#         self.mlp = BruteForceMoELinear(
#             activation, num_expert, d_model, d_hidden, 1, top_k
#         )
#         self.top_k = top_k
#         self.gate = gate(d_model, num_expert, world_size, top_k)
#         self.pre_lnorm = pre_lnorm
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.d_model = d_model

#     def forward(self, inp):
#         if self.pre_lnorm:
#             inp = self.layer_norm(inp)
#         gate_top_k_idx, gate_score = self.gate(inp)
#         inp = inp.repeat_interleave(repeats=self.top_k, dim=0)
#         x = self.mlp(inp, gate_top_k_idx, gate_score)
#         if not self.pre_lnorm:
#             x = self.layer_norm(x)
#         return x


# def benchmark_mlp(MOELayer, batch_size, in_feat, hidden_feat, num_expert, top_k):
#     torch.manual_seed(42 + rank)
#     torch.cuda.manual_seed(42 + rank)
#     if rank == 0:
#         print(
#             "Performance test of {} mm size {} {}x{} experts {}x{} topk {}".format(
#                 MOELayer.__name__,
#                 batch_size,
#                 in_feat,
#                 hidden_feat,
#                 world_size,
#                 num_expert,
#                 top_k,
#             )
#         )
#     if world_size > 1:
#         dev_name = "cuda"
#     else:
#         dev_name = dev_name_default

#     inp = torch.rand(batch_size, in_feat).cuda(dev_name)
#     inp.requires_grad = True

#     moe = MOELayer(
#         num_expert=num_expert,
#         d_model=in_feat,
#         d_hidden=hidden_feat,
#         world_size=world_size,
#         top_k=top_k,
#     ).cuda(dev_name)
#     moe.train()

#     # warm up
#     for _ in range(4):
#         _ = moe(inp)

#     n_runs = 16
#     tott = 0.0
#     backt = 0.0
#     maxt = 0.0
#     sqtot = 0.0
#     for i in range(n_runs):
#         ts = time.time()
#         o = moe(inp)
#         te = time.time()

#         loss = o.sum()

#         bts = time.time()
#         loss.backward()
#         bte = time.time()

#         tott += te - ts
#         sqtot += (te - ts) ** 2
#         maxt = max(maxt, te - ts)
#         backt += bte - bts

#     gflops = (
#         2e-9
#         * n_runs
#         * (
#             in_feat * hidden_feat * batch_size * top_k * 2
#             + batch_size * in_feat * num_expert
#         )
#         / tott
#     )
#     print(
#         "Time mean/max/stdev/back {:.3f} {:.3f} {:.3f} {:.3f} ms, {:.3f} GFLOPs".format(
#             tott * 1e3 / n_runs,
#             maxt * 1e3,
#             (sqtot / n_runs - (tott / n_runs) ** 2) * 1e3 * top_k / n_runs,
#             backt * 1e3 / n_runs,
#             gflops,
#         )
#     )


# if __name__ == "__main__":
#     if int(os.environ["WORLD_SIZE"]) > 1:
#         torch.distributed.init_process_group(backend="nccl")
#         rank = torch.distributed.get_rank()
#         world_size = torch.distributed.get_world_size()
#     else:
#         rank = 0
#         world_size = 1
#     batch_size = int(os.environ.get("BATCH_SIZE", "4096"))
#     d_model = int(os.environ.get("D_MODEL", "1024"))
#     d_hidden = int(os.environ.get("D_HIDDEN", "4096"))
#     num_expert = int(os.environ.get("NUM_EXPERT", "64"))
#     top_k = int(os.environ.get("TOP_K", "2"))
#     benchmark_mlp(FMoETransformerMLP, batch_size, d_model, d_hidden, num_expert, top_k)
#     if world_size == 1:
#         benchmark_mlp(BruteForceMoE, batch_size, d_model, d_hidden, num_expert, top_k)
