r"""
The fmoe.functions module contains functions that are directly warped up from
C/CUDA functions to complete distributed communication, computation and gradient
computation.
"""

import torch
from torch.autograd import Function
import mix_moe_cuda
from .utils import get_torch_default_comm


_moe_group = None


def ensure_comm(t, comm):
    if comm is None:
        comm = get_torch_default_comm()
    global _moe_group
    _moe_group = comm
    mix_moe_cuda.ensure_nccl(comm, t)


def get_moe_group():
    return _moe_group


def count_by_gate(gate, num_expert, world_size, my_rank, require_pos=True):
    with torch.no_grad():
        local_expert_count = torch.zeros(
            num_expert * world_size, device=gate.device, dtype=torch.int32
        )
        mix_moe_cuda.expert_count(gate, local_expert_count)
        local_expert_count = local_expert_count.long()

        import time
        start = int(time.perf_counter() * 1000000)
        if world_size > 1:
            all_expert_count = mix_moe_cuda.expert_all_gather(
                local_expert_count, num_expert, world_size
            )
        else:
            all_expert_count = local_expert_count
        end = int(time.perf_counter() * 1000000)
        if not require_pos:
            pos = None
        else:
            lec_cum = torch.cumsum(local_expert_count, dim=0).int()
            pos_size = lec_cum[-1].item()
            pos = torch.empty((pos_size,), device=gate.device, dtype=torch.long)
            mix_moe_cuda.assign_pos(lec_cum, gate, pos)
        all_expert_count = all_expert_count.reshape(world_size, num_expert*world_size)
        global_expert_count = all_expert_count[:,my_rank*num_expert:(my_rank+1)*num_expert]
    return pos, local_expert_count, global_expert_count,all_expert_count


def prepare_forward(gate, num_expert, world_size, my_rank):
    r"""
    Prepare necessary information from gate output for MoE computation.

    Args:
        gate: a 1-d Long Tensor representing the target expert of each input
        sample.
        num_expert: number of experts on each worker.
        world_size: number of workers that hold different experts.
        comm: the communicator of all workers in the expert-parallel group.
    """
    pos, local_expert_count, global_expert_count,all_expert_count = count_by_gate(gate, 
            num_expert, world_size, my_rank)
    with torch.no_grad():
        fwd_expert_count = global_expert_count.view(world_size,
                num_expert).sum(dim=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())
    return (
        pos,
        local_expert_count.cpu(),
        global_expert_count.cpu(),
        fwd_expert_count.cpu(),
        all_expert_count.cpu(),
        fwd_batch_size,
    )

def perpare_expert_assignment(gate, num_expert, world_size, require_pos = True):
    with torch.no_grad():
        local_expert_count = torch.zeros(
            num_expert * world_size, device=gate.device, dtype=torch.int32
        )
        mix_moe_cuda.expert_count(gate, local_expert_count)
        local_expert_count = local_expert_count.long()
        if world_size > 1:
            all_expert_count = mix_moe_cuda.expert_all_gather(
                local_expert_count, num_expert, world_size
            )
        else:
            all_expert_count = local_expert_count

        if not require_pos:
            pos = None
        else:
            lec_cum = torch.cumsum(local_expert_count, dim=0).int()
            pos_size = lec_cum[-1].item()
            pos = torch.empty((pos_size,), device=gate.device, dtype=torch.long)
            mix_moe_cuda.assign_pos(lec_cum, gate, pos)
        all_expert_count = all_expert_count.reshape(world_size, num_expert*world_size)
    return pos, all_expert_count.cpu()