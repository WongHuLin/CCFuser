r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .layers import MixMOE
from .linear import FMoELinear
import math
import mix_moe_cuda

class FMoELinear(nn.Module):
    r"""
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(
        self,
        num_expert: int,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        rank: int = 0,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """
        # x = MOELinear.apply(inp.type_as(self.weight), fwd_expert_count, self.weight, self.bias)
        # return x
        pass

    def extra_repr(self) -> str:
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        # bias is left to zero, similar as megatron

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            torch.nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

from .linear import MixMOELinear1, MixMOELinear2, MixMOEInstance


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, pos, moe_inp_block_infos):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = MixMOELinear1.apply(inp, pos, moe_inp_block_infos, self.htoh4.weight, self.htoh4.bias)
        x = self.activation(x)
        x = MixMOELinear2.apply(x, pos, moe_inp_block_infos, self.h4toh.weight, self.h4toh.bias)

        return x


import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from .functions import ensure_comm, perpare_expert_assignment
import tree


class AsyncExpertAssignment:
    def __init__(self, rank_id, world_size, popular_expert_num, expert_num, total_batch_size):
        self.executor = ThreadPoolExecutor()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_event_loop, args=(self.loop,))
        self.thread.start()
        self.tasks_done_event = threading.Event()  # 事件标志位
        self.result = None  # 用于存储结果
        self.my_rank = rank_id
        self.local_rank = rank_id % torch.cuda.device_count()
        self.world_size = world_size
        self.popular_expert_num = popular_expert_num
        self.local_expert_num = popular_expert_num + expert_num
        self.total_num_expert = expert_num*world_size
        self.total_batch_size = total_batch_size
        self.num_expert = expert_num
        self.block_size = 64

        self.assignment_plan = {}
        for rank in range(world_size):
            self.assignment_plan[rank] = [ i + rank*self.num_expert for i in range(self.num_expert)]


        self.expert_ids_tensor = torch.empty([2 * total_batch_size//self.block_size * world_size + self.total_num_expert],
                                             dtype = torch.int32, device = torch.device('cuda:'+ str(self.local_rank)))

        self.expert_id_indexs_tensor = torch.empty([2 * total_batch_size//self.block_size * world_size + self.total_num_expert],
                                             dtype = torch.int32, device = torch.device('cuda:'+ str(self.local_rank)))
        self.ranks_tensor = torch.empty([2 * total_batch_size//self.block_size * world_size + self.total_num_expert],
                                             dtype = torch.int32, device = torch.device('cuda:'+ str(self.local_rank)))
        self.lens_tensor = torch.empty([2 * total_batch_size//self.block_size * world_size + self.total_num_expert],
                                             dtype = torch.int32, device = torch.device('cuda:'+ str(self.local_rank)))
        self.statr_addrs_tensor = torch.empty([2 * total_batch_size//self.block_size * world_size + self.total_num_expert],
                                             dtype = torch.int32, device = torch.device('cuda:'+ str(self.local_rank)))
        self.expert_block_ids_tensor = torch.empty([2 * total_batch_size//self.block_size * world_size + self.total_num_expert],
                                             dtype = torch.int32, device = torch.device('cuda:'+ str(self.local_rank)))
        self.expert_block_id_idxs_tensor = torch.empty([self.local_expert_num+1],
                                             dtype = torch.int32, device = torch.device('cuda:'+ str(self.local_rank)))


    def run_event_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def check_plan_balanced(self, all_expert_count):
        return False
        pass

    def divide_into_block(self, expert_partition_plan, all_expert_count, topk_indices,):
        block_size = 64
        local_block_info = []
        remote_block_info = []


        all_expert_addr_offset = all_expert_count.cumsum(-1)
        all_expert_addr_offset = torch.cat((torch.zeros([self.world_size,1], dtype=torch.int32),all_expert_addr_offset),-1)
        all_expert_addr_offset = all_expert_addr_offset.index_select(-1,torch.tensor(expert_partition_plan[self.my_rank]))
        all_expert_addr_offset_list = all_expert_addr_offset.tolist()

        # add popuplar expert block data
        popular_data_list = all_expert_count[self.my_rank, topk_indices]
        topk_indices_list = topk_indices.tolist()
        for i in range(len(popular_data_list)):
            data_size = popular_data_list[i]
            popular_expert_idx = topk_indices_list[i]
            block_num = data_size // block_size
            temp = [{"expert_id":popular_expert_idx, "rank":self.my_rank, "len": block_size, "start_addr": all_expert_addr_offset_list[self.my_rank][i] + block_size*k} for k in range(block_num)]
            local_block_info.extend(temp)
            if data_size % block_size != 0:
                remote_block_info.append({"expert_id": popular_expert_idx, "rank":self.my_rank, "len": data_size % block_size, "start_addr":all_expert_addr_offset_list[self.my_rank][i] + block_num * block_size})


        all_expert_count[:, topk_indices] = 0
        all_expert_addr_len = all_expert_count.index_select(-1,torch.tensor(expert_partition_plan[self.my_rank]))
        all_expert_addr_len_list = all_expert_addr_len.tolist()


        
        for j in range(len(self.assignment_plan[self.my_rank])):
            for i in range(self.world_size):
                if all_expert_addr_len_list[i][j] >= block_size:
                    block_num = all_expert_addr_len_list[i][j] // block_size
                    if i != self.my_rank:
                        temp = [{"expert_id":expert_partition_plan[self.my_rank][j], "rank":i, "len": block_size, "start_addr": all_expert_addr_offset_list[i][j] + block_size*k} for k in range(block_num)]
                        remote_block_info.extend(temp)
                    else:
                        temp = [{"expert_id":expert_partition_plan[self.my_rank][j], "rank":i, "len": block_size, "start_addr": all_expert_addr_offset_list[i][j] + block_size*k} for k in range(block_num)]
                        local_block_info.extend(temp)


                    if all_expert_addr_len_list[i][j] % block_size != 0:
                        remote_block_info.append({"expert_id":expert_partition_plan[self.my_rank][j], "rank":i, "len": all_expert_addr_len_list[i][j] % block_size, "start_addr":all_expert_addr_offset_list[i][j] + block_num * block_size})


        e = time.time()

        from itertools import zip_longest
        block_info = [item for pair in zip_longest(local_block_info, remote_block_info) for item in pair if item is not None]

        keys = block_info[0].keys()  # 获取所有的键
        result = {key: list(map(lambda d: d[key], block_info)) for key in keys}
        expert_index_map =   {num: idx for idx, num in enumerate(expert_partition_plan[self.my_rank])}
        
        local_expert_id_indexs = [expert_index_map[expert_id] for expert_id in result['expert_id']]
        g = time.time()

        from collections import defaultdict
        index_map = defaultdict(list)
        for idx, value in enumerate(local_expert_id_indexs):
            index_map[value].append(idx)


        expert_block_ids = []
        expert_block_id_idxs = [0]

        for i in range(self.local_expert_num):
            indices = index_map.get(i, [])  # 获取对应值的下标列表
            expert_block_ids.extend(indices)
            expert_block_id_idxs.append(len(expert_block_ids))

        total_block_size = len(block_info)
        remote_block_num = len(remote_block_info)


        mix_moe_cuda.cpy_data2tensor(result['expert_id'], self.expert_ids_tensor)
        mix_moe_cuda.cpy_data2tensor(local_expert_id_indexs, self.expert_id_indexs_tensor)
        mix_moe_cuda.cpy_data2tensor(result['rank'], self.ranks_tensor)
        mix_moe_cuda.cpy_data2tensor(result['len'], self.lens_tensor)
        mix_moe_cuda.cpy_data2tensor(result['start_addr'], self.statr_addrs_tensor)
        mix_moe_cuda.cpy_data2tensor(expert_block_ids, self.expert_block_ids_tensor)
        mix_moe_cuda.cpy_data2tensor(expert_block_id_idxs, self.expert_block_id_idxs_tensor)

        # print(result['expert_id'], self.expert_ids_tensor[0:len(result['expert_id'])])
        # print(self.my_rank, "local_expert_id_indexs", local_expert_id_indexs)
        # print(self.my_rank, "rank", result['rank'])
        # print(self.my_rank, "len", result['len'])
        # print(self.my_rank, "start_addr", result['start_addr'])
        # print(self.my_rank, "expert_block_ids", expert_block_ids)
        # print(self.my_rank, "expert_block_id_idxs", expert_block_id_idxs)

        return {'expert_ids': self.expert_ids_tensor,
                'local_expert_id_indexs': self.expert_id_indexs_tensor,
                'ranks': self.ranks_tensor,
                'lens': self.lens_tensor,
                'start_addrs': self.statr_addrs_tensor,
                'remote_block_num': remote_block_num,
                'total_block_num': total_block_size,
                'expert_block_ids': self.expert_block_ids_tensor,
                'expert_block_id_idxs': self.expert_block_id_idxs_tensor}

    def generate_kernel_paras(self, all_expert_count, expert_partition_plan, topk_indices, device):

        block_infos = self.divide_into_block(expert_partition_plan, all_expert_count, topk_indices)
        return block_infos

    def get_balanced_assignment(self, all_expert_count):
        if self.check_plan_balanced(all_expert_count):
            return


        def check_expert_is_planned(expert_id, partition_plan):
            for key, value in partition_plan.items():
                if expert_id in value:
                    return True
            return False

        def partition_expert(gpu_topo, partition_expert_count, partition_expert_idx, local_expert_ids, partition_expert_num, local_expert_num):
            tmp_partition_plan = {}
            total_expert_count = len(partition_expert_count)

            for i in range(len(gpu_topo)):
                tmp_partition_plan[i] = []

            for i in range(total_expert_count):
                if partition_expert_count[i] < 64:
                    break
                expert_group_id_idx = partition_expert_idx[i] % partition_expert_num
                expert_group_id = local_expert_ids[expert_group_id_idx]
                if not check_expert_is_planned(expert_group_id, tmp_partition_plan):
                    partition_id = partition_expert_idx[i] // partition_expert_num
                    if len(tmp_partition_plan[partition_id]) < local_expert_num:
                        tmp_partition_plan[partition_id].append(expert_group_id)

            unallocted_expert = []
            for i in range(len(gpu_topo)):
                for world_id in gpu_topo[i]:
                    for expert_id in self.assignment_plan[world_id]:
                        if not check_expert_is_planned(expert_id, tmp_partition_plan):
                            if len(tmp_partition_plan[i]) < local_expert_num:
                                tmp_partition_plan[i].append(expert_id)
                            else:
                                unallocted_expert.append(expert_id)
            index_start = 0
            for key, value in tmp_partition_plan.items():
                if len(value) < local_expert_num:
                    tmp_partition_plan[key].extend(unallocted_expert[index_start : index_start + local_expert_num - len(value)])
                if index_start >= len(unallocted_expert):
                    break

            return tmp_partition_plan

        def perpare_expert_count(all_expert_count:torch.Tensor, partition_groups, expert_ids = None):
            partiton_expert_count = []
            for i in range(len(partition_groups)):
                if expert_ids == None:
                    tmp = torch.index_select(all_expert_count, 0, torch.tensor(partition_groups[i], dtype = torch.int32))
                    tmp = torch.sum(tmp, 0)
                    partiton_expert_count.append(tmp)
                else:
                    tmp = torch.index_select(
                                torch.index_select(all_expert_count, 0, torch.tensor(partition_groups[i], dtype = torch.int32)),
                                1,
                                torch.tensor(expert_ids, dtype = torch.int32)
                                )
                    tmp = torch.sum(tmp, 0)
                    partiton_expert_count.append(tmp)

            sorted_expert_count, sorted_expert_idx = torch.cat(partiton_expert_count, 0).flatten().sort(descending=True)

            sorted_expert_count = sorted_expert_count.tolist()
            sorted_expert_idx = sorted_expert_idx.tolist()
            return sorted_expert_count, sorted_expert_idx

        def flatten_array(arr):
            flatten_result = []
            def _flatten(elem):
                if isinstance(elem, list):
                    for item in elem:
                        _flatten(item)
                else:
                    flatten_result.append(elem)
            _flatten(arr)
            return flatten_result

        def temp(gpu_topo, all_expert_count, expert_ids):
            if len(gpu_topo) == 1:
                return [{gpu_topo[0] : expert_ids}]
            partition_groups = []


            for i in range(len(gpu_topo)):
                tmp = flatten_array(gpu_topo[i])
                partition_groups.append(tmp)

            sorted_expert_count, sorted_expert_idx = perpare_expert_count(all_expert_count, partition_groups, expert_ids)
            tmp_partition_plan = partition_expert(partition_groups, sorted_expert_count, sorted_expert_idx, expert_ids, len(expert_ids), self.num_expert * len(partition_groups[0]))

            result = []
            for i in range(len(tmp_partition_plan)):
                tmp = temp(gpu_topo[i], all_expert_count, tmp_partition_plan[i])
                result.extend(tmp)

            return result

        gpu_topo = [[0],[1]]

        result = temp(gpu_topo, all_expert_count, list(range(self.num_expert * self.world_size)))
        result_1 = {}
        for it in result:
            result_1.update(it)
        self.assignment_plan = result_1
        pass

    def select_popular_expert(self, all_expert_count):
        expert_count = torch.sum(all_expert_count, 0)
        topk_values, topk_indices = torch.topk(expert_count, self.popular_expert_num)
        topk_indices_list = topk_indices.tolist()
        for rank in range(self.world_size):
            self.assignment_plan[rank] = []
            self.assignment_plan[rank].extend(topk_indices_list)
            self.assignment_plan[rank].extend([ i + rank*self.num_expert for i in range(self.num_expert) if  i + rank*self.num_expert  not in topk_indices_list])

        return  topk_indices

    def process_data(self, gate, attn_inp, num_expert, world_size):

        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, None)

            tree.map_structure(ensure_comm_func, attn_inp)

        import time
        a = time.time()
        gate_top_k_idx, gate_score = gate(attn_inp)
        device = attn_inp.device
        b = time.time()


        pos, all_expert_count = perpare_expert_assignment(gate_top_k_idx, num_expert, world_size)
        c = time.time()

        topk_indices = self.select_popular_expert(all_expert_count)
        # self.get_balanced_assignment(unpopular_all_expert_count)
        d = time.time()

        moe_inp_block_infos = self.generate_kernel_paras(all_expert_count,  self.assignment_plan, topk_indices, device)
        e = time.time()


        return pos, gate_score,moe_inp_block_infos

    async def async_process_data(self, gate, attn_inp, num_expert, world_size):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self.process_data, gate, attn_inp, num_expert, world_size)
        self.result = result  # 存储结果
        self.tasks_done_event.set()  # 设置标志位表示任务完成

    def add_task(self, gate, attn_inp, num_expert, world_size):
        self.tasks_done_event.clear()  # 清除标志位
        asyncio.run_coroutine_threadsafe(self.async_process_data(gate, attn_inp, num_expert, world_size), self.loop)

    def get_result(self):
        return self.result

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.executor.shutdown()

    def wait_for_completion(self):
        self.tasks_done_event.wait()  # 等待任务完成

    def __del__(self):
        self.close()

class MixMOETransformerMLP(MixMOE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        gate_type,
        experts,
        model_dim: int,
        expert_rank,
        world_size,
        scan_expert_func=None,
        total_batch_size = 4096,
        expert_dp_comm="none",
        activation=torch.nn.GELU(),
        mix_moe = None,
        **kwargs
    ):
        popular_expert_num = 3

        # 初始化mix_moe
        super().__init__(num_expert=experts['num_local_expert'], d_model=model_dim, ffn_hidden_size = experts['hidden_size'], expert=None, my_rank=expert_rank, world_size=world_size, **kwargs)
        self.experts = _Expert(experts['num_local_expert'] + popular_expert_num, model_dim, experts['hidden_size'], activation, expert_rank)
        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)

        self.init_mix_moe(total_batch_size)
        self.mark_parallel_comm(expert_dp_comm)
        self.async_ = False
        self.async_expert_assign = AsyncExpertAssignment(expert_rank, world_size, popular_expert_num, experts['num_local_expert'], total_batch_size)

    def async_expert_assignment(self, attn_inp: torch.Tensor):
        inp = attn_inp.reshape(-1, self.d_model)
        self.async_expert_assign.add_task(self.gate, inp, self.num_expert, self.world_size)
        pass

    def __del__(self):
        self.async_expert_assign.close()

    def close(self):
        self.async_expert_assign.close()


    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        # self.async_expert_assign.process_data(self.gate, inp, self.num_expert, self.world_size)

        # self.async_expert_assign.add_task(self.gate, inp, self.num_expert, self.world_size)
        import time
        start = int(time.perf_counter() * 1000000)
        end = int(time.perf_counter() * 1000000)
        if self.async_:
            self.async_expert_assign.wait_for_completion()
            pos, gate_score,moe_inp_block_infos = self.async_expert_assign.result
        else:
            pos, gate_score,moe_inp_block_infos = self.async_expert_assign.process_data(self.gate, inp, self.num_expert, self.world_size)
        # if self.my_rank == 0:
        #     print(moe_inp_block_infos)
        output = super().forward(inp, pos, gate_score,moe_inp_block_infos)
        return output.reshape(original_shape)
