import torch
import tree
import torch.nn as nn
from .gates import NaiveGate
from .functions import ensure_comm, prepare_forward
from .linear import MixMOEInstance

import mix_moe_cuda

def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _local_scatter(inp, pos):
    inp_buf = torch.index_select(inp, 0, pos)
    return inp_buf


def _local_gather(inp, pos, out_batch_size, maybe_overlap=True):
    inp_buf = torch.zeros(out_batch_size, inp.shape[-1],
            dtype=inp.dtype, device=inp.device)
    if maybe_overlap:
        inp_buf.index_add_(0, pos, inp)
    else:
        inp_buf.index_copy_(0, pos, inp)
    return inp_buf

class MixMOE(nn.Module):
    def __init__(
        self,
        num_expert=16,
        d_model=1024,
        ffn_hidden_size = 1024,
        world_size=1,
        top_k=2,
        gate=NaiveGate,
        my_rank = 0,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        mix_moe = None
    ) -> None:
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.ffn_hidden_size = ffn_hidden_size
        self.world_size = world_size
        self.my_rank = my_rank
        self.top_k = top_k
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.mix_moe = mix_moe
        self.expert_partition = []
        for i in range(self.world_size):
            self.expert_partition.append([j + i * self.num_expert for j in range(self.num_expert)])


    def init_mix_moe(self, total_batch_size):
        if self.mix_moe is None:
            self.mix_moe = MixMOEInstance.instance(total_batch_size, self.num_expert, self.top_k, 
                        self.d_model, self.ffn_hidden_size, 64,)
           


    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def balance_compute(self, all_expert_count: torch.Tensor):


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
                    for expert_id in self.expert_partition[world_id]:
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

                
        gpu_topo = [[[0],[1]], [[2],[3]]]
        result = temp(gpu_topo, all_expert_count, list(range(self.num_expert * self.world_size)))
        result_1 = {}
        for it in result:
            result_1.update(it)

        # expert_repartition = []
        # for i in range(self.world_size):
        #     tmp = [ [] for i in range(self.world_size)]
        #     for expert_id in result_1[i]:
        #         for j in range(self.world_size):
        #             if expert_id in self.expert_partition[j]:
        #                 tmp[j].append(expert_id)
        #                 break
        #     expert_repartition.extend(tmp)

        # expert_repartition_idx = [0]
        

        # # expert_repartition_res = []
        # for expert_ids in expert_repartition:
        #     # expert_repartition_res.extend(expert_ids)
        #     # expert_repartition_idx.append(len(expert_repartition_res))
        #     expert_repartition_idx.append(len(expert_ids) + expert_repartition_idx[-1])


        # import functools
        # import operator
        # expert_repartition = functools.reduce(operator.iconcat, expert_repartition, [])

        # import mix_moe_cuda
        

        # weights = mix_moe_cuda.expert_rebalance(self.experts.htoh4.weight, self.experts.htoh4.bias, expert_repartition,
        #             expert_repartition_idx, self.expert_partition[self.my_rank], self.my_rank,
        #             self.world_size)
        
        # if self.my_rank == 0:
        # print(expert_repartition_res)



        # sorted_expert_count, sorted_expert_idx = all_expert_count.flatten().sort(descending=True)

        # sorted_expert_count = sorted_expert_count.tolist()
        # sorted_expert_idx = sorted_expert_idx.tolist()

        # total_expert_num = self.world_size * self.num_expert


        # world_num = self.world_size


        # partition_plan = {}
        # for i in range(world_num):
        #     partition_plan[i] = []
        # for i in range(len(sorted_expert_count)):
        #     expert_id = sorted_expert_idx[i] % total_expert_num
        #     if not check_expert_is_planned(expert_id, partition_plan):
        #         world_id = sorted_expert_idx[i] // total_expert_num
        #         if len(partition_plan[world_id]) < self.num_expert:
        #             partition_plan[world_id].append(expert_id)
        
        return result_1
    
    def divide_into_block(self, expert_partition_plan, all_expert_addr_len, all_expert_addr_offset, device):
        all_expert_addr_len_list = all_expert_addr_len.tolist()
        all_expert_addr_offset_list = all_expert_addr_offset.tolist()


        local_block_info = []
        remote_block_info = []

        block_info = []
        block_size = 64
        import time
        a = time.time()
        for j in range(self.num_expert):
            for i in range(self.world_size):
                if all_expert_addr_len_list[i][j] >= block_size:
                    block_num = all_expert_addr_len_list[i][j] // block_size
                    if i != self.my_rank and i // 2 == self.my_rank // 2:
                        temp = [{"expert_id":expert_partition_plan[self.my_rank][j], "rank":i, "len": block_size, "start_addr": all_expert_addr_offset_list[i][j] + block_size*k} for k in range(block_num)]
                        remote_block_info.extend(temp)
                    else:
                        temp = [{"expert_id":expert_partition_plan[self.my_rank][j], "rank":i, "len": block_size, "start_addr": all_expert_addr_offset_list[i][j] + block_size*k} for k in range(block_num)]
                        local_block_info.extend(temp)
                    all_expert_addr_len_list[i][j] = all_expert_addr_len_list[i][j] % block_size
                    all_expert_addr_offset_list[i][j] = all_expert_addr_offset_list[i][j] + block_num * block_size
                    
        b = time.time()
             

        for i in range(self.num_expert):
            remain_tokens = [all_expert_addr_len_list[j][i] for j in range(self.world_size)]
            sorted_tokens = sorted(enumerate(remain_tokens), key=lambda x: x[1])
            for rank,expert_addr_len in sorted_tokens:
                if expert_addr_len > 0 :
                    temp = {"expert_id":expert_partition_plan[self.my_rank][i], "rank":rank, "len": expert_addr_len, "start_addr": all_expert_addr_offset_list[rank][i]}
                    remote_block_info.append(temp)

        c = time.time()


        total_block_size = len(local_block_info) + len(remote_block_info)

        block_info = [{} for i in range(total_block_size)]

        if len(local_block_info) > len(remote_block_info):
            for i in range(len(remote_block_info)):
                block_info[i*2 + 1] = remote_block_info[i]
                block_info[i*2] = local_block_info[i]
            block_info[len(remote_block_info)*2:] = local_block_info[len(remote_block_info):]
        else:
            for i in range(len(local_block_info)):
                block_info[i*2 + 1] = local_block_info[i]
                block_info[i*2] = remote_block_info[i]
            block_info[len(local_block_info)*2:] = remote_block_info[len(local_block_info):]

        d = time.time()


        expert_ids_list = [block_info[i]['expert_id'] for i in range(len(block_info))]
        local_expert_id_indexs = [ expert_partition_plan[self.my_rank].index(expert_id) for expert_id in expert_ids_list]
        ranks_list = [block_info[i]['rank'] for i in range(len(block_info))]
        lens_list = [block_info[i]['len'] for i in range(len(block_info))]
        statr_addrs_list = [block_info[i]['start_addr'] for i in range(len(block_info))]

        expert_block_ids = []
        expert_block_id_idxs = [0]
        tmp = local_expert_id_indexs
        for i in range(self.num_expert):
            indexes = [index for index, value in enumerate(tmp) if value == i]
            expert_block_ids.extend(indexes)
            expert_block_id_idxs.append(len(expert_block_ids))

        remote_block_num = len(remote_block_info)

        e1 = time.time()


        expert_ids_tensor = torch.tensor(expert_ids_list, dtype = torch.int32, device = device)
        e = time.time()
        
        
        expert_id_indexs_tensor = torch.tensor(local_expert_id_indexs, dtype = torch.int32, device = device)
        
        ranks_tensor = torch.tensor(ranks_list, dtype = torch.int32, device = device)
        lens_tensor = torch.tensor(lens_list, dtype = torch.int32, device = device)
        statr_addrs_tensor = torch.tensor(statr_addrs_list, dtype = torch.int32, device = device)
        expert_block_ids_tensor = torch.tensor(expert_block_ids, dtype = torch.int32, device = device)
        expert_block_id_idxs_tensor = torch.tensor(expert_block_id_idxs, dtype = torch.int32, device = device)

        f = time.time()

        # print("part a: {}, b: {}, c: {}, d: {}, e: {}".format((b - a)*1000 ,(c - b)*1000,(d - c)*1000,(e - e1)*1000, (f - e)*1000))
        


        return {'expert_ids': expert_ids_tensor,
                'local_expert_id_indexs': expert_id_indexs_tensor,
                'ranks': ranks_tensor,
                'lens': lens_tensor,
                'start_addrs': statr_addrs_tensor,
                'remote_block_num': remote_block_num,
                'total_block_num': total_block_size,
                'expert_block_ids': expert_block_ids_tensor,
                'expert_block_id_idxs': expert_block_id_idxs_tensor}


    def generate_kernel_paras(self, all_expert_count: torch.Tensor, expert_partition_plan, device):
        thread_block_num = 80
        
        world_num = self.world_size

        all_expert_addr_offset = all_expert_count.cumsum(-1)
        all_expert_addr_offset = torch.cat((torch.zeros([world_num,1], dtype=torch.int32),all_expert_addr_offset),-1)

        all_expert_addr_offset = all_expert_addr_offset.index_select(-1,torch.tensor(expert_partition_plan[self.my_rank]))
        all_expert_addr_len = all_expert_count.index_select(-1,torch.tensor(expert_partition_plan[self.my_rank]))

        block_infos = self.divide_into_block(expert_partition_plan, all_expert_addr_len, all_expert_addr_offset, device)

        return block_infos

    def compute_with_nccl(self, moe_inp, pos, all_expert_count, expert_partition_plan):

        def local_scatter(inp, pos):
            inp_buf = torch.index_select(inp, 0, pos)
            return inp_buf
        
        def scatter_func(inp, pos, local_expert_count, global_expert_count, fwd_batch_size, world_size):
            local_input_buf = _local_scatter(inp, pos)
        
        local_expert_count = all_expert_count[self.my_rank]
        global_expert_count = all_expert_count.index_select(-1,torch.tensor(expert_partition_plan[self.my_rank]))

        fwd_expert_count = global_expert_count.view(self.world_size, self.num_expert).sum(dim=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())

        expert_idxs = []
        for i in range(len(expert_partition_plan)):
            expert_idxs.extend(expert_partition_plan[i])

        expert_idxs_tensor = torch.tensor(expert_idxs, dtype=local_expert_count.dtype)

        local_input_buf = _local_scatter(moe_inp, torch.div(pos, self.top_k, rounding_mode='floor'),)
        global_input_buf = mix_moe_cuda.global_scatter(
            local_input_buf,
            local_expert_count,
            global_expert_count,
            expert_idxs_tensor,
            fwd_batch_size,
            self.world_size
        )



        with torch.no_grad():

            if isinstance(fwd_expert_count, torch.Tensor):
                fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
            outputs = []
            base_idx = 0

            activation = torch.nn.GELU()

            for i in range(self.num_expert):
                batch_size = fwd_expert_count_cpu[i]
                inp_slice = global_input_buf[base_idx : base_idx + batch_size]
                weight_0 = self.experts.htoh4.weight[i].transpose(-2, -1)
                weight_1 = self.experts.h4toh.weight[i].transpose(-2, -1)
                temp = torch.mm(inp_slice, weight_0)
                temp = temp + self.experts.htoh4.bias[i]
                temp = activation(temp)
                temp = torch.mm(temp, weight_1)
                temp = temp + self.experts.h4toh.bias[i]

                # temp = torch.zeros_like(temp) + self.experts.htoh4.bias[i]



                outputs.append(temp)
                base_idx += batch_size


            output = torch.cat(outputs, dim=0)

            # print(output.shape)

            # weight_0 = self.experts.htoh4.weight[0].transpose(-2, -1)
            # weight_1 = self.experts.h4toh.weight[0].transpose(-2, -1)

            # result = torch.mm(global_input_buf, weight_0)


            # result = activation(result)

            # result = torch.mm(result, weight_1)

            out_batch_size = tree.flatten(moe_inp)[0].shape[0] * self.top_k

            gloabl_result = mix_moe_cuda.global_gather(
                output,
                local_expert_count,
                global_expert_count,
                expert_idxs_tensor,
                out_batch_size,
                self.world_size)
            
            output = _local_gather(gloabl_result, pos, gloabl_result.shape[0],False)

            

        return output, global_input_buf, fwd_expert_count

        print(global_input_buf.shape)

        # print(expert_idxs_tensor.dtype)
        # # print(type(expert_idxs[0]))
        # # print(local_expert_count.dtype)

        # # print(expert_idxs)
        # # print(pos[0])
        # # local_input_buf = local_scatter(moe_inp, pos)

        # # if self.my_rank == 0:
        # print(all_expert_count)
        # print(expert_partition_plan)
        # print(local_expert_count)
        # print(global_expert_count)
        print(fwd_batch_size)
            
    def compute_weight_grad(self, input: torch.Tensor, grad_out :torch.Tensor, expert_block_ids, expert_id_idxs, block_num):

        expert_id_idxs_list = expert_id_idxs.tolist()
        input = input.reshape(block_num, -1, input.shape[-1])
        grad_out = grad_out.transpose(-2, -1)
        grad_out = grad_out.reshape(grad_out.shape[0], block_num, -1)
        print(input.shape, grad_out.shape)
        print(expert_block_ids)

        result = []
        for expert_id in range(self.num_expert):
            block_ids = expert_block_ids[expert_id_idxs[expert_id] : expert_id_idxs[expert_id + 1]]
            grad_out_data = grad_out.index_select(1, block_ids).reshape(grad_out.shape[0], -1)
            input_data = input.index_select(0, block_ids).reshape(-1, input.shape[-1])
            grad_weight_temp =torch.mm(grad_out_data, input_data)
            result.append(grad_weight_temp)

        output = torch.cat(result, dim=0).reshape(self.num_expert, grad_out.shape[0], input.shape[-1])
        return output

    def moe_forward(self, moe_inp, pos, moe_inp_block_infos):
        if self.experts is not None:
            moe_output = self.experts.forward(moe_inp, pos, moe_inp_block_infos)
            return moe_output

    def forward(self, moe_inp, pos, gate_score,moe_inp_block_infos):
        # moe_inp_batch_size = tree.flatten(
        #     tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        # )
        # assert all(
        #     [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        # ), "MoE inputs must have the same batch size"

        # import torch.cuda.nvtx as nvtx

        # nvtx.range_push("perpare")

        # if self.world_size > 1:
        #     def ensure_comm_func(tensor):
        #         ensure_comm(tensor, None)

        #     tree.map_structure(ensure_comm_func, moe_inp)

        # gate_top_k_idx, gate_score = self.gate(moe_inp)
        # device = moe_inp.device

        # import time
        # start = int(time.perf_counter() * 1000000)

        # with torch.no_grad():
        #     (
        #     pos,
        #     local_expert_count,
        #     global_expert_count,
        #     fwd_expert_count,
        #     all_expert_count,
        #     fwd_batch_size,
        #     ) =  prepare_forward(gate_top_k_idx, self.num_expert, self.world_size, self.my_rank)

        #     end1 = int(time.perf_counter() * 1000000)
        #     print("prepare_forward", (end1 - start)/1000)

        #     expert_partition_plan = self.balance_compute(all_expert_count)
        #     end2 = int(time.perf_counter() * 1000000)
        #     print("balance_compute", (end2 - end1)/1000)


        #     moe_inp_block_infos = self.generate_kernel_paras(all_expert_count, expert_partition_plan, device)

        # end = int(time.perf_counter() * 1000000)
        # print("generate_kernel_paras", (end - end2)/1000)
        # # print(moe_inp_block_infos)

        # nvtx.range_pop()
        moe_output = self.moe_forward(moe_inp, pos, moe_inp_block_infos)
        moe_output = moe_output.view(-1, self.top_k, self.d_model)
        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_output)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )

        # print(all_expert_count, expert_partition_plan)


        return moe_outp

        # if self.my_rank == 0:
        #     # print(all_expert_count) 
        #     print(local_expert_count)
        #     print(global_expert_count)

        


        temp_result, global_input_buf, fwd_expert_count = self.compute_with_nccl(moe_inp, pos, all_expert_count, expert_partition_plan)



        # pos = torch.div(pos, self.top_k, rounding_mode='floor')
        
        print(moe_inp_block_infos)

        # print(moe_inp.shape)
        # print(pos.shape)




        # moe_out_1 =  self.mix_moe.forward(moe_inp, pos, moe_inp_block_infos['local_expert_id_indexs'], moe_inp_block_infos['ranks'],
        #                     moe_inp_block_infos['lens'], moe_inp_block_infos['start_addrs'], moe_inp_block_infos['total_block_num'],
        #                     moe_inp_block_infos['remote_block_num'], temp_result)
        

        merged_input, ffn1_out =  self.mix_moe.mix_moe.first_gemm_forward(moe_inp, pos, moe_inp_block_infos['local_expert_id_indexs'], moe_inp_block_infos['ranks'],
                            moe_inp_block_infos['lens'], moe_inp_block_infos['start_addrs'], moe_inp_block_infos['total_block_num'],
                            moe_inp_block_infos['remote_block_num'], self.experts.htoh4.weight.transpose(-2, -1).contiguous(), self.experts.htoh4.bias.contiguous())
        


        
        # grad_out = torch.randn_like(ffn1_out).contiguous()
        # grad_in, grad_weight, grad_bias = self.mix_moe.mix_moe.first_gemm_backward(grad_out, 
        #                     self.experts.htoh4.weight, self.experts.htoh4.bias,
        #                     merged_input, moe_inp_block_infos['expert_block_ids'],
        #                     moe_inp_block_infos['expert_block_id_idxs'], pos, 
        #                     moe_inp_block_infos['local_expert_id_indexs'], 
        #                     moe_inp_block_infos['ranks'],
        #                     moe_inp_block_infos['lens'], 
        #                     moe_inp_block_infos['start_addrs'],
        #                     moe_inp_block_infos['total_block_num'],
        #                     moe_inp_block_infos['remote_block_num'])
        
        # grad_weight_local = self.compute_weight_grad(merged_input, grad_out, moe_inp_block_infos['expert_block_ids'],
        #                     moe_inp_block_infos['expert_block_id_idxs'], moe_inp_block_infos['total_block_num'],)
        
        # print("local grad weight", grad_weight_local.shape)
        
        # print("grad weight: ", grad_weight.shape)

        # print(torch.abs(grad_weight - grad_weight_local).max())
        # # print(grad_weight[0 : 128].mean(), merged_input[0 : 128].mean())
        # print(grad_weight.max(), grad_weight_local.max())
        # print(grad_weight.min(), grad_weight_local.min())
        # print(grad_weight.mean(), grad_weight_local.mean())

        activation = torch.nn.GELU()

        ffn1_out = activation(ffn1_out)

        print(pos.is_contiguous())


        ffn1_out, moe_out =   self.mix_moe.mix_moe.second_gemm_forward(ffn1_out, pos, moe_inp_block_infos['local_expert_id_indexs'], moe_inp_block_infos['ranks'],
                            moe_inp_block_infos['lens'], moe_inp_block_infos['start_addrs'], moe_inp_block_infos['total_block_num'],
                            moe_inp_block_infos['remote_block_num'], self.experts.h4toh.weight.transpose(-2, -1).contiguous(), self.experts.h4toh.bias.contiguous())
      
        torch.cuda.synchronize()

        # print(moe_out_1[0:10], moe_out[0:10])


        # print(torch.abs(moe_out_1 - moe_out).max())

       
        # print(moe_out.shape)
        
        # merged_input = self.recover_origin_data(merged_input, moe_inp_block_infos, expert_partition_plan)
        # print(torch.abs(merged_input - global_input_buf).max())
        # print(global_input_buf[0 : 128].mean(), merged_input[0 : 128].mean())
        # print(global_input_buf.max(), merged_input.max())
        # print(global_input_buf.min(), merged_input.min())
        # print(global_input_buf.mean(), merged_input.mean())

        # print(temp_result[0:10], moe_out[0:10])

        print(torch.abs(temp_result - moe_out).max())
        print(temp_result.sum(), moe_out.sum())
        print(temp_result.max(), moe_out.max())
        print(temp_result.min(), moe_out.min())
        print(temp_result.mean(), moe_out.mean())


        # moe_out = moe_out.reshape(moe_inp_block_infos['total_block_num'], -1 ,self.d_model*4)

        # print(moe_inp_block_infos)
        
        # print(moe_out[0].numel())

        # for i in range(moe_inp_block_infos['total_block_num']):
        #     if moe_out[i].numel() - moe_out[i].nonzero().size(0) >= 4096:
        #         print("my_rank: {}, block_id:{} is empty   {}".format(self.my_rank, i, moe_out[i].numel() - moe_out[i].nonzero().size(0)))





    def recover_origin_data(self, moe_out: torch.Tensor, moe_inp_block_infos, expert_partition_plan):

        print(expert_partition_plan)
        
        block_info = {}
        block_lens = {}
        for i in range(self.world_size):
            block_info[i] = {}
            block_lens[i] = {}

            for j in expert_partition_plan[self.my_rank]:
                block_info[i][j] = []
                block_lens[i][j] = []

        
        remote_block_num = moe_inp_block_infos['remote_block_num']
        total_block_num = moe_inp_block_infos['total_block_num']
        expert_ids = moe_inp_block_infos['expert_ids'].tolist()
        ranks = moe_inp_block_infos['ranks'].tolist()
        lens = moe_inp_block_infos['lens'].tolist()

        

        # moe_out = moe_out.reshape(total_block_num, -1 ,self.d_model*4)

        local_block_index_1 = [i*2 for i in range(remote_block_num)]
        local_block_index_2 = [remote_block_num*2 + i for i in range(total_block_num - remote_block_num * 2)]
        local_block_indexs = []
        local_block_indexs.extend(local_block_index_1)
        local_block_indexs.extend(local_block_index_2)
        remote_block_indexs = [i*2 + 1 for i in range(remote_block_num)]

        for local_block_index in local_block_indexs:
            block_info[ranks[local_block_index]][expert_ids[local_block_index]].append(local_block_index)
            block_lens[ranks[local_block_index]][expert_ids[local_block_index]].append(lens[local_block_index])

        for remote_block_index in remote_block_indexs:
            block_info[ranks[remote_block_index]][expert_ids[remote_block_index]].append(remote_block_index)
            block_lens[ranks[remote_block_index]][expert_ids[remote_block_index]].append(lens[remote_block_index])

        result_tensor = torch.empty(0, self.d_model, dtype=moe_out.dtype, layout=moe_out.layout, device=moe_out.device)

        for i in expert_partition_plan[self.my_rank]:
            for j in range(self.world_size):
                for k in range(len(block_info[j][i])):
                    result_tensor = torch.cat([result_tensor, moe_out[block_info[j][i][k]*64:(block_info[j][i][k]*64 + block_lens[j][i][k])]], 0)
                    # print(i, j, block_info[j][i][k], block_lens[j][i][k])


        return result_tensor

        print(result_tensor.shape)

        # print(block_info)
        # print(block_lens)
        # local_moe_out = moe_out.index_select(0, )
        # print(remote_block_num, total_block_num)
        

        
        
