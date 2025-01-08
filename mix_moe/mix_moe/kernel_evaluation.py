import torch
import torch.distributed 
import mix_moe_cuda
import os

LOCAL_RANK = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
RANK = int(os.getenv('OMPI_COMM_WORLD_RANK'))
WORLD_SIZE = int(os.getenv('OMPI_COMM_WORLD_SIZE'))

backend = 'nccl'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

locality_perf = int(os.environ.get('LOCALITY', "0"))

torch.distributed.init_process_group(backend=backend, rank=RANK, world_size=WORLD_SIZE)
my_rank = RANK
world_size = WORLD_SIZE
device = torch.device('cuda:'+str(my_rank))
world_num = 4
num_expert = 16


total_batch_size = 8192*2
repaet = 100
top_k = 1
hidden_size = 1024

expert_partition_plan ={}
for rank_id in range(world_size):
    expert_partition_plan[rank_id] = [ i + rank_id*num_expert for i in range(num_expert)]



import fmoe_cuda
import torch.distributed as dist
def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")



def divide_into_block(expert_partition_plan, all_expert_addr_len, all_expert_addr_offset):
    all_expert_addr_len_list = all_expert_addr_len.tolist()
    all_expert_addr_offset_list = all_expert_addr_offset.tolist()
    local_block_info = []
    remote_block_info = []
    block_info = []
    block_size = 64
    global my_rank, world_size, world_num, num_expert, device
    for j in range(num_expert):
        for i in range(world_num):
            if all_expert_addr_len_list[i][j] >= block_size:
                block_num = all_expert_addr_len_list[i][j] // block_size
                if i != my_rank:
                    temp = [{"expert_id":expert_partition_plan[my_rank][j], "rank":i, "len": block_size, "start_addr": all_expert_addr_offset_list[i][j] + block_size*k} for k in range(block_num)]
                    remote_block_info.extend(temp)
                else:
                    temp = [{"expert_id":expert_partition_plan[my_rank][j], "rank":i, "len": block_size, "start_addr": all_expert_addr_offset_list[i][j] + block_size*k} for k in range(block_num)]
                    local_block_info.extend(temp)
                all_expert_addr_len_list[i][j] = all_expert_addr_len_list[i][j] % block_size
                all_expert_addr_offset_list[i][j] = all_expert_addr_offset_list[i][j] + block_num * block_size
    

                    
    for i in range(num_expert):
        remain_tokens = [all_expert_addr_len_list[j][i] for j in range(world_num)]
        sorted_tokens = sorted(enumerate(remain_tokens), key=lambda x: x[1])
        # ranks = []
        # start_addrs = []
        # lens = []
        # tmp_len = 0
        # for rank,expert_addr_len in sorted_tokens:
        #     if (tmp_len + expert_addr_len) >= block_size:
        #         temp = {"expert_id":expert_partition_plan[rank][i], "rank":ranks, "len": lens, "start_addr": start_addrs}
        #         remote_block_info.append(temp)
        #         ranks = []
        #         start_addrs = []
        #         lens = []
        #         tmp_len = 0
        #     if expert_addr_len > 0:
        #         ranks.append(rank)
        #         lens.append(expert_addr_len)
        #         start_addrs.append(all_expert_addr_offset_list[rank][i])
        #         tmp_len += expert_addr_len
        # temp = {"expert_id":expert_partition_plan[rank][i], "rank":ranks, "len": lens, "start_addr": start_addrs}
        # if len(ranks) > 0:
        #     remote_block_info.append(temp)
        for rank,expert_addr_len in sorted_tokens:
            if expert_addr_len > 0 :
                temp = {"expert_id":expert_partition_plan[my_rank][i], "rank":rank, "len": expert_addr_len, "start_addr": all_expert_addr_offset_list[rank][i]}
                remote_block_info.append(temp)
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
    expert_ids_list = [block_info[i]['expert_id'] for i in range(len(block_info))]
    local_expert_id_indexs = [ expert_partition_plan[my_rank].index(expert_id) for expert_id in expert_ids_list]
    ranks_list = [block_info[i]['rank'] for i in range(len(block_info))]
    lens_list = [block_info[i]['len'] for i in range(len(block_info))]
    statr_addrs_list = [block_info[i]['start_addr'] for i in range(len(block_info))]
    expert_block_ids = []
    expert_block_id_idxs = [0]
    tmp = local_expert_id_indexs
    for i in range(num_expert):
        indexes = [index for index, value in enumerate(tmp) if value == i]
        expert_block_ids.extend(indexes)
        expert_block_id_idxs.append(len(expert_block_ids))
    remote_block_num = len(remote_block_info)
    expert_ids_tensor = torch.tensor(expert_ids_list, dtype = torch.int32, device = device)
    expert_id_indexs_tensor = torch.tensor(local_expert_id_indexs, dtype = torch.int32, device = device)
    ranks_tensor = torch.tensor(ranks_list, dtype = torch.int32, device = device)
    lens_tensor = torch.tensor(lens_list, dtype = torch.int32, device = device)
    statr_addrs_tensor = torch.tensor(statr_addrs_list, dtype = torch.int32, device = device)
    expert_block_ids_tensor = torch.tensor(expert_block_ids, dtype = torch.int32, device = device)
    expert_block_id_idxs_tensor = torch.tensor(expert_block_id_idxs, dtype = torch.int32, device = device)
    return {'expert_ids': expert_ids_tensor,
            'local_expert_id_indexs': expert_id_indexs_tensor,
            'ranks': ranks_tensor,
            'lens': lens_tensor,
            'start_addrs': statr_addrs_tensor,
            'remote_block_num': remote_block_num,
            'total_block_num': total_block_size,
            'expert_block_ids': expert_block_ids_tensor,
            'expert_block_id_idxs': expert_block_id_idxs_tensor}



def generate_kernel_paras(all_expert_count):
    all_expert_addr_offset = all_expert_count.cumsum(-1)
    all_expert_addr_offset = torch.cat((torch.zeros([world_num,1], dtype=torch.int32),all_expert_addr_offset),-1)
    all_expert_addr_offset = all_expert_addr_offset.index_select(-1,torch.tensor(expert_partition_plan[my_rank]))
    all_expert_addr_len = all_expert_count.index_select(-1,torch.tensor(expert_partition_plan[my_rank]))

    block_infos = divide_into_block(expert_partition_plan, all_expert_addr_len, all_expert_addr_offset)

    return block_infos
    pass

import random
def generate_high_variance_array(N, M):
    # 初始化一个随机数组
    arr = [random.uniform(0, 1) for _ in range(N)]
    
    # 计算当前数组和
    current_sum = sum(arr)
    
    # 将数组中的每个元素按比例缩放，使得数组和为M
    arr = [x * M / current_sum for x in arr]
    
    # 将浮点数转换为整数，并调整总和为M
    arr = [int(x) for x in arr]
    diff = M - sum(arr)
    
    # 随机选择一些元素进行调整
    for _ in range(abs(diff)):
        idx = random.randint(0, N-1)
        arr[idx] += 1 if diff > 0 else -1
    return arr

_moe_group = get_torch_default_comm()



import mix_moe_cuda
def generate_data(batch_size, locality, hidden_size):
    local_data = int(batch_size * locality / 64) 
    remote_data = int(batch_size / 64 - local_data)
    local_expert_count = []
    
    array = generate_high_variance_array(num_expert*world_num - num_expert, remote_data)
    local_expert_count.extend(array)
    array = generate_high_variance_array(num_expert, local_data)
    local_expert_count[my_rank*num_expert:my_rank*num_expert] = array
    
    # print(local_expert_count)

    local_expert_count = [value*64 for value in local_expert_count]
    local_expert_count_tensor = torch.tensor(local_expert_count, dtype=torch.long).to(device)
    mix_moe_cuda.ensure_nccl(_moe_group, local_expert_count_tensor)

    all_expert_count = mix_moe_cuda.expert_all_gather(
                local_expert_count_tensor, num_expert, world_size
            )
    global_expert_count = mix_moe_cuda.expert_exchange(
                local_expert_count_tensor, num_expert, world_size
            )
    block_info = generate_kernel_paras(all_expert_count.cpu())

    # if my_rank == 0:
    #     print(local_expert_count)
    #     print(block_info['remote_block_num']/block_info['total_block_num'])

    return block_info, local_expert_count_tensor.cpu(), global_expert_count.cpu()


def expert_fn(inp, weights, bias, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0

        for i in range(num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(torch.mm(inp_slice, weights[i]) + bias[i])
            base_idx += batch_size
        return torch.cat(outputs, dim=0)



block_info, local_expert_count, global_expert_count= generate_data(total_batch_size, 0.9, hidden_size)


from linear import MixMOEInstance
import mix_moe_cuda
mix_moe_instance = MixMOEInstance.instance(total_batch_size, num_expert, top_k, hidden_size, hidden_size*4, 64,)

warm_up = 10

inp = torch.rand(total_batch_size, hidden_size).half().to(device).contiguous()
weight = torch.zeros([num_expert, hidden_size, hidden_size*4]).half().to(device).contiguous()
bias = torch.zeros([num_expert, hidden_size*4]).half().to(device).contiguous()
torch.nn.init.kaiming_uniform_(weight)

weight1 = torch.rand([num_expert, hidden_size*4, hidden_size]).half().to(device).contiguous()
torch.nn.init.kaiming_uniform_(weight1)
bias1 = torch.zeros([num_expert, hidden_size]).half().to(device).contiguous()

mix_moe_instance.mix_moe.AG_test(inp, 
                        block_info['local_expert_id_indexs'], 
                        block_info['ranks'],
                        block_info['lens'],
                        block_info['start_addrs'], 
                        block_info['total_block_num'],
                        block_info['remote_block_num'], weight, bias)

import time
time.sleep(4)

# for locality in [0.55,  0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
#     block_info, local_expert_count, global_expert_count = generate_data(total_batch_size, locality, hidden_size)
#     mix_moe_instance.mix_moe.AG_test(inp, 
#                         block_info['local_expert_id_indexs'], 
#                         block_info['ranks'],
#                         block_info['lens'],
#                         block_info['start_addrs'], 
#                         block_info['total_block_num'],
#                         block_info['remote_block_num'], weight, bias)


#     fmoe_cuda.ensure_nccl(_moe_group, inp)
#     fwd_expert_count = global_expert_count.view(world_size,
#                 num_expert).sum(dim=0).cpu().numpy()
#     for i in range(warm_up):
#         global_input_buf = fmoe_cuda.global_scatter(
#             inp,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )
#         expert_fn(inp, weight, bias, fwd_expert_count)


#     all2all_duration = 0
#     duration = 0
#     for i in range(repaet):
#         torch.cuda.synchronize()
#         start = time.time()
#         global_input_buf = fmoe_cuda.global_scatter(
#             inp,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )
#         torch.cuda.synchronize()
#         all2all_end = time.time()
#         all2all_duration += all2all_end - start
#         ffn1_out = expert_fn(global_input_buf, weight, bias, fwd_expert_count)
#         end = time.time()
#         duration += end - start
#         torch.cuda.synchronize()
#     if my_rank == 0:
#         print("All2All-GEMM, {:.2}, {:.2}, {:.2}".format(float(1 - locality), float(duration) *1000.0/repaet, float(all2all_duration) *1000.0/repaet))


#     ffn1_out = torch.rand(block_info['total_block_num']*64, hidden_size*4).half().to(device).contiguous()
#     mix_moe_instance.mix_moe.GA_test(ffn1_out, 
#                 block_info['local_expert_id_indexs'], 
#                 block_info['ranks'],
#                 block_info['lens'],
#                 block_info['start_addrs'], 
#                 block_info['total_block_num'],
#                 block_info['remote_block_num'], weight1, bias1)
    
#     ffn1_out = expert_fn(global_input_buf, weight, bias, fwd_expert_count)
#     for i in range(warm_up):
#         output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
#         local_output = fmoe_cuda.global_gather(
#             output,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )


#     import time
#     all2all_duration = 0
#     duration = 0
#     for i in range(repaet):
#         torch.cuda.synchronize()
#         start = time.time()
#         output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
#         torch.cuda.synchronize()
#         all2all_start = time.time()

#         local_output = fmoe_cuda.global_gather(
#             output,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )

#         end = time.time()
#         all2all_duration += end - all2all_start
#         duration += end - start
#         torch.cuda.synchronize()
#     if my_rank == 0:
#         print("GEMM-All2All, {:.2}, {:.2}, {:.2}".format(float(1 - locality), float(duration) *1000.0/repaet,float(all2all_duration) *1000.0/repaet))



if locality_perf == 1:
    for locality in [0.55,  0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
        block_info, local_expert_count, global_expert_count = generate_data(total_batch_size, locality, hidden_size)
        mix_moe_instance.mix_moe.AG_test(inp, 
                            block_info['local_expert_id_indexs'], 
                            block_info['ranks'],
                            block_info['lens'],
                            block_info['start_addrs'], 
                            block_info['total_block_num'],
                            block_info['remote_block_num'], weight, bias)


        fmoe_cuda.ensure_nccl(_moe_group, inp)
        fwd_expert_count = global_expert_count.view(world_size,
                    num_expert).sum(dim=0).cpu().numpy()
        for i in range(warm_up):
            global_input_buf = fmoe_cuda.global_scatter(
                inp,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )
            expert_fn(inp, weight, bias, fwd_expert_count)


        all2all_duration = 0
        duration = 0
        for i in range(repaet):
            torch.cuda.synchronize()
            start = time.time()
            global_input_buf = fmoe_cuda.global_scatter(
                inp,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )
            torch.cuda.synchronize()
            all2all_end = time.time()
            all2all_duration += all2all_end - start
            ffn1_out = expert_fn(global_input_buf, weight, bias, fwd_expert_count)
            end = time.time()
            duration += end - start
            torch.cuda.synchronize()
        if my_rank == 0:
            print("All2All-GEMM, {:.2}, {:.2}, {:.2}".format(float(1 - locality), float(duration) *1000.0/repaet, float(all2all_duration) *1000.0/repaet))




        ffn1_out = torch.rand(block_info['total_block_num']*64, hidden_size*4).half().to(device).contiguous()
        mix_moe_instance.mix_moe.GA_test(ffn1_out, 
                    block_info['local_expert_id_indexs'], 
                    block_info['ranks'],
                    block_info['lens'],
                    block_info['start_addrs'], 
                    block_info['total_block_num'],
                    block_info['remote_block_num'], weight1, bias1)

        ffn1_out = expert_fn(global_input_buf, weight, bias, fwd_expert_count)
        for i in range(warm_up):
            output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
            local_output = fmoe_cuda.global_gather(
                output,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )


        import time
        all2all_duration = 0
        duration = 0
        for i in range(repaet):
            torch.cuda.synchronize()
            start = time.time()
            output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
            torch.cuda.synchronize()
            all2all_start = time.time()

            local_output = fmoe_cuda.global_gather(
                output,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )

            end = time.time()
            all2all_duration += end - all2all_start
            duration += end - start
            torch.cuda.synchronize()
        if my_rank == 0:
            print("GEMM-All2All, {:.2}, {:.2}, {:.2}".format(float(1 - locality), float(duration) *1000.0/repaet,float(all2all_duration) *1000.0/repaet))


# for num_expert_temp in [2,4,8,16,32,64]:
#     num_expert = num_expert_temp
#     expert_partition_plan ={}
#     for rank_id in range(world_size):
#         expert_partition_plan[rank_id] = [ i + rank_id*num_expert for i in range(num_expert)]
#     locality = 0.75

#     inp = torch.rand(total_batch_size, hidden_size).half().to(device).contiguous()
#     weight = torch.zeros([num_expert, hidden_size, hidden_size*4]).half().to(device).contiguous()
#     bias = torch.zeros([num_expert, hidden_size*4]).half().to(device).contiguous()
#     torch.nn.init.kaiming_uniform_(weight)

#     weight1 = torch.rand([num_expert, hidden_size*4, hidden_size]).half().to(device).contiguous()
#     torch.nn.init.kaiming_uniform_(weight1)
#     bias1 = torch.zeros([num_expert, hidden_size]).half().to(device).contiguous()


#     block_info, local_expert_count, global_expert_count = generate_data(total_batch_size, locality, hidden_size)
#     mix_moe_instance.mix_moe.AG_test(inp, 
#                         block_info['local_expert_id_indexs'], 
#                         block_info['ranks'],
#                         block_info['lens'],
#                         block_info['start_addrs'], 
#                         block_info['total_block_num'],
#                         block_info['remote_block_num'], weight, bias)



#     mix_moe_cuda.ensure_nccl(_moe_group, inp)
#     fwd_expert_count = global_expert_count.view(world_size,
#                 num_expert).sum(dim=0).cpu().numpy()
#     for i in range(warm_up):
#         global_input_buf = mix_moe_cuda.global_scatter(
#             inp,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )
#         expert_fn(inp, weight, bias, fwd_expert_count)


#     all2all_duration = 0
#     duration = 0
#     for i in range(repaet):
#         torch.cuda.synchronize()
#         start = time.time()
#         global_input_buf = mix_moe_cuda.global_scatter(
#             inp,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )
#         torch.cuda.synchronize()
#         all2all_end = time.time()
#         all2all_duration += all2all_end - start
#         ffn1_out = expert_fn(global_input_buf, weight, bias, fwd_expert_count)
#         end = time.time()
#         duration += end - start
#         torch.cuda.synchronize()
#     if my_rank == 0:
#         print("All2All-GEMM, {:.2}, {:.2}, {:.2}".format(float(1 - locality), float(duration) *1000.0/repaet, float(all2all_duration) *1000.0/repaet))




#     ffn1_out = torch.rand(block_info['total_block_num']*64, hidden_size*4).half().to(device).contiguous()
#     mix_moe_instance.mix_moe.GA_test(ffn1_out, 
#                 block_info['local_expert_id_indexs'], 
#                 block_info['ranks'],
#                 block_info['lens'],
#                 block_info['start_addrs'], 
#                 block_info['total_block_num'],
#                 block_info['remote_block_num'], weight1, bias1)
    
#     ffn1_out = expert_fn(global_input_buf, weight, bias, fwd_expert_count)
#     for i in range(warm_up):
#         output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
#         local_output = mix_moe_cuda.global_gather(
#             output,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )


#     import time
#     all2all_duration = 0
#     duration = 0
#     for i in range(repaet):
#         torch.cuda.synchronize()
#         start = time.time()
#         output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
#         torch.cuda.synchronize()
#         all2all_start = time.time()
#         mix_moe_cuda.ensure_nccl(_moe_group, output)

#         local_output = mix_moe_cuda.global_gather(
#             output,
#             local_expert_count,
#             global_expert_count,
#             total_batch_size,
#             world_size
#         )

#         end = time.time()
#         all2all_duration += end - all2all_start
#         duration += end - start
#         torch.cuda.synchronize()
#     if my_rank == 0:
#         print("GEMM-All2All, {:.2}, {:.2}, {:.2}".format(float(1 - locality), float(duration) *1000.0/repaet,float(all2all_duration) *1000.0/repaet))


else:
    for num_expert_temp in [2,4,8,16,32,64]:
        num_expert = num_expert_temp
        expert_partition_plan ={}
        for rank_id in range(world_size):
            expert_partition_plan[rank_id] = [ i + rank_id*num_expert for i in range(num_expert)]
        locality = 0.75

        inp = torch.rand(total_batch_size, hidden_size).half().to(device).contiguous()
        weight = torch.rand([num_expert, hidden_size, hidden_size*4]).half().to(device).contiguous()
        bias = torch.zeros([num_expert, hidden_size*4]).half().to(device).contiguous()
        torch.nn.init.kaiming_uniform_(weight)

        weight1 = torch.rand([num_expert, hidden_size*4, hidden_size]).half().to(device).contiguous()
        torch.nn.init.kaiming_uniform_(weight1)
        bias1 = torch.zeros([num_expert, hidden_size]).half().to(device).contiguous()


        block_info, local_expert_count, global_expert_count = generate_data(total_batch_size, locality, hidden_size)
        mix_moe_instance.mix_moe.AG_test(inp, 
                            block_info['local_expert_id_indexs'], 
                            block_info['ranks'],
                            block_info['lens'],
                            block_info['start_addrs'], 
                            block_info['total_block_num'],
                            block_info['remote_block_num'], weight, bias)



        fmoe_cuda.ensure_nccl(_moe_group, inp)
        fwd_expert_count = global_expert_count.view(world_size,
                    num_expert).sum(dim=0).cpu().numpy()
        for i in range(warm_up):
            global_input_buf = fmoe_cuda.global_scatter(
                inp,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )
            expert_fn(inp, weight, bias, fwd_expert_count)


        all2all_duration = 0
        duration = 0
        for i in range(repaet):
            fmoe_cuda.ensure_nccl(_moe_group, inp)
            torch.cuda.synchronize()
            start = time.time()
            global_input_buf = fmoe_cuda.global_scatter(
                inp,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )
            torch.cuda.synchronize()
            all2all_end = time.time()
            all2all_duration += all2all_end - start
            ffn1_out = expert_fn(global_input_buf, weight, bias, fwd_expert_count)
            end = time.time()
            duration += end - start
            torch.cuda.synchronize()
        if my_rank == 0:
            print("All2All-GEMM, {:.2}, {:.2}, {:.2}".format(float(1 - locality), float(duration) *1000.0/repaet, float(all2all_duration) *1000.0/repaet))




        ffn1_out = torch.rand(block_info['total_block_num']*64, hidden_size*4).half().to(device).contiguous()
        mix_moe_instance.mix_moe.GA_test(ffn1_out, 
                    block_info['local_expert_id_indexs'], 
                    block_info['ranks'],
                    block_info['lens'],
                    block_info['start_addrs'], 
                    block_info['total_block_num'],
                    block_info['remote_block_num'], weight1, bias1)


        for i in range(warm_up):
            output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
            local_output = fmoe_cuda.global_gather(
                output,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )


        import time
        all2all_duration = 0
        duration = 0
        for i in range(repaet):
            torch.cuda.synchronize()
            start = time.time()
            output = expert_fn(ffn1_out, weight1, bias1, fwd_expert_count)
            mix_moe_cuda.ensure_nccl(_moe_group, output)

            torch.cuda.synchronize()
            all2all_start = time.time()

            local_output = mix_moe_cuda.global_gather(
                output,
                local_expert_count,
                global_expert_count,
                total_batch_size,
                world_size
            )

            end = time.time()
            all2all_duration += end - all2all_start
            duration += end - start
            torch.cuda.synchronize()
        if my_rank == 0:
            print("GEMM-All2All, {:.2}, {:.2}, {:.2}, {}".format(float(1 - locality), float(duration) *1000.0/repaet,float(all2all_duration) *1000.0/repaet, num_expert))


# ffn1_out = torch.rand(block_info['total_block_num']*64, hidden_size*4).half().to(device).contiguous()
# weight = torch.rand([hidden_size*4, hidden_size]).half().to(device).contiguous()
# torch.nn.init.kaiming_uniform_(weight)
# bias = torch.zeros([hidden_size]).half().to(device).contiguous()



# for locality in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
#     block_info = generate_data(total_batch_size, locality, hidden_size)
#     mix_moe_instance.mix_moe.GA_test(ffn1_out, 
#                         block_info['local_expert_id_indexs'], 
#                         block_info['ranks'],
#                         block_info['lens'],
#                         block_info['start_addrs'], 
#                         block_info['total_block_num'],
#                         block_info['remote_block_num'], weight, bias)




# import time
# start = time.time()

# for i in range(repaet):
#     torch.cuda.synchronize()

#     output = torch.mm(inp, weight)
#     output += bias
#     torch.cuda.synchronize()

# end = time.time()
# print("AG time cost: ", float(end - start) *1000.0/repaet, "ms")


# ffn1_out = torch.rand(block_info['total_block_num']*64, hidden_size*4).half().to(device).contiguous()
# weight = torch.rand([hidden_size*4, hidden_size]).half().to(device).contiguous()
# torch.nn.init.kaiming_uniform_(weight)
# bias = torch.zeros([hidden_size]).half().to(device).contiguous()



# for locality in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
#     block_info = generate_data(total_batch_size, locality, hidden_size)
#     mix_moe_instance.mix_moe.GA_test(ffn1_out, 
#                         block_info['local_expert_id_indexs'], 
#                         block_info['ranks'],
#                         block_info['lens'],
#                         block_info['start_addrs'], 
#                         block_info['total_block_num'],
#                         block_info['remote_block_num'], weight, bias)


# for i in range(warm_up):
#     output = torch.mm(ffn1_out, weight)


# import time
# start = time.time()

# for i in range(repaet):
#     torch.cuda.synchronize()

#     output = torch.mm(ffn1_out, weight)
#     output += bias
#     torch.cuda.synchronize()

# end = time.time()
# print("GA time cost: ", float(end - start) *1000.0/repaet, "ms")





# all_expert_count = torch.load('/workspace/mix_moe/mix_moe/all_expert_count.pt')
# print(all_expert_count)