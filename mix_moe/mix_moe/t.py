import torch

world_size = 4
num_expert = 16
expert_partition = [[ i * num_expert + j for j in range(num_expert)] for i in range(world_size)]

def balance_compute(all_expert_count: torch.Tensor):


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
                    for expert_id in expert_partition[world_id]:
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
        
        def temp(gpu_topo, all_expert_count, expert_ids):
            if len(gpu_topo) == 1:
                return [{gpu_topo[0] : expert_ids}]
            import operator
            from functools import reduce
            import numpy as np
            partition_groups = []
            for i in range(len(gpu_topo)):
                tmp = list(np.array(gpu_topo[i]).flatten())
                partition_groups.append(tmp)

            sorted_expert_count, sorted_expert_idx = perpare_expert_count(all_expert_count, partition_groups, expert_ids)
            tmp_partition_plan = partition_expert(partition_groups, sorted_expert_count, sorted_expert_idx, expert_ids, len(expert_ids), num_expert * len(partition_groups[0]))

            result = []
            for i in range(len(tmp_partition_plan)):
                tmp = temp(gpu_topo[i], all_expert_count, tmp_partition_plan[i])
                result.extend(tmp)
                

            return result

                

        

        gpu_topo = [[[0],[1]], [[2],[3]]]

        result = temp(gpu_topo, all_expert_count, list(range(num_expert * world_size)))
        result_1 = {}
        for it in result:
            result_1.update(it)


        expert_repartition = []
        for i in range(world_size):
            tmp = [ [] for i in range(world_size)]
            for expert_id in result_1[i]:
                for j in range(world_size):
                    if expert_id in expert_partition[j]:
                        tmp[j].append(expert_id)
                        break
            expert_repartition.append(tmp)
        print(expert_repartition)



        # sorted_expert_count, sorted_expert_idx = all_expert_count.flatten().sort(descending=True)

        # sorted_expert_count = sorted_expert_count.tolist()
        # sorted_expert_idx = sorted_expert_idx.tolist()

        # total_expert_num = world_size * num_expert


        # world_num = world_size


        # partition_plan = {}
        # for i in range(world_num):
        #     partition_plan[i] = []
        # for i in range(len(sorted_expert_count)):
        #     expert_id = sorted_expert_idx[i] % total_expert_num
        #     if not check_expert_is_planned(expert_id, partition_plan):
        #         world_id = sorted_expert_idx[i] // total_expert_num
        #         if len(partition_plan[world_id]) < num_expert:
        #             partition_plan[world_id].append(expert_id)
        
        return result_1

all_expert_count = torch.load('/workspace/mix_moe/mix_moe/all_expert_count.pt')
data = [736, 14, 53, 20, 68, 1455, 879, 74, 54, 337, 15, 22, 72, 42, 9, 1034, 94, 14, 223, 115, 29, 1085, 905, 1506, 41, 75, 284, 67, 916, 80, 3, 111, 6, 202, 62, 47, 62, 38, 67, 63, 154, 164, 3, 307, 6, 159, 9, 2407, 821, 7, 117, 95, 6, 2, 53, 20, 599, 2, 45, 0, 77, 108, 237, 7,
746, 16, 42, 10, 69, 1469, 909, 45, 118, 326, 12, 39, 51, 25, 4, 1073, 95, 22, 216, 121, 88, 977, 994, 1549, 29, 117, 245, 67, 846, 81, 2, 73, 11, 169, 93, 28, 72, 39, 77, 58, 78, 127, 8, 309, 7, 125, 6, 2364, 898, 8, 96, 154, 37, 2, 51, 14, 560, 2, 26, 0, 73, 149, 264, 3,
701, 18, 67, 14, 53, 1260, 810, 28, 58, 274, 1, 25, 111, 27, 5, 930, 122, 17, 225, 140, 29, 1242, 951, 1539, 68, 63, 277, 78, 931, 93, 1, 153, 11, 141, 43, 45, 34, 52, 75, 77, 503, 120, 1, 314, 7, 132, 7, 2298, 632, 9, 57, 60, 8, 2, 70, 19, 672, 0, 233, 2, 119, 92, 235, 3,
716, 17, 85, 25, 66, 1192, 966, 21, 57, 273, 2, 42, 117, 37, 4, 1054, 88, 18, 280, 129, 27, 1023, 1031, 1456, 68, 65, 250, 82, 844, 77, 2, 193, 6, 194, 65, 23, 36, 55, 99, 73, 146, 140, 4, 350, 5, 141, 3, 2473, 606, 6, 77, 57, 9, 1, 64, 21, 729, 0, 201, 1, 91, 123, 278, 0]
data_tensor = torch.tensor(data).reshape(4, 64)
balance_compute(data_tensor)


# [[[9, 4, 14, 5, 0, 1, 2, 3, 6, 7, 8], [22, 21], [34], [50, 49]], [[12], [16, 17, 18, 19, 20, 23, 24, 25, 26], [46, 43, 41], [58, 59, 51]], [[15, 11, 6], [30], [36, 44, 32, 34, 35, 38, 39, 40, 41, 42], [54, 53]], [[10, 2], [27, 24, 17], [33, 37], [48, 61, 60, 49, 50, 51, 52, 55, 56]]]


