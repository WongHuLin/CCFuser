#include <torch/extension.h>
#include "global_exchange.h"
#include "utils.h"

#ifdef FMOE_USE_NCCL
#include <nccl.h>

int find_index(std::vector<int> array, int len, int value){
    for(int i = 0; i < len; i++){
        if(array[i] == value)
            return i;
    }
}


void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
        NCCL_SAFE_CALL(ncclRecv(
                global_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
}

void _cpy_data2tensor(pybind11::list source_data, torch::Tensor target_tensor){

    pybind11::array_t<int> np_array = pybind11::cast<pybind11::array_t<int>>(source_data);
    int* source_data_ptr = np_array.mutable_data();
    int source_data_len = np_array.size();
    int* target_tensor_ptr = target_tensor.data_ptr<int>();
    cudaMemcpy(target_tensor_ptr, source_data_ptr, source_data_len*sizeof(int), cudaMemcpyHostToDevice);
}


torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers) {
    auto global_expert_count = torch::empty_like(local_expert_count);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_exchange_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr);
    return global_expert_count;
}

torch::Tensor _expert_rebalance(
    torch::Tensor weights,
    torch::Tensor bias,
    std::vector<int> repartition_plan,
    std::vector<int> repartition_plan_index,
    std::vector<int> local_expert_ids,
    int my_rank_id,
    int world_size){
    
    auto n_expert = weights.size(0);
    long expert_numel = weights.numel() / n_expert;
    auto new_weights = torch::empty_like(weights);
    auto smgr = getCudaStreamManager(weights.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            weights.scalar_type(), "mixmoe_cuda_expert_rebalance", ([&] {
        mixmoe_cuda_expert_rebalance_impl<scalar_t>(
            weights.data_ptr<scalar_t>(),
            new_weights.data_ptr<scalar_t>(),
            expert_numel,
            n_expert,
            repartition_plan,
            repartition_plan_index,
            local_expert_ids,
            my_rank_id,
            world_size,
            smgr
        );
    }));


    // NCCL_SAFE_CALL(ncclGroupStart());
    // long recv_expert_idx = 0;
    // for(int i = 0; i < world_size; i++){
        
    //     long send_expert_start = repartition_plan_index[i * world_size + my_rank_id];
    //     long send_expert_end = repartition_plan_index[i * world_size + my_rank_id + 1];
    //     for(int j = send_expert_start; j < send_expert_end; j++){
    //         long send_expert_id = repartition_plan[j];
    //         int send_expert_idx = find_index(local_expert_ids, n_expert, send_expert_id);
    //         NCCL_SAFE_CALL(ncclSend(
    //             weights_ptr + send_expert_idx * expert_numel,
    //             expert_numel * sizeof(half),
    //             ncclChar,
    //             i,
    //             smgr->ncclcomm,
    //             smgr->torchStream()));

    //     }

    //     long recv_expert_start = repartition_plan_index[my_rank_id * world_size + i];
    //     long recv_expert_end = repartition_plan_index[my_rank_id * world_size + i + 1];

    //     for(int j = recv_expert_start; j < recv_expert_end; j++){
    //         int recv_expert_id = 0;
    //         NCCL_SAFE_CALL(ncclRecv(
    //             new_weights_ptr + recv_expert_idx * expert_numel,
    //             expert_numel * sizeof(half),
    //             ncclChar,
    //             i,
    //             smgr->ncclcomm,
    //             smgr->torchStream()));

    //         recv_expert_idx += 1;
            
    //     }     
    // }

    // NCCL_SAFE_CALL(ncclGroupEnd());

    return new_weights;
}


void fmoe_cuda_expert_all_gather_impl(
        const long* local_expert_count,
        long* all_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count,
                n_expert * world_size,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
        NCCL_SAFE_CALL(ncclRecv(
                all_expert_count + world_size * n_expert * i,
                n_expert * world_size,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
}

torch::Tensor _expert_all_gather(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers) {
    auto options = torch::TensorOptions().dtype(torch::kLong).device(local_expert_count.device());
    auto all_expert_count = torch::empty({n_workers, n_workers*n_expert}, options);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_all_gather_impl(
            local_expert_count.data_ptr<long>(),
            all_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr);
    return all_expert_count;
}

torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            input_buf.scalar_type(), "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            smgr
        );
    }));
    return global_input_buf;
}

torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(output_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            output_buf.scalar_type(), "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_output_buf.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr
        );
    }));
    return local_output_buf;
}

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13))
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#else
#include <c10d/ProcessGroupNCCL.hpp>
#endif

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        broadcastUniqueNCCLID(&ncclID,
                false,
                "fastmoe_nccl_comm",
                rank);
#elif defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        return comm;
    }
};

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
void _ensure_nccl(c10d::ProcessGroup& p, torch::Tensor t) {
#else
void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
#endif  // TORCH_VERSION
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood) {
        return;
    }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)
        (p.getBackend(c10d::ProcessGroup::NCCL).get());
#else
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
#endif  // TORCH_VERSION
    smgr->ncclcomm = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}

#endif  // FMOE_USE_NCCL
