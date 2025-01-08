#pragma once
#include "../stream_manager.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef FMOE_USE_NCCL
#include <vector>
int find_index(std::vector<int> array, int len, int value);


torch::Tensor _expert_all_gather(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers);
torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers);
torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers);
torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers);

torch::Tensor _expert_rebalance(
    torch::Tensor weights,
    torch::Tensor bias,
    std::vector<int> repartition_plan,
    std::vector<int> repartition_plan_index,
    std::vector<int> local_expert_ids,
    int my_rank_id,
    int world_size);

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr);

void _cpy_data2tensor(pybind11::list source_data, torch::Tensor target_tensor);


template<typename scalar_t>
void mixmoe_cuda_expert_rebalance_impl(
    const scalar_t* weights,
    scalar_t* new_weights,
    long expert_numel,
    int n_expert,
    std::vector<int> repartition_plan,
    std::vector<int> repartition_plan_index,
    std::vector<int> local_expert_ids,
    int my_rank_id,
    int world_size,
    CudaStreamManager* smgr){

    NCCL_SAFE_CALL(ncclGroupStart());
    int recv_expert_idx = 0;
    for(int i = 0; i < world_size; i++){
        
        int send_expert_start = repartition_plan_index[i * world_size + my_rank_id];
        int send_expert_end = repartition_plan_index[i * world_size + my_rank_id + 1];
        for(int j = send_expert_start; j < send_expert_end; j++){
            int send_expert_id = repartition_plan[j];
            int send_expert_idx = find_index(local_expert_ids, n_expert, send_expert_id);
            NCCL_SAFE_CALL(ncclSend(
                weights + send_expert_idx * expert_numel,
                expert_numel * sizeof(scalar_t),
                ncclChar,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));

        }

        int recv_expert_start = repartition_plan_index[my_rank_id * world_size + i];
        int recv_expert_end = repartition_plan_index[my_rank_id * world_size + i + 1];

        for(int j = recv_expert_start; j < recv_expert_end; j++){
            int recv_expert_id = 0;
            NCCL_SAFE_CALL(ncclRecv(
                new_weights + recv_expert_idx * expert_numel,
                expert_numel * sizeof(scalar_t),
                ncclChar,
                i,
                smgr->ncclcomm,
                smgr->torchStream()));

            recv_expert_idx += 1;
            
        }     
    }

    NCCL_SAFE_CALL(ncclGroupEnd());

}


template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    size_t in_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    // assert world_size > 1
    int recv_ptr = 0;
    /* TODO: may save for backward */
    long*expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            long expert_idx = idx;
            if (local_expert_count[expert_idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[expert_idx] * in_feat,
                        local_expert_count[expert_idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_ptr * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
                recv_ptr += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
}

template<typename scalar_t>
void fmoe_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    size_t out_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    long send_ptr = 0;
    /* TODO: may save for backward */
    long *expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            long expert_idx = idx;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        output_buf + send_ptr * out_feat,
                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[expert_idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[expert_idx] * out_feat,
                        local_expert_count[expert_idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
}


#endif  // FMOE_USE_NCCL
