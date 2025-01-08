#pragma once
#include <cuda_fp16.h>

void first_mix_gemm(half* __restrict__ moe_inp, half* __restrict__ temp_remote_mem, half* __restrict__ merged_input, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream);


void first_mix_gemm_test(half* __restrict__ moe_inp, half* __restrict__ temp_remote_mem, half* __restrict__ merged_input, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream);

void second_mix_gemm(half* __restrict__ hidden_output, half* __restrict__ temp_remote_mem, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ nvshmem_out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream);


void second_mix_gemm_test(half* __restrict__ hidden_output, half* __restrict__ temp_remote_mem, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ nvshmem_out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream);

void launch_compute_weight_grad(half * __restrict__ input, half * __restrict__ gradout, 
    half * __restrict__ weight_grad, int * expert_block_ids, int * expert_id_idxs, 
    int expert_num, int block_num, int M, int N);

// void test_gemm(half* a, half* b, half* c, int M, int N, int K);