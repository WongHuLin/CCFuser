#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mix_moe_kernel.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


__global__ void _local_scatter_kernel(half* __restrict__ input, half* __restrict__ output, int hidden_size, int* pos){
    int bidx = blockIdx.x;
    int tid = threadIdx.x;
    auto block = cooperative_groups::this_thread_block();

    #pragma unroll
    for(int i=0;i<32;i++){
        int temp = bidx*32 + i;
        int source_data_offset = pos[temp];
        cooperative_groups::memcpy_async(block, output + temp * hidden_size, input + source_data_offset * hidden_size, sizeof(half) * hidden_size);

    }

    cooperative_groups::wait(block);

}

void _local_scatter_launch(half* input, half* output, int hidden_size, int* pos, int pos_len){
    int grid_num = pos_len / 32;
    dim3 gridDim(grid_num);
    dim3 blockDim(256);

    _local_scatter_kernel<<<gridDim, blockDim>>>(input, output, hidden_size, pos);

}

__global__ void _local_gather_kernel_add(half* input, half* output, int hidden_size, int* pos){
    int bidx = blockIdx.x;
    int tid = threadIdx.x;
    int bidx_dim = blockDim.x;

    auto block = cooperative_groups::this_thread_block();

    // half* data_smem[2][hidden_size];

}


__global__ void _local_gather_kernel(half* __restrict__ input, half* __restrict__ output, int hidden_size, int* pos){
    int bidx = blockIdx.x;
    int tid = threadIdx.x;
    auto block = cooperative_groups::this_thread_block();

    #pragma unroll
    for(int i=0;i<32;i++){
        int temp = bidx*32 + i;
        int target_data_offset = pos[temp];
        cooperative_groups::memcpy_async(block, output + target_data_offset * hidden_size, input + temp * hidden_size, sizeof(half) * hidden_size);

    }

    cooperative_groups::wait(block);
}

void _local_gather_launch(half* input, half* output, int hidden_size, int* pos, int pos_len){
    int grid_num = pos_len / 32;
    dim3 gridDim(grid_num);
    dim3 blockDim(256);

    _local_gather_kernel<<<gridDim, blockDim>>>(input, output, hidden_size, pos);

}