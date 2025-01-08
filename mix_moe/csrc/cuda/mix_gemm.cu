
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "nvshmem.h"
#include "nvshmemx.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer))[0])


__device__ void myHGEMMAlignedV5(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    half * __restrict__ bias,
    half * smem,
    const int M, const int N, const int K, int I = -1) {

    const int BM = 64;
    const int BN = 512;
    const int BK = 32;

    int tid = threadIdx.x;
    int wid = tid >> 5;


    const int APAD = 8;
    const int BPAD = 8;



    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    half* temp_smem = smem;
    FLOAT(temp_smem[tid * 2]) = FLOAT(bias[tid * 2]);

    #pragma unroll
    for (int i = 1; i < 16; i++){
        FLOAT(temp_smem[i * (BN + BPAD) + tid * 2]) = FLOAT(temp_smem[tid * 2]);
    }

    __syncthreads();

    // if(blockIdx.x == 0 && tid == 0 && I == 0){
    //     printf("%d %d %d \n", blockIdx.x, tid, I);
    //     for(int i = 0; i < 64; i++)
    //         printf("%f ", __half2float(temp_smem[i]));
    //     printf("\n");
    // }
    // __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 4; j++){
            wmma::load_matrix_sync(frag_c[i][j], &temp_smem[wid * 64 + j * 16], BN + BPAD, wmma::mem_row_major);
        }
    }
    
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
    //     }
    // }

    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    int load_a_smem_m = (tid >> 2);
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 6) << 3;
    int load_b_smem_n = (tid & 63) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    // int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_4 = load_b_smem_addr_0 + 4 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_5 = load_b_smem_addr_0 + 5 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_6 = load_b_smem_addr_0 + 6 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_7 = load_b_smem_addr_0 + 7 * (BN + BPAD) * sizeof(half);
   

    int load_a_gmem_addr = OFFSET(load_a_smem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_smem_n, N);

    // int comp_c_frag_m = wid &  1;
    // int comp_c_frag_n = wid >> 1;

    int comp_c_frag_m = 0;
    int comp_c_frag_n = wid;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7), "l"(&b[load_b_gmem_addr + 7 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_4 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 4 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_5 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 5 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_6 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 6 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_7 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 7 * N]));


        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    int store_c_gmem_m = comp_c_frag_m * 64;
    int store_c_gmem_n = comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}


struct inputDataBlockInfo {
  int expert_id;
  int block_id;
  int rank_id;
  int len;
  int start_addr;
  int prop2;
  char *name;
};

// block_dim(256) 
__global__ void mix_gemm(half * __restrict__ moe_inp, half * __restrict__ temp_remote_mem, half * __restrict__ weight, half * __restrict__ bias, half * __restrict__ output,
    const int M, const int N, const int K){
    
    const int BM = 64;
    const int BN = 512;
    const int BK = 32;


    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;

    if (bx >= N / BN || by >= M / BM)
        return;

    extern __shared__ half smem[];

    half * compute_a = moe_inp + by * BM * K;
    half * compute_b = weight + bx * BN;
    half * compute_c = output + by * BM * N + bx * BN;

    myHGEMMAlignedV5(compute_a, compute_b, compute_c, bias, smem, M, N, K);
}


__global__ void moe_first_mm(half* __restrict__ moe_inp, half*  temp_remote_mem, half*  merged_input, half*  weight, half* bias,
    half* out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int local_block_num, int min_block_num, const int M, const int N, const int K){
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    auto block = cooperative_groups::this_thread_block();

    const int BM = 64;
    const int BN = 512;
    const int BK = 32;
    extern __shared__ half smem[];

    if(block_idx < min_block_num){
        int local_temp_offset = block_idx * 64;
        int remote_offset = start_addrs[block_idx*2+1];
        int remote_len = expert_lens[block_idx*2+1];
        int rank_id = ranks[block_idx*2+1];
        int remote_expert_id = expert_ids[block_idx*2+1];

        
        int remote_merged_index = (block_idx * 2 + 1) * 64; 
        if(rank_id != my_rank){
            nvshmemx_get16_nbi_block(merged_input + remote_merged_index * K, 
                moe_inp + remote_offset*K, 
                remote_len*K, 
                rank_id);
        }
        else{
            cooperative_groups::memcpy_async(block, 
            merged_input + remote_merged_index * K, 
            moe_inp + remote_offset * K, 
            sizeof(half) * remote_len * K);
        }

        int local_offset = start_addrs[block_idx*2];
        int local_len = expert_lens[block_idx*2];
        int local_expert_id = expert_ids[block_idx*2];

        int merged_index = (block_idx * 2) * 64; 
        cooperative_groups::memcpy_async(block, 
            merged_input + merged_index * K, 
            moe_inp + local_offset * K, 
            sizeof(half) * local_len * K);


        for(int i=0;i<N/BN;i++){
            half * compute_a = moe_inp + local_offset * K;
            half * compute_b = weight + i * BN + local_expert_id * K * N;
            half * compute_bias = bias + i * BN + local_expert_id * N;
            half * compute_c = out + block_idx * 2 * 64 * N + i * BN;
            myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K, i);
        }

        cooperative_groups::wait(block);

        nvshmem_quiet();

        for(int i=0;i<N/BN;i++){
            half * compute_a = merged_input + remote_merged_index * K;
            half * compute_b = weight + i * BN + remote_expert_id * K * N;
            half * compute_bias = bias + i * BN + remote_expert_id * N;
            half * compute_c = out +  (block_idx * 2 + 1) * 64  * N + i * BN;
            myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
        }
        return ;
    }
    
    if(local_block_num > remote_block_num){

        int local_offset = start_addrs[block_idx*2];
        int local_len = expert_lens[block_idx*2];
        int local_expert_id = expert_ids[block_idx*2];
        int merged_index = (block_idx * 2) * 64; 
        cooperative_groups::memcpy_async(block, 
            merged_input + merged_index * K, 
            moe_inp + local_offset * K, 
            sizeof(half) * local_len * K);

        for(int i=0;i<N/BN;i++){
            half * compute_a = moe_inp + local_offset * K;
            half * compute_b = weight + i * BN + local_expert_id * K * N;
            half * compute_bias = bias + i * BN + local_expert_id * N;
            half * compute_c = out + block_idx * 2 * 64 * N + i * BN;
            myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
        }

        if(block_idx * 2 + 1 < block_num)
        {
            int local_offset = start_addrs[block_idx*2+1];
            int local_len = expert_lens[block_idx*2+1];
            int local_expert_id = expert_ids[block_idx*2 + 1];

            int merged_index = (block_idx * 2 + 1) * 64; 
            cooperative_groups::memcpy_async(block, 
                merged_input + merged_index * K, 
                moe_inp + local_offset * K, 
                sizeof(half) * local_len * K);

            for(int i=0;i<N/BN;i++){
                half * compute_a = moe_inp + local_offset * K;
                half * compute_b = weight + i * BN + local_expert_id * K * N;
                half * compute_bias = bias + i * BN + local_expert_id * N;
                half * compute_c = out + (block_idx * 2 + 1) * 64 * N + i * BN;
                myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
            }
        }
    }
    else{
        for(int i =0 ;i< 2;i++){
            if(block_idx * 2 + i < block_num)
                break;
            int remote_offset = start_addrs[block_idx*2 + i];
            int remote_len = expert_lens[block_idx*2 + i];
            int rank_id = ranks[block_idx*2 + i];
            int remote_expert_id = expert_ids[block_idx*2 + i];


            int remote_merged_index = (block_idx * 2 + i) * 64; 
            if(rank_id != my_rank){
                nvshmemx_get16_nbi_block(merged_input + remote_merged_index * K, 
                    moe_inp + remote_offset*K, 
                    remote_len*K, 
                    rank_id);
            }
            else{
                cooperative_groups::memcpy_async(block, 
                merged_input + remote_merged_index * K, 
                moe_inp + remote_offset * K, 
                sizeof(half) * remote_len * K);
            }
        }


        cooperative_groups::wait(block);
        nvshmem_quiet();

        for(int j=0; j<2; j++)
        {
            if(block_idx * 2 + j < block_num)
                break;
            int remote_merged_index = (block_idx * 2 + j) * 64; 
            int remote_expert_id = expert_ids[block_idx*2 + j];

            for(int i=0;i<N/BN;i++){
                half * compute_a = merged_input + remote_merged_index * K;
                half * compute_b = weight + i * BN + remote_expert_id * K * N;
                half * compute_bias = bias + i * BN + remote_expert_id * N;
                half * compute_c = out +  (block_idx * 2 + 1) * 64  * N + i * BN;
                myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
            }
        }
    }


}


__global__ void moe_second_mm(half* __restrict__ moe_inp, half*  temp_remote_mem, half*  weight, half* bias,
    half* out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int local_block_num, int min_block_num, const int M, const int N, const int K){
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    auto block = cooperative_groups::this_thread_block();

    const int BM = 64;
    const int BN = 512;
    const int BK = 32;
    extern __shared__ half smem[];

    if(block_idx < min_block_num){
        int local_temp_offset = block_idx * 64;
        int remote_offset = start_addrs[block_idx*2+1];
        int remote_len = expert_lens[block_idx*2+1];
        int rank_id = ranks[block_idx*2+1];
        int expert_id = expert_ids[block_idx*2+1];

        for(int i=0;i<N/BN;i++){
            half * compute_a = moe_inp + (block_idx * 2 + 1) * 64 * K;
            half * compute_b = weight + i * BN + expert_id * K * N;
            half * compute_bias = bias + i * BN + expert_id * N;
            half * compute_c = temp_remote_mem +  local_temp_offset  * N + i * BN;
            myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
        }

        if(rank_id != my_rank){
            nvshmemx_put128_nbi_block(out + remote_offset*N, 
                temp_remote_mem + local_temp_offset * N, 
                remote_len * N / 8, 
                rank_id);
        }
        else{
            cooperative_groups::memcpy_async(block, 
            out + remote_offset * N, 
            temp_remote_mem + local_temp_offset * N, 
            sizeof(half) * remote_len * N);
        }


        int local_offset = start_addrs[block_idx*2];
        int local_len = expert_lens[block_idx*2];
        expert_id = expert_ids[block_idx*2];

        for(int i=0;i<N/BN;i++){
            half * compute_a = moe_inp + block_idx * 2 * 64 * K;
            half * compute_b = weight + i * BN + expert_id * K * N;
            half * compute_bias = bias + i * BN + expert_id * N;
            half * compute_c = out + local_offset * N + i * BN; 
            myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
        }

    }
    if(local_block_num > remote_block_num){

        int local_offset = start_addrs[block_idx*2];
        int local_len = expert_lens[block_idx*2];
        int expert_id = expert_ids[block_idx*2];

        for(int i=0;i<N/BN;i++){
            half * compute_a = moe_inp +  block_idx * 2 * 64 * K;
            half * compute_b = weight + i * BN + expert_id * K * N;
            half * compute_bias = bias + i * BN + expert_id * N;
            half * compute_c = out + local_offset * N + i * BN;
            myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
        }

        if(block_idx * 2 + 1 < block_num)
        {
            int local_offset = start_addrs[block_idx*2+1];
            int local_len = expert_lens[block_idx*2+1];
            int expert_id = expert_ids[block_idx*2+1];

            for(int i=0;i<N/BN;i++){
                half * compute_a = moe_inp + (block_idx * 2 + 1) * 64 * K;
                half * compute_b = weight + i * BN + expert_id * K * N;
                half * compute_bias = bias + i * BN + expert_id * N;
                half * compute_c = out + local_offset * N+ i * BN;
                myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
            }
        }
    }
    else{
        for(int i =0 ;i< 2;i++){
            if(block_idx * 2 + i < block_num)
                break;
            int local_temp_offset = (min_block_num + (block_idx - min_block_num - 1)*2 + i) * 64;
            int remote_offset = start_addrs[block_idx*2+i];
            int remote_len = expert_lens[block_idx*2+i];
            int rank_id = ranks[block_idx*2+i];
            int expert_id = expert_ids[block_idx*2+i];

            for(int j=0;j<N/BN;j++){
                half * compute_a = moe_inp + (block_idx * 2 + i) * 64 * K;
                half * compute_b = weight + j * BN + expert_id * K * N;
                half * compute_bias = bias + j * BN + expert_id * N;
                half * compute_c = temp_remote_mem +  local_temp_offset  * N + j * BN;
                myHGEMMAlignedV5(compute_a, compute_b, compute_c, compute_bias, smem, M, N, K);
            }

            if(rank_id != my_rank){
                nvshmemx_put128_nbi_block(out + remote_offset*N, 
                    temp_remote_mem + local_temp_offset * N, 
                    remote_len * N / 8, 
                    rank_id);
            }
            else{
                cooperative_groups::memcpy_async(block, 
                out + remote_offset * N, 
                temp_remote_mem + local_temp_offset * N, 
                sizeof(half) * remote_len * N);
            }
        }

    }
}

void first_mix_gemm(half* __restrict__ moe_inp, half* __restrict__ temp_remote_mem, half* __restrict__ merged_input, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream){
        // printf("%d %d %d \n", my_rank, block_num, remote_block_num);


        int repeat = 1;
        int warm_up = 0;
        
        cudaFuncSetAttribute(moe_first_mm,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        const int BM = 64;
        const int BN = 512;
        const int BK = 32;
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);


        int local_block_num = block_num - remote_block_num;
        int min_block_num = local_block_num > remote_block_num? remote_block_num: local_block_num;

        // for(int i=0;i<repeat;i++)
            // moe_first_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(moe_inp, temp_remote_mem, merged_input, weight, bias, out, 
            // expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,
            // M, N, K);
        cudaEvent_t warmup;
        cudaEventCreate(&warmup);
        for(int i=0;i<warm_up;i++)
            moe_first_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(moe_inp, temp_remote_mem, merged_input, weight, bias, out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num, local_block_num, min_block_num,
            M, N, K);
        cudaEventSynchronize(warmup);
        

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        for(int i=0;i<repeat;i++)
            moe_first_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(moe_inp, temp_remote_mem, merged_input, weight, bias, out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,local_block_num, min_block_num,
            M, N, K);

        cudaEventRecord(end);
        cudaEventSynchronize(end);
    
        float msec, sec;
        cudaEventElapsedTime(&msec, start, end);
        sec = msec / repeat;
        // if(my_rank == 0)
            // printf("AG, %f, %f, 0 \n", float(remote_block_num)/block_num, sec);

}


void first_mix_gemm_test(half* __restrict__ moe_inp, half* __restrict__ temp_remote_mem, half* __restrict__ merged_input, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream){
        // printf("%d %d %d \n", my_rank, block_num, remote_block_num);


        int repeat = 100;
        int warm_up = 20;
        
        cudaFuncSetAttribute(moe_first_mm,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        const int BM = 64;
        const int BN = 512;
        const int BK = 32;
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);


        int local_block_num = block_num - remote_block_num;
        int min_block_num = local_block_num > remote_block_num? remote_block_num: local_block_num;



        cudaEvent_t warmup;
        cudaEventCreate(&warmup);
        for(int i=0;i<warm_up;i++)
            moe_first_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(moe_inp, temp_remote_mem, merged_input, weight, bias, out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num, local_block_num, min_block_num,
            M, N, K);
        cudaEventSynchronize(warmup);
        

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        for(int i=0;i<repeat;i++)
            moe_first_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(moe_inp, temp_remote_mem, merged_input, weight, bias, out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,local_block_num, min_block_num,
            M, N, K);

        cudaEventRecord(end);
        cudaEventSynchronize(end);
    
        float msec, sec;
        cudaEventElapsedTime(&msec, start, end);
        sec = msec / repeat;
        if(my_rank == 0)
            printf("AG, %f, %f, 0 \n", float(remote_block_num)/block_num, sec);

}



void second_mix_gemm(half* __restrict__ hidden_output, half* __restrict__ temp_remote_mem, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ nvshmem_out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream){
        // printf("%d %d %d \n", my_rank, block_num, remote_block_num);


        int repeat = 1;
        int warm_up = 0;
        
        cudaFuncSetAttribute(moe_second_mm,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        const int BM = 64;
        const int BN = 512;
        const int BK = 32;
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
        int local_block_num = block_num - remote_block_num;
        int min_block_num = local_block_num > remote_block_num? remote_block_num: local_block_num;


        // printf("M:%d N:%d K:%d\n", M, N, K);

        // for(int i=0;i<repeat;i++)
            // moe_second_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(hidden_output, temp_remote_mem, weight, bias, nvshmem_out, 
            // expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,
            // M, N, K);
        cudaEvent_t warmup;
        cudaEventCreate(&warmup);
        for(int i=0;i<warm_up;i++)
            moe_second_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(hidden_output, temp_remote_mem, weight, bias, nvshmem_out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,local_block_num, min_block_num,
            M, N, K);

        cudaEventSynchronize(warmup);


        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        for(int i=0;i<repeat;i++)
            moe_second_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(hidden_output, temp_remote_mem, weight, bias, nvshmem_out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,local_block_num, min_block_num,
            M, N, K);

        cudaEventRecord(end);
        cudaEventSynchronize(end);
    
        float msec, sec;
        cudaEventElapsedTime(&msec, start, end);
        sec = msec /  repeat;
}


void second_mix_gemm_test(half* __restrict__ hidden_output, half* __restrict__ temp_remote_mem, half* __restrict__ weight, half* __restrict__ bias,
    half* __restrict__ nvshmem_out, int* expert_ids, int* ranks, int* expert_lens, int* start_addrs, int my_rank,
    int block_num, int remote_block_num, int M, int N, int K, cudaStream_t &stream){
        // printf("%d %d %d \n", my_rank, block_num, remote_block_num);


        int repeat = 100;
        int warm_up = 20;
        
        cudaFuncSetAttribute(moe_second_mm,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        const int BM = 64;
        const int BN = 512;
        const int BK = 32;
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
        int local_block_num = block_num - remote_block_num;
        int min_block_num = local_block_num > remote_block_num? remote_block_num: local_block_num;


        // printf("M:%d N:%d K:%d\n", M, N, K);

        // for(int i=0;i<repeat;i++)
            // moe_second_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(hidden_output, temp_remote_mem, weight, bias, nvshmem_out, 
            // expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,
            // M, N, K);
        cudaEvent_t warmup;
        cudaEventCreate(&warmup);
        for(int i=0;i<warm_up;i++)
            moe_second_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(hidden_output, temp_remote_mem, weight, bias, nvshmem_out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,local_block_num, min_block_num,
            M, N, K);

        cudaEventSynchronize(warmup);


        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        for(int i=0;i<repeat;i++)
            moe_second_mm<<<(block_num + 1)/2 , 256, dsmem, stream>>>(hidden_output, temp_remote_mem, weight, bias, nvshmem_out, 
            expert_ids, ranks, expert_lens, start_addrs, my_rank, block_num, remote_block_num,local_block_num, min_block_num,
            M, N, K);

        cudaEventRecord(end);
        cudaEventSynchronize(end);
    
        float msec, sec;
        cudaEventElapsedTime(&msec, start, end);
        sec = msec /  repeat;

        if(my_rank == 0)
            printf("GA, %f, %f, 0 \n", float(remote_block_num)/block_num, sec);
}

__global__ void myHGEMMAlignedV5WithBlock(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    int * expert_block_ids,
    int * expert_id_idxs,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int expert_id = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    int block_id_idx_start = expert_id_idxs[expert_id];
    int block_id_idx_end = expert_id_idxs[expert_id + 1];
    int block_num = block_id_idx_end - block_id_idx_start;
    int block_size = 64;
    int block_id = expert_block_ids[block_id_idx_start];

    if(block_num == 0)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], __float2half(0.0));
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);


    int block_id_start = expert_block_ids[block_id_idx_start];

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_start_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_a_gmem_addr = load_a_gmem_start_addr + block_id * block_size;
    int load_b_gmem_start_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);
    int load_b_gmem_addr = load_b_gmem_start_addr + block_id * block_size * N;

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 32
    for(int bk = 1; bk < block_num * (block_size / BK); bk++) {
        int block_id_idx_offset = bk / 2;
        int block_inside = bk % 2;

        block_id = expert_block_ids[block_id_idx_start + block_id_idx_offset];

        int load_a_gmem_addr = load_a_gmem_start_addr + block_id * block_size + block_inside * BK;
        int load_b_gmem_addr = load_b_gmem_start_addr + block_id * block_size * N + block_inside * BK * N;

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));

        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N) + expert_id * M * N;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }

}


void launch_compute_weight_grad(half * __restrict__ input, half * __restrict__ gradout, 
    half * __restrict__ weight_grad, int * expert_block_ids, int * expert_id_idxs, 
    int expert_num, int block_num, int M, int N){

    int repeat = 1;


    int block_size = 64;
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;
    int K = block_num * block_size;


    unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);

    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY, expert_num);
    dim3 blockDim(256);

    cudaFuncSetAttribute(myHGEMMAlignedV5WithBlock,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);


    // printf("block_GEMM: expert_num:%d M:%d N:%d K:%d\n", expert_num, M, N, K);
    // for(int i=0;i<repeat;i++)
        // myHGEMMAlignedV5WithBlock<<<gridDim, blockDim, dsmem>>>(input, gradout, weight_grad, expert_block_ids,
        //     expert_id_idxs, M, N, K);
    // cudaEvent_t start, end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    // cudaEventRecord(start);
    // for(int i=0;i<repeat;i++)
        myHGEMMAlignedV5WithBlock<<<gridDim, blockDim, dsmem>>>(input, gradout, weight_grad, expert_block_ids,
            expert_id_idxs, M, N, K);
    // cudaEventRecord(end);
    // cudaEventSynchronize(end);

    // float msec, sec;
    // cudaEventElapsedTime(&msec, start, end);
    // sec = msec / repeat;
    // printf("blcok gemm:  Time: %f \n", sec);

}