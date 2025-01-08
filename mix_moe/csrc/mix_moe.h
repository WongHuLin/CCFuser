#pragma once
#include <torch/torch.h>

#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <cuda_fp16.h>
#include "mix_moe.h"
#include "cuda/mix_moe_kernel.cuh"
#include "cuda/mix_gemm.cuh"

#include <chrono>
using namespace std;
using namespace chrono;

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)




class _MixMOE{
    public:
        int _rank;
        int _nranks;
        int _batch_size;
        int _seq_len;
        half* _moe_data_nvshmem;
        torch::Tensor _expert_weight_ffn1;
        torch::Tensor _expert_weight_ffn2;
        torch::Tensor _expert_bias_ffn1;
        torch::Tensor _expert_bias_ffn2;

        int _local_expert_num;
        int _hidden_size;
        int _ffn_hidden_size;
        int _top_k;
        int _block_size;

        int _mype;
        int _npes;

        nvshmemx_init_attr_t _attr;

        cudaStream_t nvshmem_stream;


        void init_nvshemem(){
            
            MPI_Init(NULL, NULL);
            MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &this->_nranks);
            MPI_Comm mpi_comm = MPI_COMM_WORLD;

            printf("%d %d init success\n", this->_rank,  this->_nranks);

            int device_count = torch::cuda::device_count();
            CUDA_CHECK(cudaSetDevice(this->_rank % device_count));

            int mype, npes;
            this->_attr.mpi_comm = &mpi_comm;
            nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &this->_attr);

            this->_mype = nvshmem_my_pe();
            this->_npes = nvshmem_n_pes();

            int total_token_num = this->_batch_size * this->_seq_len;

            this->_moe_data_nvshmem = (half*)nvshmem_malloc(sizeof(half)*2*total_token_num*this->_hidden_size);
            
        }

        void compare_result(float* data1, float* data2, size_t len){
            float max_error = 0;
            for(size_t i=0;i<len;i++){
                float tmp = abs(data1[i] - data2[i]);
                max_error = max(max_error, tmp);
                if(abs(data1[i] - data2[i]) < 0.0005)
                    continue;
                std::cout<<i <<" " <<data1[i]<< " "<< data2[i]<< " " <<"result error"<<std::endl;
                return;
            }
            std::cout<<"max error: "<< max_error<<std::endl;
        }


        void finalize_nvshmem(){
            nvshmem_free(this->_moe_data_nvshmem);
            nvshmem_finalize();
            MPI_Finalize();
        }


        _MixMOE(int batch_size, int seq_len, int local_expert_num, int top_k,
                int hidden_size, int ffn_hidden_size, int block_size, torch::Tensor expert_weight_ffn1,
                torch::Tensor expert_weight_ffn2, torch::Tensor expert_bias_ffn1, torch::Tensor expert_bias_ffn2):
                _batch_size(batch_size), 
                _seq_len(seq_len), 
                _local_expert_num(local_expert_num),
                _hidden_size(hidden_size), 
                _top_k(top_k),
                _block_size(block_size),
                _ffn_hidden_size(ffn_hidden_size),
                _expert_weight_ffn1(std::move(expert_weight_ffn1)),
                _expert_weight_ffn2(std::move(expert_weight_ffn2)),
                _expert_bias_ffn1(std::move(expert_bias_ffn1)),
                _expert_bias_ffn2(std::move(expert_bias_ffn2)){

            this->init_nvshemem();
            cudaStreamCreate(&this->nvshmem_stream);

        }

        _MixMOE(int batch_size, int seq_len, int local_expert_num, int top_k,
                int hidden_size, int ffn_hidden_size, int block_size):
                _batch_size(batch_size), 
                _seq_len(seq_len), 
                _local_expert_num(local_expert_num),
                _hidden_size(hidden_size), 
                _top_k(top_k),
                _block_size(block_size),
                _ffn_hidden_size(ffn_hidden_size){

            this->init_nvshemem();
            cudaStreamCreate(&this->nvshmem_stream);

        }

        void local_scatter(half* input_data, half* output_data, int* pos, int pos_len){
            _local_scatter_launch(input_data, this->_moe_data_nvshmem, this->_hidden_size,
                pos, pos_len);
        }

        void local_gather(half* inpute_data, half* output_data, int* pos, int pos_len){
            _local_gather_launch(inpute_data, output_data, this->_hidden_size, pos, pos_len);
        }

        void AG_test(torch::Tensor source_data,  torch::Tensor expert_ids, 
                    torch::Tensor ranks, torch::Tensor expert_lens, torch::Tensor start_addrs, 
                    int total_block_num, int remote_block_num, torch::Tensor weight, torch::Tensor bias){
            half* source_data_ptr = reinterpret_cast<half*>(source_data.data_ptr());
            cudaMemcpy(source_data_ptr, this->_moe_data_nvshmem, source_data.numel()*sizeof(half), cudaMemcpyDeviceToDevice);

            auto opt = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(source_data.device());

            auto temp_remote_mem = torch::zeros({remote_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            auto ffn1_out = torch::zeros({total_block_num*this->_block_size, this->_ffn_hidden_size}, opt).contiguous();
            auto merged_input = torch::zeros({total_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();

            int M = total_block_num * this->_block_size;
            int N = this->_ffn_hidden_size;
            int K = this->_hidden_size;


            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);
            auto start = system_clock::now();
            
            first_mix_gemm_test(this->_moe_data_nvshmem, 
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(merged_input.data_ptr()),
                    reinterpret_cast<half*>(weight.data_ptr()),
                    reinterpret_cast<half*>(bias.data_ptr()),
                    reinterpret_cast<half*>(ffn1_out.data_ptr()), 
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );
            
            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            // std::cout <<"AG kernel time: " << duration.count() / 1000.0 << " microseconds" << std::endl;

        }

        void GA_test(torch::Tensor ffn1_out,  torch::Tensor expert_ids, 
                    torch::Tensor ranks, torch::Tensor expert_lens, torch::Tensor start_addrs, 
                    int total_block_num, int remote_block_num, torch::Tensor weight, torch::Tensor bias){
            // half* source_data_ptr = reinterpret_cast<half*>(source_data.data_ptr());
            // cudaMemcpy(source_data_ptr, this->_moe_data_nvshmem, source_data.numel()*sizeof(half), cudaMemcpyDeviceToDevice);

            auto opt = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(ffn1_out.device());

            auto temp_remote_mem = torch::zeros({remote_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            int M = total_block_num*this->_block_size;
            int N = this->_hidden_size;
            int K = this->_ffn_hidden_size;



            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);
            auto start = system_clock::now();
            
            second_mix_gemm_test(reinterpret_cast<half*>(ffn1_out.data_ptr()), 
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(weight.data_ptr()),
                    reinterpret_cast<half*>(bias.data_ptr()),
                    this->_moe_data_nvshmem, 
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );
            
            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);

        }

        void read_data_from_nvshmem(int batch_size, int hidden_size){
            return ;
        }

        std::vector<torch::Tensor> first_gemm_forward(torch::Tensor inp, torch::Tensor pos, torch::Tensor expert_ids, 
                    torch::Tensor ranks, torch::Tensor expert_lens, torch::Tensor start_addrs, 
                    int total_block_num, int remote_block_num, torch::Tensor weight, torch::Tensor bias){
            if(pos.dtype() != torch::kInt32){
                pos = pos.to(torch::kInt32);
            }
            if(inp.dtype() != torch::kFloat16){
                printf("moe input data type must be float 16\n");
                return {torch::empty({0})};
            }

            auto start = system_clock::now();
            

            auto temp_pos = torch::div(pos, this->_top_k, "floor");

            local_scatter(reinterpret_cast<half*>(inp.data_ptr()), this->_moe_data_nvshmem, 
                reinterpret_cast<int*>(temp_pos.data_ptr()), pos.numel());


            auto opt = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(inp.device());
            auto temp_remote_mem = torch::zeros({remote_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            auto ffn1_out = torch::zeros({total_block_num*this->_block_size, this->_ffn_hidden_size}, opt).contiguous();
            auto merged_input = torch::zeros({total_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();


            int M = total_block_num * this->_block_size;
            int N = this->_ffn_hidden_size;
            int K = this->_hidden_size;

            auto end1 = system_clock::now();
            auto duration1 = duration_cast<microseconds>(end1 - start);

            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            first_mix_gemm(this->_moe_data_nvshmem, 
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(merged_input.data_ptr()),
                    reinterpret_cast<half*>(weight.data_ptr()),
                    reinterpret_cast<half*>(bias.data_ptr()),
                    reinterpret_cast<half*>(ffn1_out.data_ptr()), 
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );

            auto end2 = system_clock::now();
            auto duration = duration_cast<microseconds>(end2 - end1);

            // std::cout<<"perpare first forward gemm: "
            //         << duration1.count() / 1000.0 
            //         <<",  fisrt gemm forward duration: "
            //         << duration.count()/1000.0
            //         <<std::endl;
            
            return  {merged_input, ffn1_out};
        }

        std::vector<torch::Tensor> first_gemm_backward(torch::Tensor grad_out, torch::Tensor weight, torch::Tensor bias,
                    torch::Tensor merged_input, torch::Tensor expert_block_ids, torch::Tensor expert_id_idxs,
                    torch::Tensor pos, torch::Tensor expert_ids, torch::Tensor ranks, torch::Tensor expert_lens, 
                    torch::Tensor start_addrs, int total_block_num, int remote_block_num){
            
            auto start = system_clock::now();


            auto opt = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(grad_out.device());

            // M N K 有问题
            int M = total_block_num*this->_block_size;
            int N = this->_hidden_size;
            int K = this->_ffn_hidden_size;
            auto temp_remote_mem = torch::zeros({remote_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            // nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            auto end1 = system_clock::now();
            auto duration1 = duration_cast<microseconds>(end1 - start);
            
            second_mix_gemm(reinterpret_cast<half*>(grad_out.data_ptr()),
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(weight.data_ptr()),
                    reinterpret_cast<half*>(bias.data_ptr()),
                    this->_moe_data_nvshmem,
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );

            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            auto end2 = system_clock::now();
            auto duration2 = duration_cast<microseconds>(end2 - end1);
            
            auto grad_weight = torch::empty_like(weight);
            auto grad_bias = torch::empty_like(bias);
            grad_out = grad_out.transpose(-2, -1).contiguous();

            // merged_input(T) * grad_out 
            M = weight.sizes()[1];
            N = weight.sizes()[2];


            launch_compute_weight_grad(reinterpret_cast<half*>(grad_out.data_ptr()),
                    reinterpret_cast<half*>(merged_input.data_ptr()),
                    reinterpret_cast<half*>(grad_weight.data_ptr()),
                    reinterpret_cast<int*>(expert_block_ids.data_ptr()),
                    reinterpret_cast<int*>(expert_id_idxs.data_ptr()),
                    this->_local_expert_num, total_block_num, M, N
                    );
            
            auto end3 = system_clock::now();
            auto duration3 = duration_cast<microseconds>(end3 - end2);

            // divide pos / 2
            auto temp_pos = torch::div(pos, this->_top_k, "floor");
            auto grad_in = torch::zeros({pos.numel()/ this->_top_k, this->_hidden_size}, opt).contiguous();



            auto nvshmem_data_tensor = torch::from_blob(this->_moe_data_nvshmem, {pos.numel(), this->_hidden_size}, opt);
            grad_in.index_add_(0, temp_pos, nvshmem_data_tensor);

            auto end4 = system_clock::now();
            auto duration4 = duration_cast<microseconds>(end4 - end3);

            // std::cout<<"perpare first backward gemm: "
            //         << duration1.count() / 1000.0 
            //         <<",  fisrt gemm backward duration: "
            //         << duration2.count()/1000.0
            //         <<",  weight grad: "
            //         << duration3.count()/1000.0
            //         <<",  other: "
            //         << duration4.count()/1000.0
            //         <<std::endl;
            
            
            return {grad_in, grad_weight, grad_bias};

            
            // grad_out(ffn2_grad)
        }

        std::vector<torch::Tensor> second_gemm_forward(torch::Tensor ffn1_out, torch::Tensor pos, torch::Tensor expert_ids, 
                    torch::Tensor ranks, torch::Tensor expert_lens, torch::Tensor start_addrs,
                    int total_block_num, int remote_block_num, torch::Tensor weight, torch::Tensor bias
                    ){
            
            auto opt = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(ffn1_out.device());
            

            // nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            auto temp_remote_mem = torch::zeros({remote_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            int M = total_block_num*this->_block_size;
            int N = this->_hidden_size;
            int K = this->_ffn_hidden_size;
            second_mix_gemm(reinterpret_cast<half*>(ffn1_out.data_ptr()),
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(weight.data_ptr()),
                    reinterpret_cast<half*>(bias.data_ptr()),
                    this->_moe_data_nvshmem,
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );

            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);


            auto output = torch::zeros({pos.numel(), this->_hidden_size}, opt).contiguous();


            // local_gather(this->_moe_data_nvshmem, reinterpret_cast<half*>(output.data_ptr()), 
            //     reinterpret_cast<int*>(pos.data_ptr()), pos.numel());



            torch::Tensor data_tensor = torch::from_blob(this->_moe_data_nvshmem, {pos.numel(), this->_hidden_size}, opt);

            output.index_copy_(0, pos, data_tensor);


            return {ffn1_out, output};
        }

        std::vector<torch::Tensor> second_gemm_backward(torch::Tensor grad_out, torch::Tensor input_buf, 
                    torch::Tensor expert_block_ids, torch::Tensor expert_id_idxs,
                    torch::Tensor weight, torch::Tensor bias, torch::Tensor pos, torch::Tensor expert_ids, 
                    torch::Tensor ranks, torch::Tensor expert_lens, torch::Tensor start_addrs, int total_block_num, int remote_block_num){

            if(pos.dtype() != torch::kInt32){
                pos = pos.to(torch::kInt32);
            }
            if(grad_out.dtype() != torch::kFloat16){
                printf("moe input data type must be float 16\n");
                return {torch::empty({0})};
            }

            local_scatter(reinterpret_cast<half*>(grad_out.data_ptr()), this->_moe_data_nvshmem, 
                reinterpret_cast<int*>(pos.data_ptr()), pos.numel());

            auto opt = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(grad_out.device());
            auto temp_remote_mem = torch::zeros({remote_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            auto grad_inp_buf = torch::zeros({total_block_num*this->_block_size, this->_ffn_hidden_size}, opt).contiguous();
            auto merged_grad_out = torch::zeros({total_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            
            // M N K 有问题
            int M = total_block_num*this->_block_size;
            int N = this->_ffn_hidden_size;
            int K = this->_hidden_size;

            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            first_mix_gemm(this->_moe_data_nvshmem, 
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(merged_grad_out.data_ptr()),
                    reinterpret_cast<half*>(weight.data_ptr()),
                    reinterpret_cast<half*>(bias.data_ptr()),
                    reinterpret_cast<half*>(grad_inp_buf.data_ptr()),
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );

            // grad_out(T) @ merged_input_buf  merged_input_buf(ffn1_out) * merged_grad_out
            // grad_sum 

            auto grad_weight = torch::empty_like(weight);
            auto grad_bias = torch::empty_like(bias);
            merged_grad_out = merged_grad_out.transpose(-2, -1).contiguous();

            // merged_input(T) * grad_out 
            M = weight.sizes()[1];
            N = weight.sizes()[2];

            launch_compute_weight_grad(reinterpret_cast<half*>(merged_grad_out.data_ptr()),
                    reinterpret_cast<half*>(input_buf.data_ptr()),
                    reinterpret_cast<half*>(grad_weight.data_ptr()),
                    reinterpret_cast<int*>(expert_block_ids.data_ptr()),
                    reinterpret_cast<int*>(expert_id_idxs.data_ptr()),
                    this->_local_expert_num, total_block_num, M, N
                    );

            return {grad_inp_buf, grad_weight, grad_bias};
            
        }

        torch::Tensor forward(torch::Tensor inp, torch::Tensor pos, torch::Tensor expert_ids, torch::Tensor ranks,
                    torch::Tensor expert_lens, torch::Tensor start_addrs, int total_block_num, int remote_block_num,
                    torch::Tensor test_out){
            if(pos.dtype() != torch::kInt32){
                pos = pos.to(torch::kInt32);
            }
            if(inp.dtype() != torch::kFloat16){
                printf("moe input data type must be float 16\n");
                return torch::empty({0});
            }

            auto temp_pos = torch::div(pos, this->_top_k, "floor");

            local_scatter(reinterpret_cast<half*>(inp.data_ptr()), this->_moe_data_nvshmem, 
                reinterpret_cast<int*>(temp_pos.data_ptr()), pos.numel());

            auto opt = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(inp.device());

            
            auto temp_remote_mem = torch::zeros({remote_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            auto merged_input = torch::zeros({total_block_num*this->_block_size, this->_hidden_size}, opt).contiguous();
            auto ffn1_out = torch::zeros({total_block_num*this->_block_size, this->_ffn_hidden_size}, opt).contiguous();


            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);

            int M = total_block_num*this->_block_size;
            int N = this->_ffn_hidden_size;
            int K = this->_hidden_size;

            first_mix_gemm(this->_moe_data_nvshmem, 
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(merged_input.data_ptr()),
                    reinterpret_cast<half*>(this->_expert_weight_ffn1.data_ptr()),
                    reinterpret_cast<half*>(this->_expert_bias_ffn1.data_ptr()),
                    reinterpret_cast<half*>(ffn1_out.data_ptr()), 
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );

            auto activate = torch::nn::GELU();

            ffn1_out = activate(ffn1_out);

            M = total_block_num*this->_block_size;
            N = this->_hidden_size;
            K = this->_ffn_hidden_size;
            second_mix_gemm(reinterpret_cast<half*>(ffn1_out.data_ptr()),
                    reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
                    reinterpret_cast<half*>(this->_expert_weight_ffn2.data_ptr()),
                    reinterpret_cast<half*>(this->_expert_bias_ffn2.data_ptr()),
                    this->_moe_data_nvshmem,
                    reinterpret_cast<int*>(expert_ids.data_ptr()),
                    reinterpret_cast<int*>(ranks.data_ptr()), 
                    reinterpret_cast<int*>(expert_lens.data_ptr()),
                    reinterpret_cast<int*>(start_addrs.data_ptr()), 
                    this->_rank, total_block_num, remote_block_num,
                    M, N, K,
                    this->nvshmem_stream
                    );

            
            nvshmemx_barrier_all_on_stream(this->nvshmem_stream);
            cudaStreamSynchronize(this->nvshmem_stream);

            auto output = torch::zeros({pos.numel(), this->_hidden_size}, opt).contiguous();

            local_gather(this->_moe_data_nvshmem, reinterpret_cast<half*>(output.data_ptr()), 
                reinterpret_cast<int*>(pos.data_ptr()), pos.numel());

                       

            // this->compare_result(reinterpret_cast<float*>(ffn1_out.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(test_out.to(torch::kFloat32).to(torch::kCPU).data_ptr()), 128*this->_ffn_hidden_size);
            
            // std::cout<< pos.sizes() << " "<< pos.device() << std::endl;

            // half* data_ = (half *)malloc(pos.numel()*this->_hidden_size*sizeof(half));
            // cudaMemcpy(data_, this->_moe_data_nvshmem, pos.numel()*this->_hidden_size*sizeof(half), cudaMemcpyDeviceToHost);
            torch::Tensor data_tensor = torch::from_blob(this->_moe_data_nvshmem, {pos.numel(), this->_hidden_size}, opt);
            // this->compare_result(reinterpret_cast<float*>(test_out.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(data_tensor.to(torch::kFloat32).to(torch::kCPU).data_ptr()),test_out.numel());
            
            return output;


        }

        ~_MixMOE(){
            this->finalize_nvshmem();
        }
};