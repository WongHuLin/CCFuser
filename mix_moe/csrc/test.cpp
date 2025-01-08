#include <torch/script.h>
#include "torch/torch.h"
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <unistd.h>
#include <cuda_runtime.h>

#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include "cuda/mix_gemm.cuh"

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));
    input.close();
    return bytes;
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template<class type>
void generate_array(type* data, int len, int pe_id){
    for(int i=0;i<len;i++){
        data[i] = type((i + len*pe_id));
    }
}

void generate_temp_data_array(float* data, int remote_data_block, int* ranks, int* start_addrs, int* lens, int n_dim, size_t total_size){
    for(int i=0;i<remote_data_block; i++){
        int idx = i*2+1;
        int rank_id = ranks[idx];
        size_t start_addr = start_addrs[idx];
        size_t start_a = rank_id*total_size;
        if(i == 0)
            printf("%ld %ld\n", total_size, start_addr*n_dim);
        int len = lens[idx];
        for(int j=0;j<len*n_dim;j++){
            data[i*64*n_dim+j] = (float)(start_a+ start_addr*n_dim + j);
        }
    }

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

void get_nvshmem_data(float* data, int remote_data_block, int* ranks, int* start_addrs, int* lens, int n_dim, size_t total_size){
    std::vector<torch::Tensor> x;
    for(int i=0; i<2; i++){
        std::string file_name = "/workspace/mix_moe/csrc/moe_inp_" + std::to_string(i);
        std::vector<char> f = get_the_bytes(file_name);
        torch::Tensor tensor = torch::pickle_load(f).toTensor().to(torch::kCPU);
        x.push_back(tensor);
    }


    for(int i=0;i<remote_data_block; i++){
        int idx = i;
        int rank_id = ranks[idx];
        size_t start_addr = start_addrs[idx];
        int len = lens[idx];
        float* temp_data = reinterpret_cast<float*>(x[rank_id].data_ptr());
        for(int j=0;j<len*n_dim;j++){
            data[i*64*n_dim+j] = temp_data[start_addr*n_dim + j];
        }
    }
}

torch::Tensor load_test_data(int rank){
    std::string file_name = "/workspace/mix_moe/csrc/moe_inp_" + std::to_string(rank);
    std::vector<char> f = get_the_bytes(file_name);
    torch::Tensor tensor = torch::pickle_load(f).toTensor().to(torch::kCPU).to(torch::kFloat16);

    return tensor;
}

int main(int c, char *v[]) {


    MPI_Init(&c, &v);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int mype, npes;

    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);

    std::string file_name = "/root/fastmoe/tests/blcok_info_" + std::to_string(rank);

    std::vector<char> f = get_the_bytes(file_name);
    c10::Dict<c10::IValue, c10::IValue> dict = torch::pickle_load(f).toGenericDict();
    torch::IValue data_expert_id("expert_id");
    torch::IValue data_rank("rank");
    torch::IValue data_expert_addr_len("expert_addr_len");
    torch::IValue data_start_addr("start_addr");
    torch::IValue remote_block_num("remote_block_num");

    torch::Device device(torch::kCUDA);

    // torch::IValue key_b("key_b");
    torch::Tensor expert_id_tensor = dict.at(data_expert_id).toTensor().to(device);
    torch::Tensor rank_tensor = dict.at(data_rank).toTensor().to(device);
    torch::Tensor expert_addr_len_tensor = dict.at(data_expert_addr_len).toTensor().to(device);
    torch::Tensor start_addr_tensor = dict.at(data_start_addr).toTensor().to(device);
    int remote_block_num_ = dict.at(remote_block_num).toInt();
    int block_num = expert_id_tensor.sizes()[0];


    int seq_len = 8192;
    int d_num = 1024;
    size_t moe_inp_size = seq_len*d_num*sizeof(half);
    torch::Tensor moe_inp_h = load_test_data(rank);
    half* moe_inp_d = (half *)nvshmem_malloc(moe_inp_size);
    cudaMemcpy(moe_inp_d, reinterpret_cast<half*>(moe_inp_h.data_ptr()), moe_inp_size, cudaMemcpyHostToDevice);

    torch::Tensor weight = torch::rand({1024,1024}, torch::kFloat16).to(device).contiguous();


    auto temp_remote_mem = torch::zeros({remote_block_num_*64, d_num}, torch::kFloat16).to(device).contiguous();

    auto temp_remote_mem_1 = torch::zeros({block_num*64, d_num}, torch::kFloat32);

    get_nvshmem_data(reinterpret_cast<float*>(temp_remote_mem_1.data_ptr()), block_num,
        reinterpret_cast<int*>(rank_tensor.to(torch::kCPU).data_ptr()),
        reinterpret_cast<int*>(start_addr_tensor.to(torch::kCPU).data_ptr()),
        reinterpret_cast<int*>(expert_addr_len_tensor.to(torch::kCPU).data_ptr()),
        d_num, seq_len*d_num
        );

    temp_remote_mem_1 = temp_remote_mem_1.to(torch::kFloat16).to(device).contiguous();

    auto out_1 = torch::matmul(temp_remote_mem_1, weight).contiguous();
    torch::Tensor out_2 = torch::zeros({block_num*64, d_num}, torch::kFloat16).to(device).contiguous();

    std::cout<< "my rank is "<< rank << " "<< remote_block_num_<< std::endl;
    // std::cout<< "my rank is "<< rank << " "<< rank_tensor.numel() << std::endl;
    // std::cout<< "my rank is "<< rank << " "<< expert_addr_len_tensor.numel() << std::endl;
    // std::cout<< "my rank is "<< rank << " "<< start_addr_tensor.numel() << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout<<"start computation: " << std::time(0) <<std::endl;

    mix_gemm(moe_inp_d, reinterpret_cast<half*>(temp_remote_mem.data_ptr()),
            reinterpret_cast<half*>(weight.data_ptr()),
            reinterpret_cast<half*>(out_2.data_ptr()), 
            reinterpret_cast<int*>(expert_id_tensor.data_ptr()),
            reinterpret_cast<int*>(rank_tensor.data_ptr()), 
            reinterpret_cast<int*>(expert_addr_len_tensor.data_ptr()),
            reinterpret_cast<int*>(start_addr_tensor.data_ptr()), 
            d_num, rank, block_num, remote_block_num_);

    MPI_Barrier(MPI_COMM_WORLD);


    // sleep(10);
    // std::cout<<"my rank is "<< rank << " "<< rank_tensor[1] <<" " << start_addr_tensor[1] << " "<< expert_addr_len_tensor[1] << std::endl;

    // std::cout<<"my rank is "<< rank << " " << temp_remote_mem_1[0][1] << std::endl;
    // std::cout<<"my rank is "<< rank << " " << temp_remote_mem[0][1] << std::endl;

    compare_result(reinterpret_cast<float*>(out_1.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(out_2.to(torch::kFloat32).to(torch::kCPU).data_ptr()),out_1.numel());

    // test(mpi_comm, reinterpret_cast<half*>(input.toType(torch::kFloat16).data_ptr()),  reinterpret_cast<half*>(weight.toType(torch::kFloat16).data_ptr()), seq_len, d_num);
    nvshmem_finalize();
    
    MPI_Finalize();
    return 0;

}






#define OFFSET(row, col, ld) ((row) * (ld) + (col))


void cpuF16F16Gemm(half *a, half *b, half *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}


float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int), 
    int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    std::cout<<"start test"<<std::endl;

    half *h_a, *h_b, *d_a, *d_b;
    half *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    std::cout<<"start cpu test"<<std::endl;
    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    std::cout<<"start gpu test"<<std::endl;
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

    std::cout<<max_error<<std::endl;

    return max_error;
}


#include <cublas_v2.h>

#define checkCublasErrors(func)                                                                     \
{                                                                                                   \
    cublasStatus_t e = (func);                                                                      \
    if (e != CUBLAS_STATUS_SUCCESS) {                                                               \
        printf ("%s %d CUDA: ", __FILE__,  __LINE__);                                               \
        switch (e) {                                                                                \
            case CUBLAS_STATUS_NOT_INITIALIZED : printf("CUBLAS_STATUS_NOT_INITIALIZED\n"); break;  \
            case CUBLAS_STATUS_ALLOC_FAILED    : printf("CUBLAS_STATUS_NOT_INITIALIZED\n"); break;  \
            case CUBLAS_STATUS_INVALID_VALUE   : printf("CUBLAS_STATUS_INVALID_VALUE\n"); break;    \
            case CUBLAS_STATUS_ARCH_MISMATCH   : printf("CUBLAS_STATUS_ARCH_MISMATCH\n"); break;    \
            case CUBLAS_STATUS_MAPPING_ERROR   : printf("CUBLAS_STATUS_MAPPING_ERROR\n"); break;    \
            case CUBLAS_STATUS_EXECUTION_FAILED: printf("CUBLAS_STATUS_EXECUTION_FAILED\n"); break; \
            case CUBLAS_STATUS_INTERNAL_ERROR  : printf("CUBLAS_STATUS_INTERNAL_ERROR\n"); break;   \
            case CUBLAS_STATUS_NOT_SUPPORTED   : printf("CUBLAS_STATUS_NOT_SUPPORTED\n"); break;    \
            case CUBLAS_STATUS_LICENSE_ERROR   : printf("CUBLAS_STATUS_LICENSE_ERROR\n"); break;    \
            default: break;                                                                         \
        }                                                                                           \
    }                                                                                               \
}



template <cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP>
void cublasF16F16Gemm(
    const half *a, const half *b, half *c, int M, int N, int K) {

    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha = 1.0;
    half beta = 0.0;

    int repeat = 10;


    for(int i=0;i<5;i++)
        checkCublasErrors(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K, &beta, c, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F, algo));
    cudaDeviceSynchronize();
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for(int i=0;i<repeat;i++)
    {
        checkCublasErrors(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K, &beta, c, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F, algo));

    }

        cudaDeviceSynchronize();


    cudaEventRecord(end);

    cudaEventSynchronize(end);


    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 /repeat;
    printf("M: %d, N: %d, K: %d, avg excution Time: %f \n",M,N,K,sec);

    cublasDestroy(handle);
}

// // test gemm
// int main() {

//     int M = 2048;
//     int N = 768 * 4;
//     int K = 768;

//     // testF16F16GemmMaxError(test_gemm, M, N, K);

//     torch::Device device(torch::kCUDA);
//     torch::Tensor a = torch::rand({M,K}, torch::kFloat16).to(device).contiguous();
//     torch::Tensor b = torch::rand({K,N}, torch::kFloat16).to(device).contiguous();
//     torch::Tensor c = torch::zeros({M,N}, torch::kFloat16).to(device).contiguous();

//     torch::Tensor c_cpu = torch::zeros({M,N}, torch::kFloat16).contiguous();
//     torch::Tensor c_cublas = torch::zeros({M,N}, torch::kFloat16).to(device).contiguous();


//     // cpuF16F16Gemm(reinterpret_cast<half*>(a.to(torch::kCPU).data_ptr()),
//     //     reinterpret_cast<half*>(b.to(torch::kCPU).data_ptr()),
//     //     reinterpret_cast<half*>(c_cpu.to(torch::kCPU).data_ptr()),
//     //     M,N,K);

//     cublasF16F16Gemm(reinterpret_cast<half*>(a.data_ptr()),
//         reinterpret_cast<half*>(b.data_ptr()),
//         reinterpret_cast<half*>(c_cublas.data_ptr()),
//         M,N,K);
    

//     torch::Tensor c_1 = torch::matmul(a, b);

//     // test_gemm(reinterpret_cast<half*>(a.data_ptr()),reinterpret_cast<half*>(b.data_ptr()),reinterpret_cast<half*>(c.data_ptr()),M,N,K);


//     // compare_result(reinterpret_cast<float*>(c_1.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(c_cublas.to(torch::kFloat32).to(torch::kCPU).data_ptr()),c.numel());

//     // compare_result(reinterpret_cast<float*>(c_cpu.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(c.to(torch::kFloat32).to(torch::kCPU).data_ptr()),c.numel());


//     // compare_result(reinterpret_cast<float*>(c_cpu.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(c_1.to(torch::kFloat32).to(torch::kCPU).data_ptr()),c.numel());


//     // compare_result(reinterpret_cast<float*>(c.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(c_1.to(torch::kFloat32).to(torch::kCPU).data_ptr()),c.numel());

//     return 0;
// }


// test local scatter
// #include "cuda/mix_moe_kernel.cuh"
// #include "mix_moe.h"
// int main(){


//     // std::cout<<torch::index_select(pos, 0, torch::arange(32).to(device))<<std::endl;

//     printf("test local scatter\n");

//     int M = 4096;
//     int N = 1024;

//     _MixMOE mixmoe(4,1024,16,2,N,N,torch::empty({0,1}),torch::empty({0,1}));

//     // torch::

//     torch::Device device(torch::kCUDA);


//     std::string file_name = "/workspace/mix_moe/csrc/pos.pt";
//     std::vector<char> f = get_the_bytes(file_name);
//     torch::Tensor pos = torch::pickle_load(f).toTensor().to(torch::kInt32).to(device);


//     torch::Tensor input = torch::rand({M,N}, torch::kFloat16).to(device).contiguous();
//     torch::Tensor output_1 = torch::zeros({pos.numel(),N}, torch::kFloat16).to(device).contiguous();

//     std::cout<< input.device()<<std::endl;
//     std::cout<< pos.device()<<std::endl;


//     mixmoe.local_scatter(reinterpret_cast<half*>(input.data_ptr()), reinterpret_cast<half*>(output_1.data_ptr()), reinterpret_cast<int*>(pos.data_ptr()), pos.numel());

//     half* data_ = (half *)malloc(pos.numel()*N*sizeof(half));
//     cudaMemcpy(data_, mixmoe._moe_data_nvshmem, pos.numel()*N*sizeof(half), cudaMemcpyDeviceToHost);
//     torch::Tensor data_tensor = torch::from_blob(data_, {pos.numel(), N}, dtype(torch::kFloat16));


//     _local_scatter_launch(reinterpret_cast<half*>(input.data_ptr()), reinterpret_cast<half*>(output_1.data_ptr()), N, reinterpret_cast<int*>(pos.data_ptr()), pos.numel());

//     torch::Tensor output_2 = torch::index_select(input, 0, pos);

//     compare_result(reinterpret_cast<float*>(data_tensor.to(torch::kFloat32).to(torch::kCPU).data_ptr()), reinterpret_cast<float*>(output_1.to(torch::kFloat32).to(torch::kCPU).data_ptr()),output_1.numel());


//     std::cout<<pos.numel()<<std::endl;
// }