cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MGG LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(HOME "/workspace")


#############################################################################
set($ENV{PATH} "${HOME}/local/openmpi/bin/:$ENV{PATH}")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(MPI_HOME "${HOME}/local/openmpi")
set(CUDA_HOME "/usr/local/cuda")
set(CUDNN_HOME "${HOME}/local/cudnn-v8.2")
set(NVSHMEM_HOME "${HOME}/local/nvshmem")
set(LIBTORCH_HOME "${HOME}/local/libtorch")


set(SM_ARCH "sm_80")
set($ENV{NVCC_GENCODE} ${SM_ARCH})
set($ENV{NVSHMEM_USE_GDRCOPY} "0")

message("${PATH}")


find_package(Torch REQUIRED
            NO_MODULE
            PATHS ${LIBTORCH_HOME}
            NO_DEFAULT_PATH)

message("${NVSHMEM_HOME}")


link_directories(${NVSHMEM_HOME}/lib)
link_directories(${CUDNN_HOME}/lib64)
link_directories(${MPI_HOME}/lib)
link_directories(/usr/local/lib)
# link_directories(${NCCL_HOME}/lib)


add_executable(test 
    ./cuda/mix_gemm.cu
    ./cuda/mix_moe_kernel.cu
    ./mix_moe.cpp
    ./test.cpp
)

target_include_directories(test 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
)

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(test
    nvshmem 
    nvidia-ml
    cuda
    mpi_cxx 
    mpi 
    cudnn 
    gomp 
    "${TORCH_LIBRARIES}"
)