# import setuptools
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# import os
# import torch

# cxx_flags = []
# nvcc_flags = []
# ext_libs = []



# is_rocm_pytorch = False
# if torch.__version__ >= '1.5':
#     from torch.utils.cpp_extension import ROCM_HOME
#     is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

# if os.environ.get('USE_NCCL', '1') == '1':
#     cxx_flags.append('-DFMOE_USE_NCCL')
#     cxx_flags.append('-DUSE_C10D_NCCL')
#     if is_rocm_pytorch:
#         ext_libs.append('rccl')
#     else:
#         ext_libs.append('nccl')

# if os.environ.get('MOE_DEBUG', '0') == '1':
#     cxx_flags.append('-DMOE_DEBUG')

# if is_rocm_pytorch:
#     define_macros=[('FMOE_USE_HIP', None)]
# else:
#     define_macros=[]

# define_macros.append(('CUDA_SEPARABLE_COMPILATION', 'ON'))
# define_macros.append(('CUDA_ARCHITECTURES', '80'))
# define_macros.append(('CUDA_RESOLVE_DEVICE_SYMBOLS', 'ON'))



# # nvcc_flags.append('cudart')
# nvcc_flags.append('-rdc=true')
# ext_libs.extend(['cuda', 'nvshmem', 'mpi_cxx', 'mpi'])

# dlink_libraries = ['cuda', 'nvshmem', 'nccl']

# include_dirs = [
#                 '/home/xiayaqi/project/mix_moe/cuda-11.7/include',
#                 '/home/xiayaqi/project/mix_moe/MixMoe//local/openmpi/include',
#                 '/home/xiayaqi/project/mix_moe/MixMoe/local/cudnn-v8.2/include',
#                 '/home/xiayaqi/project/mix_moe/MixMoe/local/nvshmem/include',
#                 '/home/xiayaqi/project/mix_moe/nccl/usr/include',
#                 ]

# library_dirs = [
#                 '/home/xiayaqi/project/mix_moe/MixMoe//local/openmpi/lib',
#                 '/home/xiayaqi/project/mix_moe/MixMoe//local/cudnn-v8.2/lib64',
#                 '/home/xiayaqi/project/mix_moe/MixMoe//local/nvshmem/lib',
#                 '/home/xiayaqi/project/mix_moe/cuda-11.7/lib64',
#                  '/home/xiayaqi/project/mix_moe/nccl/usr/lib64',
#                 ]

# if __name__ == '__main__':
#     setuptools.setup(
#         name='mixmoe',
#         version='1.1.0',
#         packages=['mix_moe', 'mix_moe.gates'],
#         ext_modules=[
#             CUDAExtension(
#                 name='mix_moe_cuda', 
#                 sources=[
#                     'csrc/cuda/global_exchange.cpp',
#                     'csrc/cuda/local_exchange.cu',
#                     'csrc/cuda/mix_moe_kernel.cu',
#                     'csrc/cuda/mix_gemm.cu',
#                     'csrc/mix_moe.cpp',
#                     'csrc/stream_manager.cpp',
#                     'csrc/pybind.cpp',
#                     ],
#                 dlink=True,
#                 dlink_libraries = dlink_libraries,
#                 include_dirs = include_dirs,
#                 library_dirs = library_dirs,
#                 define_macros=define_macros,
#                 extra_compile_args={
#                     'cxx': cxx_flags,
#                     'nvcc': nvcc_flags
#                     },
#                 libraries=ext_libs
#                 )
#             ],
#         cmdclass={
#             'build_ext': BuildExtension
#         })















import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

cxx_flags = []
nvcc_flags = []
ext_libs = []



is_rocm_pytorch = False
if torch.__version__ >= '1.5':
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

if os.environ.get('USE_NCCL', '1') == '1':
    cxx_flags.append('-DFMOE_USE_NCCL')
    cxx_flags.append('-DUSE_C10D_NCCL')
    if is_rocm_pytorch:
        ext_libs.append('rccl')
    else:
        ext_libs.append('nccl')

if os.environ.get('MOE_DEBUG', '0') == '1':
    cxx_flags.append('-DMOE_DEBUG')

if is_rocm_pytorch:
    define_macros=[('FMOE_USE_HIP', None)]
else:
    define_macros=[]

define_macros.append(('CUDA_SEPARABLE_COMPILATION', 'ON'))
define_macros.append(('CUDA_ARCHITECTURES', '80'))
define_macros.append(('CUDA_RESOLVE_DEVICE_SYMBOLS', 'ON'))



# nvcc_flags.append('cudart')
nvcc_flags.append('-rdc=true')
ext_libs.extend(['cuda', 'nvshmem', 'mpi_cxx', 'mpi', 'nvidia-ml'])

dlink_libraries = ['cuda', 'nvshmem', 'nccl', 'nvidia-ml']

include_dirs = [
                '/usr/local/cuda/include',
                '/root/local/openmpi/include',
                '/root/local/cudnn-v8.2/include',
                '/root/local/nvshmem/include',
                ]

library_dirs = [
                '/root/local/openmpi/lib',
                '/root/local/cudnn-v8.2/lib64',
                '/root/local/nvshmem/lib',
                ]

if __name__ == '__main__':
    setuptools.setup(
        name='mixmoe',
        version='1.1.0',
        packages=['mix_moe', 'mix_moe.gates'],
        ext_modules=[
            CUDAExtension(
                name='mix_moe_cuda', 
                sources=[
                    'csrc/cuda/global_exchange.cpp',
                    'csrc/cuda/local_exchange.cu',
                    'csrc/cuda/mix_moe_kernel.cu',
                    'csrc/cuda/mix_gemm.cu',
                    'csrc/mix_moe.cpp',
                    'csrc/stream_manager.cpp',
                    'csrc/pybind.cpp',
                    ],
                dlink=True,
                dlink_libraries = dlink_libraries,
                include_dirs = include_dirs,
                library_dirs = library_dirs,
                define_macros=define_macros,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': nvcc_flags
                    },
                libraries=ext_libs
                )
            ],
        cmdclass={
            'build_ext': BuildExtension
        })
