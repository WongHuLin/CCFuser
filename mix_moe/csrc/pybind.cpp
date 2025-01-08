#include <torch/extension.h>
#include "mix_moe.h"
#include "cuda/local_exchange.cuh"
#include "cuda/global_exchange.h"




// local_exchange
void _assign_pos(
        torch::Tensor cum_count,
        torch::Tensor gate,
        torch::Tensor pos);
void _expert_count(
        torch::Tensor gate_idx,
        torch::Tensor expert_count);

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

void _ensure_nccl(c10d::ProcessGroup& p, torch::Tensor t);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

#ifdef FMOE_USE_NCCL
    m.def("expert_all_gather", &_expert_all_gather, "FastMoE expert all gather (CUDA)");
    m.def("expert_exchange", &_expert_exchange, "FastMoE expert exchange (CUDA)");
    m.def("global_scatter", &_global_scatter, "FastMoE global scatter (CUDA)");
    m.def("global_gather", &_global_gather, "FastMoE global gather (CUDA)");
    m.def("ensure_nccl", &_ensure_nccl, "FastMoE global gather (CUDA)");
    m.def("expert_rebalance", &_expert_rebalance, "expert rebalance");
    m.def("cpy_data2tensor", &_cpy_data2tensor, "cpy data to cuda tensor");


    m.def("expert_count", &_expert_count, "FastMoE count gate indices (CUDA)");
    m.def("assign_pos", &_assign_pos, "FastMoE assign pos by gate (CUDA)");
#endif

    pybind11::class_<_MixMOE>(m, "MixMOE")
        .def(pybind11::init([](int batch_size, int seq_len, int local_expert_num, int top_k,
                int hidden_size, int ffn_hidden_size, int block_size, torch::Tensor &expert_weight_ffn1,
                torch::Tensor &expert_weight_ffn2, torch::Tensor &expert_bias_ffn1, torch::Tensor &expert_bias_ffn2) -> _MixMOE * {
                return new _MixMOE(batch_size, seq_len, local_expert_num, top_k, hidden_size,
                    ffn_hidden_size, block_size, std::move(expert_weight_ffn1), std::move(expert_weight_ffn2),
                    std::move(expert_bias_ffn1), std::move(expert_bias_ffn2));
        }))
        .def(pybind11::init([](int batch_size, int seq_len, int local_expert_num, int top_k,
                int hidden_size, int ffn_hidden_size, int block_size) -> _MixMOE * {
                return new _MixMOE(batch_size, seq_len, local_expert_num, top_k, hidden_size,
                    ffn_hidden_size, block_size);
        }))
        .def("forward", &_MixMOE::forward)
        .def("first_gemm_forward", &_MixMOE::first_gemm_forward)
        .def("second_gemm_forward", &_MixMOE::second_gemm_forward)
        .def("first_gemm_backward", &_MixMOE::first_gemm_backward)
        .def("second_gemm_backward", &_MixMOE::second_gemm_backward)
        .def("AG_test", &_MixMOE::AG_test)
        .def("GA_test", &_MixMOE::GA_test);

}
