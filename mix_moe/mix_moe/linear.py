r"""
FMoE's parallel linear layer
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import math
import mix_moe_cuda


import time
import threading


class MixMOEInstance(object):
    _instance_lock = threading.Lock()

    def __init__(self, total_batch_size, num_expert, top_k, d_model, ffn_hidden_size, block_size):
        self.mix_moe =  mix_moe_cuda.MixMOE(1, total_batch_size, num_expert, top_k, 
                        d_model, ffn_hidden_size, block_size, )
        time.sleep(1)

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(MixMOEInstance, "_instance"):
            with MixMOEInstance._instance_lock:
                if not hasattr(MixMOEInstance, "_instance"):
                    MixMOEInstance._instance = MixMOEInstance(*args, **kwargs)
        return MixMOEInstance._instance
    
# class MOELinear(Function):
#     r"""
#     Computes linear operators within one GPU on different experts simutaneously.
#     """

#     @staticmethod
#     def forward(ctx, global_input_buf, fwd_expert_count, weight, bias=None):
#         global_output_buf = fmoe_cuda.linear_forward(
#             global_input_buf, fwd_expert_count, weight, bias
#         )
#         variables = (global_input_buf, fwd_expert_count, weight, bias)
#         ctx.save_for_backward(*variables)
#         return global_output_buf

#     @staticmethod
#     def backward(ctx, grad_out):
#         (input_buf, fwd_expert_count, weight, bias) = ctx.saved_tensors
#         grad_inp_buf, grad_weight, grad_bias = fmoe_cuda.linear_backward(
#             grad_out, input_buf, fwd_expert_count, weight, bias
#         )

#         if not torch.is_tensor(bias):
#             grad_bias = None

#         return grad_inp_buf, None, grad_weight, grad_bias




class MixMOELinear1(Function):

    @staticmethod
    def forward(ctx, input_buf, pos, moe_inp_block_infos, weight,
                bias = None):

        mix_moe_instance = MixMOEInstance.instance()
        merged_input, ffn1_out = mix_moe_instance.mix_moe.first_gemm_forward(input_buf, pos, 
                            moe_inp_block_infos['local_expert_id_indexs'], 
                            moe_inp_block_infos['ranks'],
                            moe_inp_block_infos['lens'],
                            moe_inp_block_infos['start_addrs'], 
                            moe_inp_block_infos['total_block_num'],
                            moe_inp_block_infos['remote_block_num'], 
                            weight.transpose(-2, -1).contiguous(), bias.contiguous())

        variables = (merged_input, weight, bias, pos, moe_inp_block_infos['local_expert_id_indexs'], 
                            moe_inp_block_infos['ranks'],
                            moe_inp_block_infos['lens'],
                            moe_inp_block_infos['start_addrs'], 
                            torch.tensor([moe_inp_block_infos['total_block_num'], moe_inp_block_infos['remote_block_num']],),
                            moe_inp_block_infos['expert_block_id_idxs'],
                            moe_inp_block_infos['expert_block_ids'],)

        ctx.save_for_backward(*variables)

        return ffn1_out
        
        pass
        # return output
    
    @staticmethod
    def backward(ctx, grad_out):

        (merged_input, weight, bias, pos, local_expert_id_indexs, 
                            ranks,
                            lens,
                            start_addrs, 
                            block_num_tensor,
                            expert_block_id_idxs,
                            expert_block_ids) = ctx.saved_tensors
        mix_moe_instance = MixMOEInstance.instance()

        grad_in, grad_weight, grad_bias = mix_moe_instance.mix_moe.first_gemm_backward(grad_out, 
                            weight.contiguous(), bias,
                            merged_input, expert_block_ids,
                            expert_block_id_idxs, pos, 
                            local_expert_id_indexs, 
                            ranks, 
                            lens, 
                            start_addrs,
                            block_num_tensor.tolist()[0],
                            block_num_tensor.tolist()[1])
        return grad_in, None, None, grad_weight, grad_bias
    
class MixMOELinear2(Function):

    @staticmethod
    def forward(ctx, input_buf, pos, moe_inp_block_infos, weight,
                bias = None):

        mix_moe_instance = MixMOEInstance.instance()

        merged_input, output = mix_moe_instance.mix_moe.second_gemm_forward(input_buf, pos, 
                            moe_inp_block_infos['local_expert_id_indexs'], 
                            moe_inp_block_infos['ranks'],
                            moe_inp_block_infos['lens'],
                            moe_inp_block_infos['start_addrs'],
                            moe_inp_block_infos['total_block_num'],
                            moe_inp_block_infos['remote_block_num'], 
                            weight.transpose(-2, -1).contiguous(), bias.contiguous())

        variables = (merged_input, weight, bias, pos,
                     moe_inp_block_infos['local_expert_id_indexs'], 
                            moe_inp_block_infos['ranks'],
                            moe_inp_block_infos['lens'],
                            moe_inp_block_infos['start_addrs'], 
                            torch.tensor([moe_inp_block_infos['total_block_num'], moe_inp_block_infos['remote_block_num']],),
                            moe_inp_block_infos['expert_block_id_idxs'],
                            moe_inp_block_infos['expert_block_ids'],)

        ctx.save_for_backward(*variables)

        return output
        
        pass
        # return output
    
    @staticmethod
    def backward(ctx, grad_out):

        (merged_input, weight, bias, pos,
                local_expert_id_indexs, 
                ranks,
                lens,
                start_addrs, 
                block_num_tensor,
                expert_block_id_idxs,
                expert_block_ids) = ctx.saved_tensors
        mix_moe_instance = MixMOEInstance.instance()

        torch.cuda.synchronize()
        grad_in, grad_weight, grad_bias = mix_moe_instance.mix_moe.second_gemm_backward(grad_out.contiguous(), merged_input.contiguous(), 
                            expert_block_ids.contiguous(),
                            expert_block_id_idxs.contiguous(),
                            weight.contiguous(), bias, pos,
                            local_expert_id_indexs, 
                            ranks,
                            lens, 
                            start_addrs,
                            block_num_tensor.tolist()[0],
                            block_num_tensor.tolist()[1])
        
        return grad_in, None, None, grad_weight, grad_bias
    


class FMoELinear(nn.Module):
    r"""
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(
        self,
        num_expert: int,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        rank: int = 0,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, inp, pos, expert_ids, ranks, expert_lens, 
                start_addrs, total_block_num, remote_block_num,):
        r"""
        Call MOE function
        """
        # x = MixMOELinear.apply(inp.type_as(self.weight), pos, expert_ids, ranks, expert_lens, 
                # start_addrs, total_block_num, remote_block_num,)
        # return x
        pass

    def extra_repr(self) -> str:
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        # bias is left to zero, similar as megatron

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.9))
        if self.bias is not None:
            torch.nn.init.kaiming_uniform_(self.bias, a=math.sqrt(0.9))


