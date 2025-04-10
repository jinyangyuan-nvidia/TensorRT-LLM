import torch

from .common import ReduceOp, get_rank_world_size, is_ompi

# use trtllm distributed ops to improve TP performance if possible
try:
    from ...distributed import AllReduce, allgather
    from ...modules.linear import AllReduceFusionOp, AllReduceParams, ParallelConfig

    def trtllm_allgather(tensor, dim, all_rank_split_size=None):
        rank, world_size = get_rank_world_size()
        p_config = ParallelConfig(tensor_parallel_size=world_size, tensor_parallel_rank=rank)
        return allgather(tensor, p_config, gather_dim=dim, all_rank_split_size=all_rank_split_size)

    def trtllm_allreduce(tensor, op, all_reduce_params=None):
        rank, world_size = get_rank_world_size()
        assert op == ReduceOp.SUM, "TRT-LLM all reduce only supports SUM op."
        p_config = ParallelConfig(tensor_parallel_size=world_size, tensor_parallel_rank=rank)
        torch_op = AllReduce(p_config)
        return torch_op(tensor, all_reduce_params=all_reduce_params)

    @torch.library.custom_op(
        "dist::fused_allreduce_residual_rmsnorm", mutates_args=(), device_types="cuda"
    )
    def fused_allreduce_residual_rmsnorm(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fusing allreduce, residual (add), and hf_rms_norm together."""
        all_reduce_params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            bias=None,
            residual=residual,
            norm_weight=norm_weight,
            eps=eps,
        )
        return trtllm_allreduce(tensor, ReduceOp.SUM, all_reduce_params=all_reduce_params)

    @fused_allreduce_residual_rmsnorm.register_fake
    def fused_allreduce_residual_rmsnorm_fake(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(tensor), torch.empty_like(tensor)

    TRTLLM_OP_AVAILABLE = True
except ImportError:

    def trtllm_allgather(tensor, dim, all_rank_split_size=None):
        raise ImportError("TRT-LLM is not available.")

    def trtllm_allreduce(tensor, op):
        raise ImportError("TRT-LLM is not available.")

    TRTLLM_OP_AVAILABLE = False


def is_trtllm_op_available():
    # TRT-LLM only work with MPI
    return TRTLLM_OP_AVAILABLE and is_ompi()
