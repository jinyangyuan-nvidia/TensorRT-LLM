from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import torch
from torch import nn

from ...distributed.ops import reducescatter
from ...model_config import ModelConfig
from ...utils import AuxStreamType, EventType
from .routing import BaseMoeRoutingMethod


class MoEWeightLoadingMode(Enum):
    VANILLA = 0
    FUSED_GATE_UP_PROJ = 1


class MoE(nn.Module):
    """
    Fused Mixture of Experts (MoE) Layer interface.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        bias: bool = False,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        support_chunking: bool = False,
    ):
        from ...distributed import AllReduce

        super().__init__()
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_loading_mode = weight_loading_mode
        self.bias = bias
        self.dtype = dtype
        self.reduce_results = reduce_results
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        # could be modified later
        self.quant_config = model_config.quant_config

        self.cluster_rank = model_config.mapping.moe_cluster_rank
        self.cluster_size = model_config.mapping.moe_cluster_size
        self.smart_router = True if self.cluster_size > 1 else False

        self.rank = model_config.mapping.rank

        self.tp_rank = model_config.mapping.moe_tp_rank
        self.tp_size = model_config.mapping.moe_tp_size

        self.ep_size = model_config.mapping.moe_ep_size
        self.ep_rank = model_config.mapping.moe_ep_rank

        self.moe_backend = model_config.moe_backend
        self.use_dp = model_config.mapping.enable_attention_dp

        # All ranks participate in allreduce regardless of EP/TP combination
        self.mapping = model_config.mapping
        self.parallel_size = self.mapping.tp_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.all_reduce = AllReduce(mapping=self.mapping,
                                    strategy=model_config.allreduce_strategy,
                                    dtype=self.dtype)

        max_num_tokens = model_config.max_num_tokens
        # The maximum number of tokens in MoE are multiplied by DP size when attention DP is enabled
        if self.use_dp:
            max_num_tokens *= self.parallel_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens or max_num_tokens
        self.moe_enable_overlap = model_config.moe_enable_overlap
        # The auxiliary CUDA stream and CUDA events are only used when MoE chunking is applied
        if self.moe_max_num_tokens < max_num_tokens or self.moe_enable_overlap:
            assert support_chunking, "MoE chunking is not supported"
            self.aux_stream = aux_stream_dict[
                AuxStreamType.
                MoeChunkingOverlap] if aux_stream_dict is not None else torch.cuda.Stream(
                )
            self.event_dict = {
                key: torch.cuda.Event()
                for key in [EventType.Main, EventType.MoeChunkingOverlap]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

    @abstractmethod
    def create_weights(self):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, weights: List[Dict]):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def has_any_quant(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True)

    # The following three properties are common enough to warrant inclusion in the interface.
    @property
    def has_fp8_qdq(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_qdq(
        )

    @property
    def has_deepseek_fp8_block_scales(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_block_scales(
        )

    @property
    def has_nvfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4(
        )

    @property
    def has_w4a8_mxfp4_fp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8(
        )

    @property
    def has_w4a8_mxfp4_mxfp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8(
        )

    @property
    def has_w4a16_mxfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a16_mxfp4(
        )

    @property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return False

    def compute_tune_max_num_tokens(self, num_slots: int) -> int:
        # The profiler converges on the same best tactic when the number of tokens is large enough.
        # To avoid long profiling time, the max number of tokens used in the profiling is capped to
        # around 16k tokens per expert, which is well into the compute bound domain.
        tune_max_num_tokens = min(
            self.moe_max_num_tokens,
            16384 * num_slots // self.routing_method.get_experts_per_token(),
        )
        return tune_max_num_tokens

    def compute_num_chunks(
        self,
        x: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> int:
        if self.use_dp and self.parallel_size > 1:
            assert all_rank_num_tokens is not None
            num_rows = sum(all_rank_num_tokens)
        else:
            num_rows = x.shape[0]
        # If num_rows is larger than max_chunk_size, we need to split the input into multiple chunks
        num_chunks = (num_rows + self.moe_max_num_tokens -
                      1) // self.moe_max_num_tokens
        # If moe_enable_overlap is true and num_rows is splitable, we need to split the input into at least 2 chunks
        if self.moe_enable_overlap:
            cond_dp_on = self.use_dp and num_rows > self.parallel_size
            cond_dp_off = not self.use_dp and num_rows > 1
            if cond_dp_on or cond_dp_off:
                num_chunks = max(num_chunks, 2)
        return num_chunks

    @staticmethod
    def compute_chunk_size_list(num_tokens: int, num_chunks: int) -> List[int]:
        val_div = num_tokens // num_chunks
        val_mod = num_tokens % num_chunks
        chunk_size_list = [val_div + 1] * val_mod + [val_div
                                                     ] * (num_chunks - val_mod)
        return chunk_size_list

    @staticmethod
    def split_tensor_maybe_with_dummy(
            x: torch.Tensor, chunk_size_list: List[int]) -> List[torch.Tensor]:
        x_list = x.split(chunk_size_list)
        # Avoid potential bugs by replacing empty tensors with dummy tensors
        x_list = [
            val if chunk_size > 0 else x[-1:]
            for val, chunk_size in zip(x_list, chunk_size_list)
        ]
        return x_list

    def reducescatter_or_allreduce(
        self,
        inputs: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Common helper for TP and EP in subclasses of the MoE module.
        """
        if self.parallel_size > 1:
            if self.use_dp:
                outputs = reducescatter(inputs,
                                        self.mapping,
                                        dim=0,
                                        sizes=all_rank_num_tokens)
            elif self.reduce_results:
                outputs = self.all_reduce(inputs)
        else:
            outputs = inputs
        return outputs
