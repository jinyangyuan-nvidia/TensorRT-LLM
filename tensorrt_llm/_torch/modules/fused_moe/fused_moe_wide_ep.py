import os
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe, MoEAlltoallInfo
from tensorrt_llm._utils import logger
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from ...distributed import AllReduce, allgather
from ...expert_statistic import ExpertStatistic
from ...model_config import ModelConfig
from ...utils import AuxStreamType, EventType, Fp4QuantizedTensor
from .deep_ep_utils import buffer_pool, deep_ep_installed
from .interface import MoE
from .moe_load_balancer import get_moe_load_balancer
from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethod,
                           FP8QDQFusedMoEMethod, MoEWeightLoadingMode,
                           NVFP4CutlassFusedMoEMethod,
                           UnquantizedFusedMoEMethod, WInt4AFP8FusedMoEMethod)
from .routing import BaseMoeRoutingMethod


# The type of alltoall method
class AlltoallMethodType(IntEnum):
    # Not available
    NotEnabled = 0
    # MNNVL
    MNNVL = 1
    # DeepEP intranode or internode: no CUDA Graphs support, IBGDA is required by internode
    DeepEP = 2
    # DeepEP low latency: CUDA Graphs are supported, IBGDA is required
    DeepEPLowLatency = 3


class WideEPMoE(MoE):
    """
    Fused Mixture of Experts (MoE) Layer with for wide EP.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.

    Large-scale EP:
    When we have redundant expert, we have more weight slots than `num_experts`, in that case, we separate the concepts of expert and slot.
    Expert is the concept from model's perspective while slot is the concept from model engine's perspective.
    There should be at least `num_experts` slots in the model engine. More than that is OK, in that case, some experts may have multiple replicas.
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
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
    ):

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            aux_stream_dict=aux_stream_dict,
            support_chunking=True,
        )

        assert self.use_dp, "Attention DP should be used with WideEP."
        assert self.parallel_size > 1, "WideEP should only be enabled with parallel_size > 1"
        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.layer_idx = layer_idx

        moe_load_balancer = get_moe_load_balancer()
        self.layer_load_balancer = None
        self.repeat_idx = 0
        self.repeat_count = 1

        moe_load_balancer_config = model_config.moe_load_balancer
        init_expert_size_per_partition = moe_load_balancer_config.num_local_slots if moe_load_balancer_config else self.num_experts // self.ep_size
        self.initial_global_assignments = [
            (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
            self.num_experts for ep_rank in range(self.ep_size)
            for local_slot_id in range(init_expert_size_per_partition)
        ]

        if moe_load_balancer:
            assert moe_load_balancer_config is not None
            top_k = self.routing_method.experts_per_token
            self.expert_size_per_partition = moe_load_balancer_config.num_local_slots
            self.layer_load_balancer = moe_load_balancer.add_layer(
                self.num_experts,
                top_k,
                self.expert_size_per_partition,
                aux_stream=None if aux_stream_dict is None else
                aux_stream_dict[AuxStreamType.MoeBalancer])
            self.repeat_count = self.layer_load_balancer.get_repeat_count()
            loaded_initial_global_assignments = moe_load_balancer_config.get_layer_initial_global_assignments(
                self.layer_idx)
            self.num_slots = moe_load_balancer_config.num_slots
            if loaded_initial_global_assignments is not None:
                assert isinstance(loaded_initial_global_assignments, list)
                assert len(loaded_initial_global_assignments) == self.num_slots
                assert self.num_slots >= self.num_experts
                assert set(loaded_initial_global_assignments) == set(
                    range(self.num_experts))
                self.initial_global_assignments = loaded_initial_global_assignments
            self.layer_load_balancer.set_initial_weight_assignments(
                self.initial_global_assignments)
            logger.info(
                f"MoE load balancer enabled. num_experts = {num_experts}, num_slots = {self.num_slots}, ep_size = {self.ep_size}"
            )
            logger.info(
                f"initial_global_assignments (layer {self.layer_idx}) = {self.initial_global_assignments}"
            )
        else:
            assert num_experts % self.ep_size == 0
            self.expert_size_per_partition = num_experts // self.ep_size
            self.num_slots = num_experts
        if self.layer_load_balancer and not self.layer_load_balancer.is_static_routing(
        ):
            self.allreduce = AllReduce(mapping=model_config.mapping,
                                       strategy=AllReduceStrategy.NCCL)
        else:
            self.allreduce = None

        self.slot_start = self.ep_rank * self.expert_size_per_partition
        self.slot_end = self.slot_start + self.expert_size_per_partition
        self.initial_local_expert_ids = self.initial_global_assignments[
            self.slot_start:self.slot_end]
        assert len(
            self.initial_local_expert_ids) == self.expert_size_per_partition

        self.tune_max_num_tokens = self.compute_tune_max_num_tokens(
            self.num_slots)
        self.has_been_profiled = False

        self.alltoall_method_type = self.select_alltoall_method_type(
            model_config.mapping, routing_method.experts_per_token, dtype,
            model_config.use_cuda_graph)
        logger.info_once(
            f"{self.__class__.__name__} selects alltoall_method_type {self.alltoall_method_type!r}",
            key="alltoall_method_type")
        self.use_postquant_alltoall = False
        if self.enable_alltoall:
            qm = self.quant_config.quant_mode
            self.use_postquant_alltoall = (os.environ.get(
                "TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1")
                                           == "1") and qm.has_nvfp4()
            # TODO: support alltoall without allgather for top_k % 4 != 0
            self.enable_alltoall_without_allgather = (
                os.environ.get("TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER",
                               "1") == "1"
            ) and self.alltoall_method_type == AlltoallMethodType.MNNVL and routing_method.experts_per_token % 4 == 0
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                MnnvlMemory.initialize()
                self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(
                    model_config.mapping)
                if self.enable_alltoall_without_allgather:
                    self.alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
                        model_config.mapping)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                self.deep_ep_buffer = buffer_pool.get_buffer(
                    model_config.mapping)
                self.deep_ep_buffer.reserve(hidden_size, dtype)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                self.deep_ep_max_num_tokens = int(
                    os.environ.get(
                        "TRTLLM_DEEP_EP_TOKEN_LIMIT",
                        str(
                            min(model_config.max_num_tokens,
                                self.moe_max_num_tokens))))
                self.deep_ep_buffer = buffer_pool.get_low_latency_buffer(
                    model_config.mapping)
                self.deep_ep_buffer.reserve(self.deep_ep_max_num_tokens,
                                            hidden_size, self.num_slots)
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {self.alltoall_method_type!r}"
                )

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

        # Debug function for eliminating imbalance during performance analysis.
        self.enable_dummy_allreduce = os.environ.get(
            "TRTLLM_ENABLE_DUMMY_ALLREDUCE", "0") == "1"

    def _check_configs(self):
        assert self._weights_created

        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

        if self.quant_config and self.quant_config.quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if not (self.quant_config.quant_mode.has_nvfp4()
                    | self.quant_config.quant_mode.has_fp8_block_scales()
                    | self.quant_config.quant_mode.has_fp8_qdq()
                    | self.quant_config.quant_mode.
                    is_int4_weight_only_per_group()):
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

    @staticmethod
    def select_alltoall_method_type(mapping: Mapping, top_k: int,
                                    dtype: torch.dtype,
                                    use_cuda_graph: bool) -> AlltoallMethodType:
        all2all_method_type = os.environ.get("TRTLLM_FORCE_ALLTOALL_METHOD")
        if all2all_method_type is not None:
            return AlltoallMethodType[all2all_method_type]

        if not mapping.enable_attention_dp:
            return AlltoallMethodType.NotEnabled

        if mapping.tp_size == 1:
            return AlltoallMethodType.NotEnabled

        if os.environ.get("TRTLLM_MOE_DISABLE_ALLTOALLV", "0") == "1":
            return AlltoallMethodType.NotEnabled

        if mapping.moe_ep_size <= top_k:
            return AlltoallMethodType.NotEnabled

        if MnnvlMemory.supports_mnnvl():
            return AlltoallMethodType.MNNVL

        if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "0") == "1":
            if deep_ep_installed and dtype == torch.bfloat16:
                if use_cuda_graph:
                    # Here we can only choose DeepEPLowLatency since only this method supports CUDA Graphs.
                    return AlltoallMethodType.DeepEPLowLatency
                else:
                    # Here we can choose DeepEP or DeepEPLowLatency if both are available. Now DeepEP is faster.
                    return AlltoallMethodType.DeepEP

        return AlltoallMethodType.NotEnabled

    @property
    def has_w4afp8(self):
        assert self._weights_created
        return self.quant_config and self.quant_config.quant_mode.is_int4_weight_only_per_group(
        )

    @property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return self.alltoall_method_type != AlltoallMethodType.NotEnabled

    def can_use_alltoall(self, all_rank_max_num_tokens):
        # For DeepEPLowLatency, check if tokens exceed the threshold
        if (self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency
                and all_rank_max_num_tokens > self.deep_ep_max_num_tokens):
            return False

        return self.enable_alltoall

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_qdq():
                return FP8QDQFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.is_int4_weight_only_per_group(
            ):
                return WInt4AFP8FusedMoEMethod()
            else:
                raise ValueError(
                    f"Unsupported quantization mode: {self.quant_config.quant_mode}"
                )
        else:
            return UnquantizedFusedMoEMethod()

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True
        self._check_configs()

    def dummy_allreduce(self):
        """
        Debug function for eliminating imbalance during performance analysis.
        Creates a small dummy tensor and performs allreduce to synchronize processes
        and eliminate timing imbalances for more accurate profiling measurements.
        """
        dummy_tensor = torch.zeros(4, dtype=torch.float32, device='cuda')
        dummy_tensor = self.all_reduce(dummy_tensor)
        return dummy_tensor

    def prepare_fused_moe(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        use_all_to_all: bool,
        output_dtype: torch.dtype,
        all_rank_num_tokens: Optional[List[int]] = None,
        all_rank_max_num_tokens: Optional[int] = None,
        is_first_call: bool = True,
        is_last_call: bool = True,
    ) -> torch.Tensor:
        if self.layer_load_balancer and is_first_call:
            self.layer_load_balancer.start_wait_gpu_stage()

        use_deepseek_fp8_block_scale = False
        use_w4_group_scaling = False
        weight_dtype = self.w3_w1_weight.dtype

        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)

        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        if self.apply_router_weight_on_input:
            assert x.dtype != torch.float8_e4m3fn, "Current workaround for apply_router_weight_on_input does not support fp8 input"
            x = x * token_final_scales.to(x.dtype)
            # TODO: remove this once we have correct fusedmoe kernel ready
            if self.alltoall_method_type in (
                    AlltoallMethodType.DeepEP,
                    AlltoallMethodType.DeepEPLowLatency):
                # DeepEP doesn't support token_final_scales is None
                token_final_scales = torch.ones_like(token_final_scales)
            else:
                token_final_scales = None

        if self.layer_load_balancer:
            if is_first_call:
                self.layer_load_balancer.done_wait_gpu_stage()
            if use_all_to_all and self.alltoall_method_type == AlltoallMethodType.MNNVL:
                self.layer_load_balancer.update_local_statistic(
                    token_selected_experts,
                    is_first_stage=is_first_call,
                    is_last_stage=is_last_call)
            else:
                self.layer_load_balancer.update_statistic_with_local_ids(
                    token_selected_experts,
                    is_first_stage=is_first_call,
                    is_last_stage=is_last_call,
                    allreduce=self.allreduce)
            token_selected_slots = self.layer_load_balancer.route(
                token_selected_experts, self.use_dp)
        else:
            token_selected_slots = token_selected_experts

        # If load balancer is disabled, the statistics are collected from expert IDs.
        # If load balancer is enabled, the statistics are collected from expert slot IDs.
        ExpertStatistic.set_layer(self.layer_idx)
        ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)

        use_allgather = not use_all_to_all

        # If alltoall is disabled, we need also disable use_postquant_alltoall
        use_postquant_alltoall = self.use_postquant_alltoall and use_all_to_all

        # Prepare additional information for profiling in case padding is applied when using alltoall.
        # Only the non-alltoall case is considered for profiling in the warmup phase.
        # Therefore, to get the correct tactics during the actual inference, the inputs to the tuner should be the same as when not using alltoall.
        if use_all_to_all:
            tuner_num_tokens = sum(all_rank_num_tokens)
            tuner_top_k = token_selected_slots.shape[1]
        else:
            tuner_num_tokens = None
            tuner_top_k = None

        outputs = {
            key: None
            for key in [
                "alltoall_info", "token_count", "padded", "deep_ep_handle", "deep_ep_hook_dispatch", "deep_ep_recv_expert_count",
                "deep_ep_topk_idx", "deep_ep_topk_weights"
            ]
        }
        if use_all_to_all:
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                if self.enable_dummy_allreduce:
                    self.dummy_allreduce()
                outputs["token_count"] = x.shape[0]
                if is_last_call:
                    loadbalancer_local_statistic_info = self.layer_load_balancer.get_local_statistic_tensor(
                    )
                else:
                    loadbalancer_local_statistic_info = None
                x, token_selected_slots, token_final_scales, gathered_loadbalancer_local_statistic_info, outputs["alltoall_info"] = \
                    self.alltoall_prepare_maybe_dispatch(all_rank_max_num_tokens,
                                                         x,
                                                         token_selected_slots,
                                                         token_final_scales,
                                                         use_postquant_alltoall,
                                                         loadbalancer_local_statistic_info)
                if gathered_loadbalancer_local_statistic_info is not None:
                    gathered_loadbalancer_local_statistic_info = gathered_loadbalancer_local_statistic_info.view(
                        (self.ep_size, self.num_experts))
                    self.layer_load_balancer.update_statistic_with_gathered_statistic(
                        gathered_loadbalancer_local_statistic_info)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                if not use_postquant_alltoall:
                    x, recv_topk_idx, token_final_scales, num_recv_tokens_per_expert_list, outputs["deep_ep_handle"] = \
                        self.deep_ep_buffer.dispatch(x, token_selected_slots, token_final_scales, self.num_slots,
                        self.expert_size_per_partition * self.ep_rank)
                    outputs[
                        "padded"], x, _, token_selected_slots, token_final_scales = self.pad_empty_recv_tensors(
                            x, None, recv_topk_idx, token_final_scales)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                if not use_postquant_alltoall:
                    outputs["deep_ep_topk_idx"] = token_selected_slots
                    outputs["deep_ep_topk_weights"] = token_final_scales
                    assert all_rank_max_num_tokens <= self.deep_ep_max_num_tokens
                    x, outputs["deep_ep_recv_expert_count"], outputs["deep_ep_handle"], outputs["deep_ep_hook_dispatch"] = \
                        self.deep_ep_buffer.low_latency_dispatch(x, outputs["deep_ep_topk_idx"], self.deep_ep_max_num_tokens, self.num_slots)
                    # x shape: [#local experts, EP size * all_rank_max_num_tokens, hidden_size]
                    # recv_expert_count shape: [#local experts]

                    # # Adapter between `torch.ops.trtllm.fused_moe` and DeepEP
                    # # TODO: remove the adapter by changing `torch.ops.trtllm.fused_moe` API
                    # mask = torch.arange(
                    #     x.shape[1], dtype=torch.int32, device=x.device).expand(
                    #         x.shape[0],
                    #         x.shape[1]) < recv_expert_count.unsqueeze(1)
                    # token_selected_slots = torch.where(
                    #     mask,
                    #     torch.arange(
                    #         x.shape[0] * self.ep_rank,
                    #         x.shape[0] * (self.ep_rank + 1),
                    #         dtype=torch.int32,
                    #         device=x.device).unsqueeze(1), self.num_slots)
                    # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                    # # Cheat the fused_moe API with fake top_k=1
                    # token_selected_slots = token_selected_slots.view(
                    #     x.shape[0], 1)
                    # token_final_scales = torch.ones_like(
                    #     token_selected_slots, dtype=token_final_scales.dtype)

        x_sf = None
        x_row = x.shape[0]
        x_col = x.shape[1]
        if self.has_any_quant:
            if self.has_fp8_qdq:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant)
            elif self.has_nvfp4:
                if use_allgather or use_postquant_alltoall:
                    if isinstance(x, Fp4QuantizedTensor):
                        if use_allgather:
                            assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before allgather"
                        x, x_sf = x.fp4_tensor, x.scaling_factor
                        x_row = x.shape[0]
                        # note: we use uint8 to store 2 fp4 values
                        x_col = x.shape[1] * 2
                    else:
                        # for both postquant alltoall and allgather, we need non swizzle layout
                        x_row = x.shape[0]
                        x_col = x.shape[1]
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x, self.fc31_input_scale, self.scaling_vector_size,
                            False, False)
                    x_sf = x_sf.view((x_row, -1))

            elif self.has_deepseek_fp8_block_scales:
                use_deepseek_fp8_block_scale = True
            elif self.has_w4afp8:
                use_w4_group_scaling = True
                weight_dtype = torch.quint4x2
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        if use_allgather:
            # using allgather case.
            if self.enable_dummy_allreduce:
                self.dummy_allreduce()
            x, x_sf, token_selected_slots, token_final_scales = allgather(
                [
                    x,
                    x_sf,
                    token_selected_slots,
                    token_final_scales,
                ],
                self.mapping,
                dim=0,
                sizes=all_rank_num_tokens)
            x_row = x.shape[0]

        if use_postquant_alltoall:
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                x, x_sf = self.alltoall_postquant_dispatch(
                    x, x_sf, outputs["alltoall_info"])
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                if x_sf is not None:
                    # Adapter between `x_sf` and DeepEP
                    # TODO: remove the adapter by adding dtype support to DeepEP
                    x_sf_dtype = x_sf.dtype
                    x_sf = x_sf.view(torch.float32)
                (x, x_sf), recv_topk_idx, token_final_scales, num_recv_tokens_per_expert_list, outputs["deep_ep_handle"] = \
                    self.deep_ep_buffer.dispatch((x, x_sf), token_selected_slots, token_final_scales, self.num_slots,
                    self.expert_size_per_partition * self.ep_rank)
                outputs[
                    "padded"], x, x_sf, token_selected_slots, token_final_scales = self.pad_empty_recv_tensors(
                        x, x_sf, recv_topk_idx, token_final_scales)
                if x_sf is not None:
                    x_sf = x_sf.view(x_sf_dtype)
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                token_num = x_row
                hidden_size = x_col
                assert x_sf is not None and self.has_nvfp4
                assert hidden_size % 32 == 0
                assert x.dtype == torch.uint8 and x_sf.dtype == torch.uint8
                assert x_sf.shape[0] == token_num and x_sf.shape[
                    1] == hidden_size // 16
                assert x.shape[0] == token_num and x.shape[1] == hidden_size // 2

                outputs["deep_ep_topk_idx"] = token_selected_slots
                outputs["deep_ep_topk_weights"] = token_final_scales

                assert all_rank_max_num_tokens <= self.deep_ep_max_num_tokens
                x, x_sf, recv_expert_count, outputs["deep_ep_handle"] = \
                    self.deep_ep_buffer.low_latency_dispatch_fp4(x, x_sf, outputs["deep_ep_topk_idx"], self.deep_ep_max_num_tokens, self.num_slots)
                assert x.dtype == torch.uint8 and x_sf.dtype == torch.uint8
                assert x.dim() == 3 and x_sf.dim() == 3
                assert x.shape[2] == hidden_size // 2 and x_sf.shape[
                    2] == hidden_size // 16

                mask = torch.arange(
                    x.shape[1], dtype=torch.int32, device=x.device).expand(
                        x.shape[0], x.shape[1]) < recv_expert_count.unsqueeze(1)
                token_selected_slots = torch.where(
                    mask,
                    torch.arange(x.shape[0] * self.ep_rank,
                                 x.shape[0] * (self.ep_rank + 1),
                                 dtype=torch.int32,
                                 device=x.device).unsqueeze(1), self.num_slots)
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                x_sf = x_sf.reshape(x_sf.shape[0] * x_sf.shape[1],
                                    x_sf.shape[2])
                token_selected_slots = token_selected_slots.view(x.shape[0], 1)
                token_final_scales = torch.ones_like(
                    token_selected_slots, dtype=token_final_scales.dtype)
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {self.alltoall_method_type!r}"
                )

        outputs.update({
            "x": x,
            "x_sf": x_sf,
            "token_selected_slots": token_selected_slots,
            "token_final_scales": token_final_scales,
            "use_deepseek_fp8_block_scale": use_deepseek_fp8_block_scale,
            "use_w4_group_scaling": use_w4_group_scaling,
            "tuner_num_tokens": tuner_num_tokens,
            "tuner_top_k": tuner_top_k,
            "weight_dtype": weight_dtype,
            "use_all_to_all": use_all_to_all,
            "output_dtype": output_dtype,
            "all_rank_num_tokens": all_rank_num_tokens,
            "is_first_call": is_first_call,
            "is_last_call": is_last_call,
        })

        return outputs

    def run_fused_moe(self, fused_moe_params: Dict[str, Any]):
        if fused_moe_params["deep_ep_hook_dispatch"] is not None:
            fused_moe_params["deep_ep_hook_dispatch"]()
        x = fused_moe_params["x"]
        token_selected_slots = fused_moe_params["token_selected_slots"]
        token_final_scales = fused_moe_params["token_final_scales"]
        if fused_moe_params["deep_ep_recv_expert_count"] is not None:
            # Adapter between `torch.ops.trtllm.fused_moe` and DeepEP
            # TODO: remove the adapter by changing `torch.ops.trtllm.fused_moe` API
            mask = torch.arange(
                x.shape[1], dtype=torch.int32, device=x.device).expand(
                    x.shape[0],
                    x.shape[1]) < fused_moe_params["deep_ep_recv_expert_count"].unsqueeze(1)
            token_selected_slots = torch.where(
                mask,
                torch.arange(
                    x.shape[0] * self.ep_rank,
                    x.shape[0] * (self.ep_rank + 1),
                    dtype=torch.int32,
                    device=x.device).unsqueeze(1), self.num_slots)
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            # Cheat the fused_moe API with fake top_k=1
            token_selected_slots = token_selected_slots.view(
                x.shape[0], 1)
            token_final_scales = torch.ones_like(
                token_selected_slots, dtype=token_final_scales.dtype)
        final_hidden_states = torch.ops.trtllm.fused_moe(
            x,
            token_selected_slots,
            token_final_scales,
            self.w3_w1_weight.view(fused_moe_params["weight_dtype"]),
            None,  # w3_w1_bias
            self.w2_weight.view(fused_moe_params["weight_dtype"]),
            None,  # w2_bias
            fused_moe_params["output_dtype"],
            quant_scales=self.quant_scales,
            input_sf=fused_moe_params["x_sf"],
            swizzled_input_sf=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=fused_moe_params["use_all_to_all"],
            use_deepseek_fp8_block_scale=fused_moe_params[
                "use_deepseek_fp8_block_scale"],
            use_w4_group_scaling=fused_moe_params["use_w4_group_scaling"],
            min_latency_mode=False,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=fused_moe_params["tuner_num_tokens"],
            tuner_top_k=fused_moe_params["tuner_top_k"],
        )

        # Only in cutlass_min_latency_mode, the output is a list of tensors.
        # Otherwise, the output should be unpacked as a single tensor.
        final_hidden_states = final_hidden_states[0]

        return final_hidden_states

    def finalize_fused_moe(
        self,
        inputs: torch.Tensor,
        fused_moe_params: Dict[str, Any],
    ) -> torch.Tensor:
        if self.layer_load_balancer and fused_moe_params["is_last_call"]:
            self.layer_load_balancer.start_set_cpu_stage()

        if self.enable_dummy_allreduce:
            self.dummy_allreduce()
        combine_hook = None
        if fused_moe_params["use_all_to_all"]:
            if self.alltoall_method_type == AlltoallMethodType.MNNVL:
                outputs = self.alltoall_combine(
                    inputs, fused_moe_params["alltoall_info"],
                    fused_moe_params["token_count"])
            elif self.alltoall_method_type == AlltoallMethodType.DeepEP:
                inputs = self.unpad_tensors(fused_moe_params["padded"], inputs)
                outputs = self.deep_ep_buffer.combine(
                    inputs, fused_moe_params["deep_ep_handle"])
            elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                num_tokens_per_expert_for_fused_moe = self.ep_size * self.deep_ep_max_num_tokens
                inputs = inputs.view(self.expert_size_per_partition,
                                     num_tokens_per_expert_for_fused_moe,
                                     self.hidden_size)
                outputs, combine_hook = self.deep_ep_buffer.low_latency_combine(
                    inputs, fused_moe_params["deep_ep_topk_idx"],
                    fused_moe_params["deep_ep_topk_weights"],
                    fused_moe_params["deep_ep_handle"])
            else:
                raise NotImplementedError(
                    f"Not available alltoall method type: {self.alltoall_method_type!r}"
                )
        else:
            outputs = self.reducescatter_or_allreduce(
                inputs, fused_moe_params["all_rank_num_tokens"])

        if self.layer_load_balancer and fused_moe_params["is_last_call"]:
            self.layer_load_balancer.done_set_cpu_stage()

        return outputs, combine_hook

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        do_finalize: bool = True,  # used by other MoE backends
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        all_rank_max_num_tokens: Optional[int] = None,
        idx_overlap: Optional[int] = None,
        idx_mode: Optional[int] = None,
        fused_moe_params: Optional[Dict[str, Any]] = None,
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert do_finalize, "WideEPMoE does not support do_finalize=False"
        assert all_rank_num_tokens is not None, "all_rank_num_tokens should be provided because attention DP is enabled"
        if all_rank_max_num_tokens is None:
            all_rank_max_num_tokens = max(all_rank_num_tokens)
        if output_dtype is None:
            assert not isinstance(x, Fp4QuantizedTensor)
            output_dtype = x.dtype

        num_chunks = self.compute_num_chunks(x, all_rank_num_tokens)
        use_all_to_all = self.can_use_alltoall(all_rank_max_num_tokens)
        if idx_overlap is not None:
            assert num_chunks == 1
        if num_chunks == 1:
            is_first_call = self.repeat_idx == 0 and (idx_overlap is None or idx_overlap == 0)
            is_last_call = self.repeat_idx == self.repeat_count - 1 and (idx_overlap is None or idx_overlap == 1)
            if idx_overlap is None or idx_mode is None:
                fused_moe_params = self.prepare_fused_moe(
                    x,
                    router_logits,
                    use_all_to_all,
                    output_dtype,
                    all_rank_num_tokens=all_rank_num_tokens,
                    all_rank_max_num_tokens=all_rank_max_num_tokens,
                    is_first_call=is_first_call,
                    is_last_call=is_last_call)
                outputs = self.run_fused_moe(fused_moe_params)
                outputs, hook_combine = self.finalize_fused_moe(outputs, fused_moe_params)
                if hook_combine is not None:
                    hook_combine()
            else:
                assert self.repeat_count == 1
                if idx_mode == 0:
                    assert fused_moe_params is None
                    assert outputs is None
                    fused_moe_params = self.prepare_fused_moe(
                        x,
                        router_logits,
                        use_all_to_all,
                        output_dtype,
                        all_rank_num_tokens=all_rank_num_tokens,
                        all_rank_max_num_tokens=all_rank_max_num_tokens,
                        is_first_call=is_first_call,
                        is_last_call=is_last_call)
                    return fused_moe_params
                elif idx_mode == 1:
                    assert fused_moe_params is not None
                    assert outputs is None
                    outputs = self.run_fused_moe(fused_moe_params)
                    return outputs
                elif idx_mode == 2:
                    assert fused_moe_params is not None
                    assert outputs is not None
                    outputs, hook_combine = self.finalize_fused_moe(outputs, fused_moe_params)
                    return outputs, hook_combine
                else:
                    raise ValueError(f"Invalid idx_mode: {idx_mode}")
        else:
            all_rank_chunk_size_list = [
                self.compute_chunk_size_list(val, num_chunks)
                for val in all_rank_num_tokens
            ]
            all_rank_max_num_tokens_list = self.compute_chunk_size_list(
                all_rank_max_num_tokens, num_chunks)
            all_rank_nonzero_tokens_list = [[
                max(val[idx_chunk], 1) for val in all_rank_chunk_size_list
            ] for idx_chunk in range(num_chunks)]
            chunk_size_list = all_rank_chunk_size_list[self.rank]

            x_list = self.split_tensor_maybe_with_dummy(x, chunk_size_list)
            router_logits_list = self.split_tensor_maybe_with_dummy(
                router_logits, chunk_size_list)

            self.event_dict[EventType.Main].record()
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.Main].wait()

            def _prepare_fused_moe(idx):
                is_first_call = idx == 0 and self.repeat_idx == 0 and (idx_overlap is None or idx_overlap == 0)
                is_last_call = idx == num_chunks - 1 and self.repeat_idx == self.repeat_count - 1 and (idx_overlap is None or idx_overlap == 1)
                return self.prepare_fused_moe(
                    x_list[idx],
                    router_logits_list[idx],
                    use_all_to_all,
                    output_dtype,
                    all_rank_num_tokens=all_rank_nonzero_tokens_list[idx],
                    all_rank_max_num_tokens=all_rank_max_num_tokens_list[idx],
                    is_first_call=is_first_call,
                    is_last_call=is_last_call)

            # wait_aux, run[0], finalize[0], prepare[2], record_main,                                                         wait_aux, hook_combine[0], run[2], finalize[2], prepare[4], record_main,                                                              wait_aux, hook_combine[2], run[4], finalize[4], record_main,                             hook_combine[4]
            #                                                         wait_main, run[1], finalize[1], prepare[3], record_aux,                                                                          wait_main, hook_combine[1], run[3], finalize[3], record_aux,                                                              wait_main, hook_combine[3],

            outputs_list, hook_combine_list = [], []
            if num_chunks == 2:
                fused_moe_params_main = _prepare_fused_moe(0)
                if fused_moe_params_main["deep_ep_hook_dispatch"] is not None:
                    fused_moe_params_main["deep_ep_hook_dispatch"]()
                    fused_moe_params_main["deep_ep_hook_dispatch"] = None
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    fused_moe_params_aux = _prepare_fused_moe(1)
                    self.event_dict[EventType.MoeChunkingOverlap].record()
                self.event_dict[EventType.MoeChunkingOverlap].wait()
                outputs_main = self.run_fused_moe(fused_moe_params_main)
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    if fused_moe_params_aux["deep_ep_hook_dispatch"] is not None:
                        fused_moe_params_aux["deep_ep_hook_dispatch"]()
                        fused_moe_params_aux["deep_ep_hook_dispatch"] = None
                    self.event_dict[EventType.MoeChunkingOverlap].record()
                self.event_dict[EventType.MoeChunkingOverlap].wait()
                outputs_main, hook_combine_main = self.finalize_fused_moe(
                    outputs_main, fused_moe_params_main)
                outputs_list.append(outputs_main)
                hook_combine_list.append(hook_combine_main)
                outputs_aux = self.run_fused_moe(fused_moe_params_aux)
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.MoeChunkingOverlap].record()
                self.event_dict[EventType.MoeChunkingOverlap].wait()
                if hook_combine_main is not None:
                    hook_combine_main()
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    outputs_aux, hook_combine_aux = self.finalize_fused_moe(
                        outputs_aux, fused_moe_params_aux)
                    outputs_list.append(outputs_aux)
                    hook_combine_list.append(hook_combine_aux)
                    if hook_combine_aux is not None:
                        hook_combine_aux()
            else:
                fused_moe_params_main = _prepare_fused_moe(0)
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    fused_moe_params_aux = _prepare_fused_moe(1)
                    self.event_dict[EventType.MoeChunkingOverlap].record()
                for idx_chunk in range(num_chunks):
                    if idx_chunk % 2 == 0:
                        self.event_dict[EventType.MoeChunkingOverlap].wait()
                        if idx_chunk - 2 >= 0 and hook_combine_list[idx_chunk - 2] is not None:
                            hook_combine_list[idx_chunk - 2]()
                        outputs_main = self.run_fused_moe(fused_moe_params_main)
                        outputs_main, hook_combine_main = self.finalize_fused_moe(
                            outputs_main, fused_moe_params_main)
                        outputs_list.append(outputs_main)
                        hook_combine_list.append(hook_combine_main)
                        if idx_chunk + 2 < num_chunks:
                            fused_moe_params_main = _prepare_fused_moe(idx_chunk + 2)
                        self.event_dict[EventType.Main].record()
                    else:
                        with torch.cuda.stream(self.aux_stream):
                            self.event_dict[EventType.Main].wait()
                            if idx_chunk - 2 >= 0 and hook_combine_list[idx_chunk - 2] is not None:
                                hook_combine_list[idx_chunk - 2]()
                            outputs_aux = self.run_fused_moe(fused_moe_params_aux)
                            outputs_aux, hook_combine_aux = self.finalize_fused_moe(
                                outputs_aux, fused_moe_params_aux)
                            outputs_list.append(outputs_aux)
                            hook_combine_list.append(hook_combine_aux)
                            if idx_chunk + 2 < num_chunks:
                                fused_moe_params_aux = _prepare_fused_moe(idx_chunk + 2)
                            self.event_dict[EventType.MoeChunkingOverlap].record()
                if num_chunks % 2 == 0:
                    self.event_dict[EventType.MoeChunkingOverlap].wait()
                    if hook_combine_list[-2] is not None:
                        hook_combine_list[-2]()
                    with torch.cuda.stream(self.aux_stream):
                        if hook_combine_list[-1] is not None:
                            hook_combine_list[-1]()
                else:
                    with torch.cuda.stream(self.aux_stream):
                        self.event_dict[EventType.Main].wait()
                        if hook_combine_list[-2] is not None:
                            hook_combine_list[-2]()
                    if hook_combine_list[-1] is not None:
                        hook_combine_list[-1]()
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.MoeChunkingOverlap].record()
            self.event_dict[EventType.MoeChunkingOverlap].wait()
            outputs = torch.cat(outputs_list)
            # Dummy tensors may be used and need to be discarded when attention DP is enabled.
            outputs = outputs[:all_rank_num_tokens[self.rank]]
        self.repeat_idx = 0 if self.repeat_idx == self.repeat_count - 1 else self.repeat_idx + 1
        return outputs

    def alltoall_prepare_maybe_dispatch(
            self, all_rank_max_num_tokens: int, x: torch.Tensor,
            token_selected_slots: torch.Tensor,
            token_final_scales: torch.Tensor, use_postquant_alltoall: bool,
            local_statistic_tensor: Optional[torch.Tensor]):
        top_k = self.routing_method.experts_per_token

        if self.enable_alltoall_without_allgather:
            alltoall_info, token_selected_slots, token_final_scales, gathered_local_statistic_tensor = MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                token_selected_slots, token_final_scales,
                local_statistic_tensor, self.alltoall_prepare_workspace,
                all_rank_max_num_tokens, self.ep_rank, self.ep_size,
                self.num_experts, self.num_slots, top_k)
        else:
            if all_rank_max_num_tokens > token_selected_slots.shape[0]:
                token_selected_slots = torch.nn.functional.pad(
                    token_selected_slots,
                    (0, 0, 0,
                     all_rank_max_num_tokens - token_selected_slots.shape[0]),
                    'constant', self.num_slots)
            if token_final_scales is not None and all_rank_max_num_tokens > token_final_scales.shape[
                    0]:
                token_final_scales = torch.nn.functional.pad(
                    token_final_scales,
                    (0, 0, 0,
                     all_rank_max_num_tokens - token_final_scales.shape[0]))
            gathered_token_selected_slots, gathered_token_final_scales, gathered_local_statistic_tensor = allgather(
                [
                    token_selected_slots, token_final_scales,
                    local_statistic_tensor
                ],
                self.mapping,
                dim=0)
            gathered_token_selected_slots = torch.flatten(
                gathered_token_selected_slots.contiguous(),
                start_dim=0,
                end_dim=-2)
            if gathered_token_final_scales is not None:
                gathered_token_final_scales = torch.flatten(
                    gathered_token_final_scales.contiguous(),
                    start_dim=0,
                    end_dim=-2)
            gathered_target_rank_ids = MnnvlMoe.compute_target_rank_id(
                gathered_token_selected_slots, self.num_slots, self.ep_size)
            alltoall_info, token_selected_slots, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv_prepare(
                gathered_target_rank_ids, None, gathered_token_selected_slots,
                gathered_token_final_scales, all_rank_max_num_tokens,
                self.num_slots, top_k, self.ep_rank, self.ep_size)

        if not use_postquant_alltoall:
            assert not isinstance(
                x, Fp4QuantizedTensor
            ), "pre-quant alltoall doesn't support fp4 tensor"
            x = MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info,
                                             self.alltoall_workspace,
                                             self.ep_rank, self.ep_size)

        return x, token_selected_slots, token_final_scales, gathered_local_statistic_tensor, alltoall_info

    def alltoall_postquant_dispatch(self, x: torch.Tensor, x_sf: torch.Tensor,
                                    alltoall_info: MoEAlltoallInfo):
        x = MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info,
                                         self.alltoall_workspace, self.ep_rank,
                                         self.ep_size)

        if x_sf is not None:
            x_sf = MnnvlMoe.mnnvl_moe_alltoallv(x_sf, alltoall_info,
                                                self.alltoall_workspace,
                                                self.ep_rank, self.ep_size)

        return x, x_sf

    def alltoall_combine(self, final_hidden_states: torch.Tensor,
                         alltoall_info: MoEAlltoallInfo, token_count: int):
        top_k = self.routing_method.experts_per_token
        if isinstance(final_hidden_states, list):
            final_hidden_states = final_hidden_states[0]
        final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
            final_hidden_states,
            alltoall_info,
            self.alltoall_workspace,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            top_k=top_k,
            token_count=token_count)

        return final_hidden_states

    def pad_empty_recv_tensors(
        self, x: torch.Tensor, x_sf: Optional[torch.Tensor],
        recv_topk_idx: torch.Tensor, token_final_scales: torch.Tensor
    ) -> Tuple[bool, torch.Tensor, Optional[torch.Tensor], torch.Tensor,
               torch.Tensor]:
        """
        Pad the output of DeepEP `dispatch` if the output length is zero.
        We can remove the adapter if both `fused_moe` op and `swizzle_sf`
        accept zero-length inputs.
        """
        if x.shape[0] == 0:
            padded = True
            x = torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)
            if x_sf is not None:
                x_sf = torch.zeros((1, x_sf.shape[1]),
                                   dtype=x_sf.dtype,
                                   device=x_sf.device)
            recv_topk_idx = torch.full((1, recv_topk_idx.shape[1]),
                                       self.num_slots,
                                       dtype=recv_topk_idx.dtype,
                                       device=recv_topk_idx.device)
            token_final_scales = torch.ones((1, token_final_scales.shape[1]),
                                            dtype=token_final_scales.dtype,
                                            device=token_final_scales.device)
        else:
            padded = False
        return padded, x, x_sf, recv_topk_idx, token_final_scales

    def unpad_tensors(self, padded: bool,
                      final_hidden_states: torch.Tensor) -> torch.Tensor:
        if padded:
            final_hidden_states = final_hidden_states[:0]
        return final_hidden_states

    def register_parameter_weight_slot_fn(self, weight_name: str,
                                          local_slot_id: int):
        assert hasattr(
            self,
            weight_name), f"FusedMoE doesn't has weight attr: {weight_name}"
        weight_tensor = getattr(self, weight_name).data[local_slot_id]
        self.layer_load_balancer.register_weight_slot(local_slot_id,
                                                      weight_name,
                                                      weight_tensor)

    def register_to_fix_weight_fn(self, weight_name: str):
        assert hasattr(
            self,
            weight_name), f"FusedMoE doesn't has weight attr: {weight_name}"
        param = getattr(self, weight_name)
        weight_tensor = param.detach()
        assert isinstance(
            weight_tensor,
            torch.Tensor), f'weight {weight_name} should be a tensor'
        assert weight_tensor.is_contiguous(
        ), f'weight {weight_name} should be a is_contiguous, shape={weight_tensor.shape}, strides={weight_tensor.is_contiguous()}'
        assert weight_tensor.numel() * weight_tensor.element_size() == weight_tensor.untyped_storage().size(),\
            f'weight {weight_name} shape={weight_tensor.shape} storage_size = {weight_tensor.untyped_storage().size()}, numel={weight_tensor.numel()}, eltsize={weight_tensor.element_size()}, dtype={weight_tensor.dtype}'
        self.layer_load_balancer.make_tensor_host_accessible(weight_tensor)
        param.data = weight_tensor

    def register_all_parameter_slot_and_to_fix_weight_fns(
            self, weight_and_tensor_dict: Dict[str, torch.Tensor]):
        """
        weight_and_tensor_dict: key is the name of the weight, value is the tensor of loaded shared tensor shard.
            E.g. if num_experts=256 and 4 GPUs per node, then each rank need to load 256 / 4 = 64 expert weights for host sharing.
            By this way, host_tensor_sharer can share the weights and each rank has access to all 256 experts.
        """
        for local_slot_id, expert_id in enumerate(
                self.initial_local_expert_ids):
            for weight_name in weight_and_tensor_dict:
                self.layer_load_balancer.add_register_weight_fn(
                    self.register_parameter_weight_slot_fn,
                    (weight_name, local_slot_id))
        for weight_name in weight_and_tensor_dict:
            self.layer_load_balancer.add_to_migrate_weight_fn(
                self.register_to_fix_weight_fn, (weight_name, ))

        local_shared_load_expert_ids = self.layer_load_balancer.get_load_expert_ids(
        )
        for expert_id in range(self.num_experts):
            for weight_name, weight_tensor in weight_and_tensor_dict.items():
                if expert_id in local_shared_load_expert_ids:
                    local_slot_id = local_shared_load_expert_ids.index(
                        expert_id)
                    self.layer_load_balancer.host_tensor_sharer.share_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor[local_slot_id])
                else:
                    self.layer_load_balancer.host_tensor_sharer.pre_register_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor.dtype,
                        weight_tensor[0].shape)

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)
