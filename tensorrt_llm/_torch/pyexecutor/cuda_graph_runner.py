import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..attention_backend.interface import AttentionMetadata
from ..metadata import KVCacheParams
from ..speculative.interface import SpecMetadata
from ..utils import make_weak_ref, set_piecewise_cuda_graph_flag


class graph_capturing_local(threading.local):

    def __init__(self):
        self.is_graph_capturing = False


_local = graph_capturing_local()


def set_graph_capturing(enable: bool):
    _local.is_graph_capturing = enable


def is_graph_capturing() -> bool:
    return _local.is_graph_capturing


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        attn_metadata_overlap_0: Optional[AttentionMetadata] = None,
        attn_metadata_overlap_1: Optional[AttentionMetadata] = None,
    ) -> None:
        """
        Stores a CUDA graph and its associated input buffers.

        Each CUDA graph runner is associated with an AttentionMetadata object
        if flashinfer is being used. Make sure to call attn_metadata.prepare()
        before run()!

        Note that torch.compile w/ mode reduce-overhead supports CUDA graphs
        with memory pool sharing. However, we have our own manager here because,
        at the time of writing this, torch.compile takes way too long to warmup
        graphs compared to doing it manually (not to mention, custom ops from
        e.g. FlashInfer cause graph breaks).
        """
        self.batch_size = batch_size
        self.max_beam_width = max_beam_width
        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        token_per_request = spec_metadata.max_draft_len + 1 if spec_metadata is not None else 1

        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones(
            (batch_size * max_beam_width * token_per_request, ),
            device=device,
            dtype=torch.int32)
        self.position_ids = torch.zeros(
            (1, batch_size * max_beam_width * token_per_request),
            device=device,
            dtype=torch.int32)
        self.mrope_position_deltas = torch.zeros(
            (batch_size,
             1), device=device, dtype=torch.int32) if use_mrope else None

        self.attn_metadata = attn_metadata
        self.spec_metadata = spec_metadata
        self._output = None
        self._graph = None
        self.optional_extra_model_inputs = ["mrope_position_deltas"]
        if batch_size > 1 and attn_metadata_overlap_0 is not None:
            assert attn_metadata_overlap_1 is not None
            offset_split = (batch_size + 1) // 2
            self.input_ids_overlap_0 = torch.ones(
                (offset_split * max_beam_width * token_per_request, ),
                device=device,
                dtype=torch.int32)
            self.input_ids_overlap_1 = torch.ones(
                ((batch_size - offset_split) * max_beam_width *
                 token_per_request, ),
                device=device,
                dtype=torch.int32)
            self.position_ids_overlap_0 = torch.zeros(
                (1, offset_split * max_beam_width * token_per_request),
                device=device,
                dtype=torch.int32)
            self.position_ids_overlap_1 = torch.zeros(
                (1, (batch_size - offset_split) * max_beam_width *
                 token_per_request),
                device=device,
                dtype=torch.int32)
            self.attn_metadata_overlap_0 = attn_metadata_overlap_0
            self.attn_metadata_overlap_1 = attn_metadata_overlap_1
        else:
            self.input_ids_overlap_0 = None
            self.input_ids_overlap_1 = None
            self.position_ids_overlap_0 = None
            self.position_ids_overlap_1 = None
            self.attn_metadata_overlap_0 = None
            self.attn_metadata_overlap_1 = None

    def __del__(self):
        self._graph.reset()

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        self._graph = torch.cuda.CUDAGraph()
        inputs = {
            "attn_metadata": self.attn_metadata,
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "inputs_embeds": None,
            "spec_metadata": self.spec_metadata,
            "mrope_position_deltas": self.mrope_position_deltas,
        }
        if self.batch_size > 1 and self.attn_metadata_overlap_0 is not None:
            assert self.attn_metadata_overlap_1 is not None
            inputs.update({
                "input_ids_overlap_0":
                self.input_ids_overlap_0,
                "input_ids_overlap_1":
                self.input_ids_overlap_1,
                "position_ids_overlap_0":
                self.position_ids_overlap_0,
                "position_ids_overlap_1":
                self.position_ids_overlap_1,
                "attn_metadata_overlap_0":
                self.attn_metadata_overlap_0,
                "attn_metadata_overlap_1":
                self.attn_metadata_overlap_1,
            })

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        set_graph_capturing(True)
        set_piecewise_cuda_graph_flag(False)
        for _ in range(2):
            forward_fn(inputs)
        with torch.cuda.graph(self._graph, pool=pool):
            output = forward_fn(inputs)
        set_graph_capturing(False)
        set_piecewise_cuda_graph_flag(True)
        # Mark weak ref here. The output tensor should be freed properly.
        self._output = make_weak_ref(output)
        return self._graph.pool()

    def needs_capture(self) -> bool:
        return self._output is None

    def run(self, inputs: Dict[str, Any], is_capture=False) -> torch.Tensor:
        assert "input_ids" in inputs
        assert "position_ids" in inputs
        assert "attn_metadata" in inputs

        attn_metadata = inputs["attn_metadata"]
        assert attn_metadata is self.attn_metadata, (
            "attn_metadata does not match the attn_metadata instance that was used to "
            "capture this graph.")

        if "spec_metadata" in inputs:
            spec_metadata = inputs["spec_metadata"]
            assert spec_metadata is self.spec_metadata, (
                "spec_metadata does not match the spec_metadata instance that was used to "
                "capture this graph.")

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        seqlen = input_ids.shape[0]
        self.input_ids[:seqlen].copy_(input_ids)
        self.position_ids[:, :seqlen].copy_(position_ids)
        if "attn_metadata_overlap_0" in inputs and self.attn_metadata_overlap_0 is not None:
            assert inputs[
                "attn_metadata_overlap_0"] is self.attn_metadata_overlap_0, (
                    "attn_metadata_overlap_0 does not match the attn_metadata_overlap_0 instance that was used to "
                    "capture this graph.")
            assert inputs[
                "attn_metadata_overlap_1"] is self.attn_metadata_overlap_1, (
                    "attn_metadata_overlap_1 does not match the attn_metadata_overlap_1 instance that was used to "
                    "capture this graph.")

            if not is_capture:
                assert len(attn_metadata.request_ids) == len(
                    attn_metadata.prompt_lens)
                assert input_ids.shape[0] == position_ids.shape[1]
                split_request = (len(attn_metadata.request_ids) + 1) // 2
                split_context = (attn_metadata.num_contexts + 1) // 2
                assert len(attn_metadata.kv_cache_params.
                           num_cached_tokens_per_seq) % len(
                               attn_metadata.request_ids) == 0
                unit_cached_tokens = len(attn_metadata.kv_cache_params.
                                         num_cached_tokens_per_seq) // len(
                                             attn_metadata.request_ids)
                split_cached_tokens = split_request * unit_cached_tokens
                assert input_ids.shape[0] % len(attn_metadata.request_ids) == 0
                unit_input = input_ids.shape[0] // len(
                    attn_metadata.request_ids)
                split_input = split_request * unit_input
                self.attn_metadata_overlap_0.beam_width = attn_metadata.beam_width
                self.attn_metadata_overlap_1.beam_width = attn_metadata.beam_width
                self.attn_metadata_overlap_0.request_ids = attn_metadata.request_ids[:
                                                                                     split_request]
                self.attn_metadata_overlap_1.request_ids = attn_metadata.request_ids[
                    split_request:]
                self.attn_metadata_overlap_0.prompt_lens = attn_metadata.prompt_lens[:
                                                                                     split_request]
                self.attn_metadata_overlap_1.prompt_lens = attn_metadata.prompt_lens[
                    split_request:]
                self.attn_metadata_overlap_0.num_contexts = split_context
                self.attn_metadata_overlap_1.num_contexts = attn_metadata.num_contexts - split_context
                self.attn_metadata_overlap_0.kv_cache_params = KVCacheParams(
                    use_cache=attn_metadata.kv_cache_params.use_cache,
                    block_ids_per_seq=attn_metadata.kv_cache_params.
                    block_ids_per_seq,
                    num_cached_tokens_per_seq=attn_metadata.kv_cache_params.
                    num_cached_tokens_per_seq[:split_cached_tokens])
                self.attn_metadata_overlap_1.kv_cache_params = KVCacheParams(
                    use_cache=attn_metadata.kv_cache_params.use_cache,
                    block_ids_per_seq=attn_metadata.kv_cache_params.
                    block_ids_per_seq,
                    num_cached_tokens_per_seq=attn_metadata.kv_cache_params.
                    num_cached_tokens_per_seq[split_cached_tokens:])
                self.attn_metadata_overlap_0.kv_cache_manager = attn_metadata.kv_cache_manager
                self.attn_metadata_overlap_1.kv_cache_manager = attn_metadata.kv_cache_manager
                self.attn_metadata_overlap_0.prepare()
                self.attn_metadata_overlap_1.prepare()
                input_ids_overlap_0 = input_ids[:split_input]
                input_ids_overlap_1 = input_ids[split_input:]
                position_ids_overlap_0 = position_ids[:, :split_input]
                position_ids_overlap_1 = position_ids[:, split_input:]
            else:
                input_ids_overlap_0 = inputs["input_ids_overlap_0"]
                input_ids_overlap_1 = inputs["input_ids_overlap_1"]
                position_ids_overlap_0 = inputs["position_ids_overlap_0"]
                position_ids_overlap_1 = inputs["position_ids_overlap_1"]
            seqlen_overlap_0 = input_ids_overlap_0.shape[0]
            seqlen_overlap_1 = input_ids_overlap_1.shape[0]
            self.input_ids_overlap_0[:seqlen_overlap_0].copy_(
                input_ids_overlap_0)
            self.input_ids_overlap_1[:seqlen_overlap_1].copy_(
                input_ids_overlap_1)
            self.position_ids_overlap_0[:, :seqlen_overlap_0].copy_(
                position_ids_overlap_0)
            self.position_ids_overlap_1[:, :seqlen_overlap_1].copy_(
                position_ids_overlap_1)
            inputs.update({
                "input_ids_overlap_0":
                self.input_ids_overlap_0,
                "input_ids_overlap_1":
                self.input_ids_overlap_1,
                "position_ids_overlap_0":
                self.position_ids_overlap_0,
                "position_ids_overlap_1":
                self.position_ids_overlap_1,
                "attn_metadata_overlap_0":
                self.attn_metadata_overlap_0,
                "attn_metadata_overlap_1":
                self.attn_metadata_overlap_1,
            })
        if "mrope_position_deltas" in inputs:
            self.mrope_position_deltas[:self.batch_size].copy_(
                inputs["mrope_position_deltas"])

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
