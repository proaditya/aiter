# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Static configuration matching AITER's compiled GPT-OSS 120B megakernel."""

import ctypes
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str = "gpt-oss-120b"
    family: str = "moe"
    num_layers: int = 36
    hidden_size: int = 2880
    padded_hidden_size: int = 2944
    intermediate_size: int = 2880
    vocab_size: int = 201088
    padded_vocab_size: int = 201216
    num_q_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 64
    q_per_kv: int = 8
    qkv_output_size: int = 5120
    gateup_output_size: int = 0
    rope_theta: float = 150000.0
    max_position_embeddings: int = 131072
    sliding_window: int = 128
    has_qk_norm: bool = False
    has_sinks: bool = True
    is_moe: bool = True
    num_experts: int = 128
    num_experts_per_tok: int = 4
    num_xcds: int = 8
    workers_per_xcd: int = 30
    num_kv_chunks: int = 8
    output_per_wg: int = 64
    gateup_opw: int = 128
    page_size: int = 16
    kv_cache_stride: int = 512
    # 13 packed tensors per layer on the vLLM path. The standalone demo packs 14
    # because it also keeps an unread MXFP4 router; K/V are aliased from vLLM.
    weights_per_layer: int = 13
    ptrs_in: int = 26
    ptrs_out: int = 11
    ptrs_extra: int = 1
    counters_per_layer: int = 1648
    counter_tail_ints: int = 624
    slot_qkv_barrier: int = 1168
    timing_slots_per_layer: int = 14
    timing_tail_slots: int = 4
    gemm: dict = field(
        default_factory=lambda: {
            "qkv_output_per_wg": 64,
            "oproj_output_per_wg": 16,
            "w13_output_per_wg": 128,
            "w2_output_per_wg": 64,
            "oproj_kmajor": True,
            "w13_kmajor": True,
            "lm_head_kmajor": True,
        }
    )
    launch_symbol: str = "gpt_oss_120b_launch"
    init_symbol: str = "gpt_oss_120b_init"
    finalize_symbol: str = "gpt_oss_120b_finalize"

    @property
    def extra_launch_args(self) -> tuple[type, ...]:
        return (
            ctypes.c_void_p,  # logits_output
            ctypes.c_void_p,  # timing_buf
            ctypes.c_void_p,  # embed_weight
            ctypes.c_int,  # cur_token_id
            ctypes.c_void_p,  # decode_ctrl
        )

    @property
    def so_path(self) -> str:
        path = os.environ.get("AITERMK_FLEET_PROVIDER") or os.environ.get("FLEET_MK_SO")
        if not path:
            raise RuntimeError(
                "set AITERMK_FLEET_PROVIDER to libaitermk_provider_gpt_oss_gfx950_fleet.so"
            )
        return path

    @property
    def ptrs_per_layer(self) -> int:
        return self.ptrs_in + self.ptrs_out + self.ptrs_extra

    @property
    def rank_counter_ints(self) -> int:
        return self.num_xcds * 16

    @property
    def attn_reduction(self) -> int:
        return self.num_q_heads * self.head_dim


_SPEC = ModelSpec()


def supported_arches() -> tuple[str, ...]:
    return ("GptOssForCausalLM",)


def load_spec(vllm_arch: str = "GptOssForCausalLM") -> ModelSpec:
    if vllm_arch not in supported_arches():
        raise KeyError(f"no Fleet megakernel for {vllm_arch!r}; known: {supported_arches()}")
    return _SPEC


__all__ = ["ModelSpec", "load_spec", "supported_arches"]
