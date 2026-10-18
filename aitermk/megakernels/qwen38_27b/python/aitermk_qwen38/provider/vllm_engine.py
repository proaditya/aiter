# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One speculative verify of the megakernel over vLLM's caches (one sequence).

``VllmMonoRound`` holds the converted MXFP4 weights, one chain build (``m8`` by default), its pointer table and
the KV mirror. ``bind`` takes vLLM's per-layer caches once they exist; ``verify`` launches the build over 8
consecutive positions of a request, reading its recurrent state slot in place and its conv window and KV;
``commit`` settles the accepted prefix back into vLLM's caches the way vLLM's own speculative step leaves them
(``vllm_state_kernels``), so vLLM's path can take any later step. vLLM samples from the launch's logits.
"""

import logging

import torch

from aitermk_qwen38.provider import dims as d
from aitermk_qwen38.provider import vllm_state_kernels as vk
from aitermk_qwen38.provider.binding import KernelBinding, SharedBuffers
from aitermk_qwen38.provider.kernels.build import VARIANTS, build
from aitermk_qwen38.provider.kernels.hip import Megakernel

logger = logging.getLogger("aitermk_qwen38")

# the residual streams DFlash 2 drafts from: after target layers 5, 19, 33, 47, 61
DFLASH2_AUX_LAYERS = (5, 19, 33, 47, 61)


class _Inputs:
    """A launch's device inputs, the fields ``vllm_state_kernels.launch_begin`` reads."""

    def __init__(self, rows: int, device) -> None:
        def scalar(v=0):
            return torch.full((1,), v, dtype=torch.int32, device=device)

        self.token_ids = torch.zeros(rows, dtype=torch.int64, device=device)
        self.base = scalar()
        # each DeltaNet layer's recurrent state slot the launch read (resolved on the device)
        self.state_slot = torch.zeros(len(d.DELTA_LAYERS), dtype=torch.int32, device=device)
        # the mirror's name for the sequence: one sequence, renamed by ``invalidate_mirror``
        self.request = scalar()


def _ptrs(tensors, device) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device=device)


def _uniform(name: str, values) -> tuple:
    values = [tuple(v) for v in values]
    if any(v != values[0] for v in values):
        raise ValueError(f"{name}: the layers' caches differ in layout: {sorted(set(values))}")
    return values[0]


class VllmMonoRound:
    def __init__(self, weights, max_model_len: int, *, variant: str = "m8", device=None) -> None:
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        dev = self.device
        v = VARIANTS[variant]
        self.rows = v.rows
        max_pages = (max_model_len + self.rows + d.PAGE_TOKENS - 1) // d.PAGE_TOKENS
        self.shared = SharedBuffers(self.rows, max_pages, dev)
        self.kernel = Megakernel(build(variant))
        self.binding = KernelBinding(self.kernel.rows, weights, self.shared, dev)
        self.inp = _Inputs(self.rows, dev)
        self.delta_rows = torch.tensor(d.DELTA_LAYERS, dtype=torch.int64, device=dev)
        self.state_col = d.DELTA_SLOTS.index("initial_state")
        self.mirror_slot = torch.full((1,), -1, dtype=torch.int32, device=dev)
        self.mirror_len = torch.zeros(1, dtype=torch.int32, device=dev)
        self.import_start = torch.zeros(1, dtype=torch.int32, device=dev)
        self.import_end = torch.zeros(1, dtype=torch.int32, device=dev)
        self.bound = False

    def bind(self, *, k_views, v_views, conv_states, ssm_states) -> None:
        """vLLM's caches: per attention layer the K and V views ``[blocks, block_size, 4, 256]`` bf16; per
        DeltaNet layer the conv cache ``[slots, window, 10240]`` (or ``[slots, 10240, window]``) bf16 and the
        recurrent cache ``[slots, 48, 128, 128]`` fp32."""
        dev = self.device
        if len(k_views) != len(d.ATTENTION_LAYERS) or len(v_views) != len(d.ATTENTION_LAYERS):
            raise ValueError(f"{len(k_views)} K / {len(v_views)} V caches for {len(d.ATTENTION_LAYERS)} attention layers")
        for t in list(k_views) + list(v_views):
            if t.dtype != torch.bfloat16 or t.dim() != 4 or tuple(t.shape[2:]) != (d.KV_HEADS, d.HEAD_DIM) or t.stride(3) != 1:
                raise ValueError(f"KV cache view {tuple(t.shape)} {t.stride()} {t.dtype}: need [blocks, block_size, "
                                 f"{d.KV_HEADS}, {d.HEAD_DIM}] bf16 with a contiguous head (--kv-cache-dtype auto)")
        self.block_size = _uniform("KV block size", [(t.shape[1],) for t in list(k_views) + list(v_views)])[0]
        self.kv_strides = _uniform("KV strides", [t.stride()[:3] for t in list(k_views) + list(v_views)])
        self.k_ptrs, self.v_ptrs = _ptrs(k_views, dev), _ptrs(v_views, dev)

        n = len(d.DELTA_LAYERS)
        if len(conv_states) != n or len(ssm_states) != n:
            raise ValueError(f"{len(conv_states)} conv / {len(ssm_states)} recurrent caches for {n} DeltaNet layers")
        want = (d.V_HEADS, d.GDN_DIM, d.GDN_DIM)
        for s in ssm_states:
            if s.dtype != torch.float32 or tuple(s.shape[1:]) != want:
                raise ValueError(f"recurrent cache {tuple(s.shape)} {s.dtype}: need [slots, *{want}] fp32 "
                                 "(--mamba-ssm-cache-dtype float32)")
            if s[0].stride() != (d.GDN_DIM * d.GDN_DIM, d.GDN_DIM, 1):
                raise ValueError("recurrent cache slot must be contiguous [48][128 v][128 k]")
        self.ssm_s_slot = _uniform("recurrent slot stride", [(s.stride(0),) for s in ssm_states])[0]
        self.ssm_ptrs = _ptrs(ssm_states, dev)
        conv_strides = []
        windows = []
        for c in conv_states:
            if c.dtype != torch.bfloat16 or c.dim() != 3 or d.CONV_DIM not in c.shape[1:]:
                raise ValueError(f"conv cache {tuple(c.shape)} {c.dtype}: need [slots, window, {d.CONV_DIM}] bf16")
            ch = 2 if c.shape[2] == d.CONV_DIM else 1
            windows.append((c.shape[3 - ch],))
            conv_strides.append((c.stride(0), c.stride(3 - ch), c.stride(ch)))
        self.conv_strides = _uniform("conv strides", conv_strides)
        self.conv_window = _uniform("conv window", windows)[0]
        if self.conv_window != self.rows + 2:
            raise ValueError(f"conv window {self.conv_window}: the megakernel's commit needs num_speculative_tokens "
                             f"{self.rows - 1} (a window of {self.rows + 2})")
        self.conv_ptrs = _ptrs(conv_states, dev)
        self._keep = (list(k_views), list(v_views), list(conv_states), list(ssm_states))
        self.bound = True
        logger.info("AITerMK megakernel bound to vLLM's caches: KV block size %d strides %s, conv window %d strides "
                    "%s, recurrent slot stride %d", self.block_size, self.kv_strides, self.conv_window,
                    self.conv_strides, self.ssm_s_slot)

    def verify(self, token_ids: torch.Tensor, base: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor,
               block_rows: torch.Tensor) -> torch.Tensor:
        """Launch over ``token_ids`` (int64 [rows]) at positions base .. base + rows - 1 (``base`` int [1]);
        ``slots`` int32 [48, rows] each DeltaNet layer's state slots for the request (vLLM's KV-cache groups
        give layers different slots), ``accepted`` int32 [1] its last step's accepted count (where its state
        and conv window start), ``block_rows`` int32 [16, blocks] each attention layer's block table for it.
        Returns the logits [rows, VOCAB] fp32 (reused next launch)."""
        inp, b, s = self.inp, self.binding, self.shared
        inp.token_ids.copy_(token_ids)
        inp.base.copy_(base)
        vk.launch_begin(b, self.delta_rows, self.state_col, self.ssm_ptrs, self.ssm_s_slot * 4, slots, accepted, inp,
                        self.mirror_slot, self.mirror_len, self.import_start, self.import_end)
        vk.conv_gather(self.conv_ptrs, self.conv_strides, slots, accepted, s.conv_history)
        vk.kv_import(self.k_ptrs, self.v_ptrs, self.kv_strides, block_rows, self.block_size, self.import_start,
                     self.import_end, s.kv_ptrs)
        self.kernel.launch(b.table)
        # the mirror holds positions < base (rows past the accepted ones are overwritten next launch)
        self.mirror_slot.copy_(inp.request)
        self.mirror_len.copy_(inp.base)
        return b.logits

    def commit(self, n_rows: torch.Tensor, slots: torch.Tensor, block_rows: torch.Tensor) -> None:
        """Settle the last verify's first ``n_rows`` rows (int32 [1], vLLM's ``num_accepted_tokens`` of this
        step) into vLLM's caches: each layer's recurrent state into its slot ``slots[layer, n_rows - 1]``, the
        conv windows, and the rows' K / V into the request's pages."""
        inp, s = self.inp, self.shared
        vk.state_commit(self.ssm_ptrs, self.ssm_s_slot, inp.state_slot, slots, s.records, n_rows, s.max_rows)
        vk.conv_commit(self.conv_ptrs, self.conv_strides, slots, s.conv_history, s.raw, self.conv_window)
        vk.kv_export(self.k_ptrs, self.v_ptrs, self.kv_strides, block_rows, self.block_size, inp.base, n_rows,
                     s.kv_ptrs, self.rows)

    def invalidate_mirror(self) -> None:
        """A new sequence: the next launch imports its KV from position 0."""
        self.mirror_slot.fill_(-1)
        self.mirror_len.zero_()

    def aux(self, layers=DFLASH2_AUX_LAYERS) -> torch.Tensor:
        """The last verify's residual streams after ``layers``, [rows, len(layers), HIDDEN] bf16."""
        return torch.stack([self.binding.residual1(layer + 1) for layer in layers], 1)

    def failed(self) -> bool:
        return int(self.binding.failed) != 0
