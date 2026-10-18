# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One greedy DFlash 2 round of the megakernel over SGLang's pools (one sequence).

``SglMonoRound`` holds the converted MXFP4 weights, one chain build (``m8`` by default), its pointer table
and the KV mirror. ``verify`` launches the build over 8 consecutive positions of a request, reading SGLang's
recurrent state in place and its conv window and KV; ``commit`` settles the accepted prefix back into
SGLang's pools (state replayed from the GDN records in place, conv window, KV rows). SGLang stays the
record: after a commit its pools hold exactly what its own verify + commit would have written.
"""

import logging
import os

import torch

from aitermk_qwen38.provider import dims as d
from aitermk_qwen38.provider import launch_kernels as lk
from aitermk_qwen38.provider import sgl_state_kernels as sk
from aitermk_qwen38.provider import state_kernels as rk
from aitermk_qwen38.provider.binding import KernelBinding, SharedBuffers
from aitermk_qwen38.provider.kernels.build import VARIANTS, build
from aitermk_qwen38.provider.kernels.hip import Megakernel
from aitermk_qwen38.provider.verdict import RowArgmax

logger = logging.getLogger("aitermk_qwen38")

# diagnostics: SGLang's recurrent state read as [head][k][v] (swapped around each launch)
_STATE_T = os.environ.get("AITERMK_SGL_STATE_T") == "1"

# the residual streams DFlash 2 drafts from: after target layers 5, 19, 33, 47, 61
DFLASH2_AUX_LAYERS = (5, 19, 33, 47, 61)


class _Pools:
    """What ``launch_kernels.launch_begin`` reads off a pool set: the recurrent state."""

    def __init__(self, state: torch.Tensor) -> None:
        self.state = state


class _Inputs:
    """A launch's device inputs, the fields ``launch_kernels.launch_begin`` reads."""

    def __init__(self, rows: int, device) -> None:
        def scalar(v=0):
            return torch.full((1,), v, dtype=torch.int32, device=device)

        self.token_ids = torch.zeros(rows, dtype=torch.int64, device=device)
        self.base = scalar()
        self.state_slot = scalar()
        self.request = scalar()


class SglMonoRound:
    def __init__(self, weights, max_model_len: int, *, conv: torch.Tensor, state: torch.Tensor,
                 k_buffers: list[torch.Tensor], v_buffers: list[torch.Tensor], variant: str = "m8", device=None) -> None:
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        dev = self.device
        v = VARIANTS[variant]
        self.rows = v.rows
        self._check_pools(conv, state, k_buffers, v_buffers)
        self.conv, self.state = conv, state
        self.k_ptrs = torch.tensor([t.data_ptr() for t in k_buffers], dtype=torch.int64, device=dev)
        self.v_ptrs = torch.tensor([t.data_ptr() for t in v_buffers], dtype=torch.int64, device=dev)
        max_pages = (max_model_len + self.rows + d.PAGE_TOKENS - 1) // d.PAGE_TOKENS
        self.shared = SharedBuffers(self.rows, max_pages, dev)
        self.kernel = Megakernel(build(variant))
        self.binding = KernelBinding(self.kernel.rows, weights, self.shared, dev)
        self.pools = _Pools(state)
        self.inp = _Inputs(self.rows, dev)
        self.delta_rows = torch.tensor(d.DELTA_LAYERS, dtype=torch.int64, device=dev)
        self.state_col = d.DELTA_SLOTS.index("initial_state")
        self.mirror_slot = torch.full((1,), -1, dtype=torch.int32, device=dev)
        self.mirror_len = torch.zeros(1, dtype=torch.int32, device=dev)
        self.import_start = torch.zeros(1, dtype=torch.int32, device=dev)
        self.import_end = torch.zeros(1, dtype=torch.int32, device=dev)
        self.identity = torch.arange(self.rows, dtype=torch.int64, device=dev)
        self.ones = torch.ones(1, dtype=torch.int32, device=dev)
        self.argmax = RowArgmax(self.rows, d.VOCAB, dev)
        self.target = torch.zeros(self.rows, dtype=torch.int64, device=dev)

    @staticmethod
    def _check_pools(conv, state, k_buffers, v_buffers) -> None:
        n = len(d.DELTA_LAYERS)
        if conv.dtype != torch.bfloat16 or conv.dim() != 4 or conv.shape[0] != n or tuple(conv.shape[2:]) != (d.CONV_DIM, 3):
            raise ValueError(f"conv pool {tuple(conv.shape)} {conv.dtype}: need [{n}, slots, {d.CONV_DIM}, 3] bf16")
        want = (d.V_HEADS, d.GDN_DIM, d.GDN_DIM)
        if state.dtype != torch.float32 or state.shape[0] != n or tuple(state.shape[2:]) != want:
            raise ValueError(f"recurrent pool {tuple(state.shape)} {state.dtype}: need [{n}, slots, *{want}] fp32 "
                             "(--mamba-ssm-dtype float32)")
        if state[0, 0].stride() != (d.GDN_DIM * d.GDN_DIM, d.GDN_DIM, 1):
            raise ValueError("recurrent pool slot must be contiguous [48][128 v][128 k]")
        if len(k_buffers) != len(d.ATTENTION_LAYERS):
            raise ValueError(f"{len(k_buffers)} K buffers for {len(d.ATTENTION_LAYERS)} attention layers")
        for t in list(k_buffers) + list(v_buffers):
            if t.dtype != torch.bfloat16 or tuple(t.shape[1:]) != (d.KV_HEADS, d.HEAD_DIM) or not t.is_contiguous():
                raise ValueError(f"KV buffer {tuple(t.shape)} {t.dtype}: need contiguous [slots, {d.KV_HEADS}, {d.HEAD_DIM}] bf16")

    def verify(self, token_ids: torch.Tensor, base: torch.Tensor, mamba_slot: torch.Tensor, req_to_token: torch.Tensor,
               request: torch.Tensor) -> torch.Tensor:
        """Launch over ``token_ids`` (int64 [rows]) at positions base .. base + rows - 1 (``base`` int [1]);
        ``mamba_slot`` the request's GDN state slot, ``request`` its row of ``req_to_token`` (int [1], also the
        mirror's name for it). Returns the target's greedy token per row (int64 [rows], reused next call)."""
        inp, b, s = self.inp, self.binding, self.shared
        inp.token_ids.copy_(token_ids)
        inp.base.copy_(base)
        inp.state_slot.copy_(mamba_slot)
        inp.request.copy_(request)
        if _STATE_T:
            self._transpose_slot()
        lk.launch_begin(b, self.delta_rows, self.state_col, self.pools, inp,
                        self.mirror_slot, self.mirror_len, self.import_start, self.import_end)
        sk.conv_gather(self.conv, inp.state_slot, s.conv_history)
        sk.kv_import(self.k_ptrs, self.v_ptrs, req_to_token, inp.request, self.import_start, self.import_end, s.kv_ptrs)
        self.kernel.launch(b.table)
        # the mirror holds positions < base (rows past the accepted ones are overwritten next launch)
        self.mirror_slot.copy_(inp.request)
        self.mirror_len.copy_(inp.base)
        self.argmax(b.logits, self.rows, self.target)
        return self.target

    def commit(self, n_rows: torch.Tensor, out_slots: torch.Tensor) -> None:
        """Settle the last verify's first ``n_rows`` rows (int32 [1]) into SGLang's pools: recurrent state in
        place, conv window, and the rows' K / V into ``out_slots`` (int64 [rows], the rows' token slots)."""
        inp, s = self.inp, self.shared
        rk.state_commit(self.state, inp.state_slot, inp.state_slot, s.records, self.identity, n_rows, self.ones, s.max_rows)
        sk.conv_commit(self.conv, inp.state_slot, s.conv_history, s.raw, n_rows)
        sk.kv_export(self.k_ptrs, self.v_ptrs, out_slots, inp.base, n_rows, s.kv_ptrs, self.rows)
        if _STATE_T:
            self._transpose_slot()

    def _transpose_slot(self) -> None:
        """Diagnostics: the request's recurrent state with its last two axes swapped (in place)."""
        slot = int(self.inp.state_slot)
        st = self.state[:, slot]
        st.copy_(st.transpose(-1, -2).contiguous())

    def invalidate_mirror(self) -> None:
        """A new sequence (SGLang reuses request indices): the next launch imports its KV from position 0."""
        self.mirror_slot.fill_(-1)
        self.mirror_len.zero_()

    def aux(self, layers=DFLASH2_AUX_LAYERS) -> torch.Tensor:
        """The last verify's residual streams after ``layers``, [rows, len(layers), HIDDEN] bf16."""
        return torch.stack([self.binding.residual1(layer + 1) for layer in layers], 1)

    def failed(self) -> bool:
        return int(self.binding.failed) != 0
