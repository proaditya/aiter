# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton kernels moving the model's state between SGLang's pools and the megakernel's layout.

SGLang's layouts (``mem_cache/memory_pool.py``, Qwen3.5 hybrid GDN):

- KV: one ``[slots, 4 heads, 256]`` bf16 buffer per attention layer for K and for V, indexed by token
  slot (``req_to_token`` maps a request's positions to slots);
- conv state ``[layer, slot, 10240, 3]`` bf16 (``conv_window_axis = -1``), entry 2 the newest input;
- recurrent state ``[layer, slot, 48, 128 v, 128 k]`` fp32 (``temporal``, (HV, V, K)): the megakernel's
  own layout, read in place (``launch_kernels.launch_begin``) and committed in place
  (``state_kernels.state_commit`` with the same source and destination slot).

The kernel's: conv history ``[layer][10240][4]`` (entries 1..3 the three most recent inputs), KV pages of
32 positions ``[page][head][K | V][32][256]`` (the mirror).
"""

import os

import triton
import triton.language as tl

from aitermk_qwen38.provider import dims as d

# ---- conv ------------------------------------------------------------------


@triton.jit
def _conv_gather_kernel(conv, s_layer, s_slot, s_ch, s_win, slot, hist, DIM: tl.constexpr, BLOCK: tl.constexpr,
                        REV: tl.constexpr):
    """hist[layer, ch, 1 + j] = conv[layer, slot, ch, j], j = 0..2 (2 the newest)."""
    layer = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(slot).to(tl.int64)
    src = conv + layer.to(tl.int64) * s_layer + s * s_slot + ch.to(tl.int64) * s_ch
    dst = hist + (layer * DIM + ch).to(tl.int64) * 4
    for j in tl.static_range(3):
        tl.store(dst + 1 + j, tl.load(src + ((2 - j) if REV else j) * s_win))


@triton.jit
def _window_entry(h, x, idx, ROW: tl.constexpr):
    """Entry idx of [h1, h2, h3, x_0, ...]: the history (bf16) or a raw row."""
    from_hist = tl.load(h + 1 + tl.minimum(idx, 2))
    from_raw = tl.load(x + tl.maximum(idx - 3, 0).to(tl.int64) * ROW)
    return tl.where(idx < 3, from_hist, from_raw.to(from_hist.dtype))


@triton.jit
def _conv_commit_kernel(conv, s_layer, s_slot, s_ch, s_win, slot, hist, raw, s_raw_layer, n_rows,
                        DIM: tl.constexpr, ROW: tl.constexpr, BLOCK: tl.constexpr, REV: tl.constexpr):
    """The window after the first n rows of the launch: the last three of [h1, h2, h3, x_0 .. x_{n-1}]."""
    layer = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(slot).to(tl.int64)
    n = tl.load(n_rows)
    dst = conv + layer.to(tl.int64) * s_layer + s * s_slot + ch.to(tl.int64) * s_ch
    h = hist + (layer * DIM + ch).to(tl.int64) * 4
    x = raw + layer.to(tl.int64) * s_raw_layer + ch
    v0 = _window_entry(h, x, n, ROW)
    v1 = _window_entry(h, x, n + 1, ROW)
    v2 = _window_entry(h, x, n + 2, ROW)
    if REV:
        tl.store(dst + 2 * s_win, v0)
        tl.store(dst + s_win, v1)
        tl.store(dst, v2)
    else:
        tl.store(dst, v0)
        tl.store(dst + s_win, v1)
        tl.store(dst + 2 * s_win, v2)


# diagnostics: the window axis in the opposite order (entry 0 the newest)
_REV = os.environ.get("AITERMK_SGL_CONV_REVERSE") == "1"


def _conv_strides(conv):
    assert conv.dim() == 4 and conv.shape[2] == d.CONV_DIM and conv.shape[3] == 3, conv.shape
    return conv.stride(0), conv.stride(1), conv.stride(2), conv.stride(3)


def conv_gather(conv, slot, hist) -> None:
    """``conv`` SGLang's GDN conv pool [48, slots, 10240, 3]; ``slot`` int32 [1]; ``hist`` [48, 10240 * 4]."""
    _conv_gather_kernel[(conv.shape[0], d.CONV_DIM // 1024)](
        conv, *_conv_strides(conv), slot, hist, DIM=d.CONV_DIM, BLOCK=1024, REV=_REV, num_warps=4)


def conv_commit(conv, slot, hist, raw, n_rows) -> None:
    """``raw`` the kernel's pre-conv rows [48, rows * PROJECTED]; ``n_rows`` int32 [1], the rows committed."""
    _conv_commit_kernel[(conv.shape[0], d.CONV_DIM // 1024)](
        conv, *_conv_strides(conv), slot, hist, raw, raw.stride(0), n_rows,
        DIM=d.CONV_DIM, ROW=d.PROJECTED, BLOCK=1024, REV=_REV, num_warps=4)


# ---- KV ----------------------------------------------------------------------


@triton.jit
def _kv_import_kernel(k_ptrs, v_ptrs, req_to_token, s_req, req_index, start_pos, end_pos, mirror_ptrs,
                      HD: tl.constexpr, KVH: tl.constexpr, PROGRAMS: tl.constexpr, T: tl.constexpr):
    """Positions [start_pos, end_pos) of one (layer, head), SGLang's token slots -> the mirror pages."""
    layer = tl.program_id(0)
    p = tl.program_id(1)
    h = tl.program_id(2)
    start = tl.load(start_pos)
    end = tl.load(end_pos)
    kc = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    vc = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    mc = tl.load(mirror_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    req_row = req_to_token + tl.load(req_index).to(tl.int64) * s_req
    t = tl.arange(0, T)
    dd = tl.arange(0, HD)
    for c in range(start // T + p, (end + T - 1) // T, PROGRAMS):
        pos = c * T + t
        ok = (pos >= start) & (pos < end)
        slot = tl.load(req_row + pos, mask=ok, other=0).to(tl.int64)
        src = (slot * KVH + h)[:, None] * HD + dd[None, :]
        kv_k = tl.load(kc + src, mask=ok[:, None])
        kv_v = tl.load(vc + src, mask=ok[:, None])
        dst = mc + (((pos // 32).to(tl.int64) * KVH + h) * 2 * 32 + pos % 32)[:, None] * HD + dd[None, :]
        tl.store(dst, kv_k, mask=ok[:, None])
        tl.store(dst + 32 * HD, kv_v, mask=ok[:, None])


@triton.jit
def _kv_export_kernel(k_ptrs, v_ptrs, slots, base, rows, mirror_ptrs, HD: tl.constexpr, KVH: tl.constexpr):
    """This launch's rows' K / V (mirror pages) -> SGLang's token slots ``slots[r]``."""
    layer = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    if r < tl.load(rows):
        slot = tl.load(slots + r).to(tl.int64)
        kc = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        vc = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        mc = tl.load(mirror_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        pos = (tl.load(base) + r).to(tl.int64)
        dd = tl.arange(0, HD)
        src = mc + ((pos // 32 * KVH + h) * 2 * 32 + pos % 32) * HD + dd
        dst = (slot * KVH + h) * HD + dd
        tl.store(kc + dst, tl.load(src))
        tl.store(vc + dst, tl.load(src + 32 * HD))


def kv_import(k_ptrs, v_ptrs, req_to_token, req_index, start_pos, end_pos, mirror_ptrs, programs=32) -> None:
    """Row ``req_index`` (int [1]) of SGLang's ``req_to_token`` (position -> slot); [start, end) int32 [1]."""
    _kv_import_kernel[(len(d.ATTENTION_LAYERS), programs, d.KV_HEADS)](
        k_ptrs, v_ptrs, req_to_token, req_to_token.stride(0), req_index, start_pos, end_pos, mirror_ptrs,
        HD=d.HEAD_DIM, KVH=d.KV_HEADS, PROGRAMS=programs, T=16, num_warps=4)


def kv_export(k_ptrs, v_ptrs, slots, base, rows, mirror_ptrs, max_rows) -> None:
    _kv_export_kernel[(len(d.ATTENTION_LAYERS), max_rows, d.KV_HEADS)](
        k_ptrs, v_ptrs, slots, base, rows, mirror_ptrs, HD=d.HEAD_DIM, KVH=d.KV_HEADS, num_warps=2)
