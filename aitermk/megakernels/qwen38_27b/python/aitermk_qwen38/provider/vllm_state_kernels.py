# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton kernels moving the model's state between vLLM's caches and the megakernel's layout.

vLLM's layouts (v1 KV cache, Qwen3.5 hybrid GDN with speculative decoding, ``mamba_cache_mode`` none):

- KV: per attention layer a paged cache; the K and V views are ``[blocks, block_size, 4 heads, 256]`` bf16
  with any strides but a contiguous head dimension (TRITON_ATTN: ``(B, H, N, 2 * 256)`` split in two);
  a request's block table maps position // block_size to a block;
- every request owns ``1 + num_spec`` = 8 state slots (``spec_state_indices``), and
  ``num_accepted_tokens`` (accepted drafts + 1) of its last step picks where the next step starts:
- conv state per layer ``[slot, 3 + num_spec, 10240]`` bf16 (or ``[slot, 10240, 3 + num_spec]``), in slot
  ``spec_state_indices[0]``; the window a step reads is entries ``a - 1 .. a + 1``, and a step of S rows writes
  ``[h2, h3, x_0 .. x_{S-1}]`` (its window less the oldest entry, then its inputs);
- recurrent state per layer ``[slot, 48, 128 v, 128 k]`` fp32, the megakernel's own layout: a step reads
  slot ``spec_state_indices[a - 1]``; after the step the state after its first n rows sits in slot
  ``spec_state_indices[n - 1]``.

vLLM's hybrid allocator gives each layer its own cache view, and layers of different KV-cache groups share
memory under different block tables: a request's state slots and KV blocks differ from layer to layer. So
every kernel takes per-layer base addresses and per-layer rows of slots / blocks (the SGLang kernels index
one stacked pool with one slot). The megakernel's own layout is as in ``sgl_state_kernels``.
"""

import triton
import triton.language as tl

from aitermk_qwen38.provider import dims as d


@triton.jit
def _launch_begin_kernel(
    # the recurrent state the step reads -> the table's initial_state cells
    TABLE, TABLE_STRIDE, DELTA_ROWS, STATE_PTRS, S_SLOT, SLOTS, ACCEPTED, STATE_SLOT,
    # the KV mirror
    MIRROR_SLOT, MIRROR_LEN, REQUEST, IMPORT_START, IMPORT_END,
    # the launch
    TOK_SRC, TOK_DST, BASE, POSITIONS, CONTEXT_BLOCKS, BLOCK_MAP, NATIVE, CONTROLS,
    COMMAND,
    STATE_COL: tl.constexpr, NDELTA: tl.constexpr, NDELTA_P2: tl.constexpr, ROWS: tl.constexpr, PAGE: tl.constexpr,
    NATIVE_TOTAL: tl.constexpr, CONTROL_N: tl.constexpr,
    COMMAND_WORD: tl.constexpr, MAX_EPOCH: tl.constexpr, BLOCK: tl.constexpr,
):
    """``launch_kernels.launch_begin`` over per-layer state bases (bytes), each layer's slot resolved on the
    device: ``SLOTS[j, ACCEPTED - 1]``, also stored in ``STATE_SLOT[j]`` for the commit."""
    j = tl.arange(0, NDELTA_P2)
    jm = j < NDELTA
    slot = tl.load(SLOTS + j * ROWS + tl.load(ACCEPTED) - 1, mask=jm, other=0)
    tl.store(STATE_SLOT + j, slot, mask=jm)
    row = tl.load(DELTA_ROWS + j, mask=jm, other=0)
    cell = tl.load(STATE_PTRS + j, mask=jm, other=0) + slot.to(tl.int64) * S_SLOT
    tl.store(TABLE + row * TABLE_STRIDE + STATE_COL, cell, mask=jm)
    # import what the engine's cache holds that the mirror does not: from its extent
    # if it holds this request, else from 0
    base = tl.load(BASE).to(tl.int32)
    held = tl.load(MIRROR_SLOT) == tl.load(REQUEST)
    tl.store(IMPORT_START, tl.where(held, tl.minimum(tl.load(MIRROR_LEN), base), 0))
    tl.store(IMPORT_END, base)
    r = tl.arange(0, ROWS)
    tl.store(TOK_DST + r, tl.load(TOK_SRC + r))
    pos = base + r
    tl.store(POSITIONS + r, pos)
    tl.store(CONTEXT_BLOCKS + r, tl.load(BLOCK_MAP + pos // PAGE))
    # a fresh control block every launch, as the qualified drivers do
    z = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(0, NATIVE_TOTAL, BLOCK):
        o = i + tl.arange(0, BLOCK)
        tl.store(NATIVE + o, z, mask=o < NATIVE_TOTAL)
    for i in range(0, CONTROL_N, BLOCK):
        o = i + tl.arange(0, BLOCK)
        tl.store(CONTROLS + o, z, mask=o < CONTROL_N)
    tl.debug_barrier()
    # the command epoch: non-zero, below MAX_EPOCH, advancing every launch
    epoch = tl.load(COMMAND) % (MAX_EPOCH - 1) + 1
    tl.store(COMMAND, epoch)
    tl.store(NATIVE + COMMAND_WORD, epoch)


def launch_begin(binding, delta_rows, state_col: int, state_ptrs, s_slot_bytes: int, slots, accepted, inp,
                 mirror_slot, mirror_len, import_start, import_end) -> None:
    """``state_ptrs`` int64 [48] the layers' recurrent caches (bytes), ``s_slot_bytes`` their slot stride;
    ``slots`` int32 [48, 8] each layer's state slots for the request, ``accepted`` int32 [1] its last step's
    accepted count; ``inp.state_slot`` int32 [48] receives the slots read."""
    b = binding
    _launch_begin_kernel[(1,)](
        b.table, b.table.stride(0), delta_rows, state_ptrs, s_slot_bytes, slots, accepted, inp.state_slot,
        mirror_slot, mirror_len, inp.request, import_start, import_end,
        inp.token_ids, b.endpoints["token_ids"], inp.base, b.positions, b.context_blocks, b.block_map,
        b.native, b.controls, b.command,
        STATE_COL=state_col, NDELTA=delta_rows.numel(),
        NDELTA_P2=triton.next_power_of_2(delta_rows.numel()), ROWS=b.rows, PAGE=d.PAGE_TOKENS,
        NATIVE_TOTAL=b.native.numel(), CONTROL_N=b.controls.numel(),
        COMMAND_WORD=d.COMMAND_WORD, MAX_EPOCH=d.MAX_COMMAND_EPOCH,
        BLOCK=4096, num_warps=8,
    )  # fmt: skip


# ---- recurrent state ---------------------------------------------------------


@triton.jit
def _state_commit_kernel(
    state_ptrs, s_slot, slot_src, slots, rec, s_rec_layer, accepted, ROWS: tl.constexpr, MAX_ROWS: tl.constexpr,
    BV: tl.constexpr,
):
    """``state_kernels.state_commit`` over per-layer caches, a chain, from ``slot_src[layer]`` into
    ``slots[layer, accepted - 1]``."""
    lh = tl.program_id(0)
    vb = tl.program_id(1)
    layer = lh // 48
    head = lh % 48
    k = tl.load(accepted)
    vs = vb * BV + tl.arange(0, BV)
    ks = tl.arange(0, 128)
    tile = head * 16384 + vs[:, None] * 128 + ks[None, :]
    base = tl.load(state_ptrs + layer).to(tl.pointer_type(tl.float32))
    st = tl.load(base + tl.load(slot_src + layer).to(tl.int64) * s_slot + tile)
    for t in tl.static_range(MAX_ROWS):
        if t < k:
            r = rec + layer.to(tl.int64) * s_rec_layer + (t * 48 + head) * 260
            dk = tl.load(r + 256)
            dl = tl.load(r + vs)
            kk = tl.load(r + 128 + ks)
            st = dk * st + dl[:, None] * kk[None, :]
    tl.store(base + tl.load(slots + layer * ROWS + k - 1).to(tl.int64) * s_slot + tile, st)


def state_commit(state_ptrs, s_slot: int, slot_src, slots, rec, accepted, max_rows: int) -> None:
    """``state_ptrs`` int64 [48] (fp32 caches, slot stride ``s_slot`` elements); ``slot_src`` int32 [48] the
    slots the launch read; ``slots`` int32 [48, 8]; ``rec`` the records [48, max_rows * 48 * 260];
    ``accepted`` int32 [1] the rows kept."""
    _state_commit_kernel[(state_ptrs.numel() * 48, 4)](
        state_ptrs, s_slot, slot_src, slots, rec, rec.stride(0), accepted, ROWS=slots.shape[1], MAX_ROWS=max_rows,
        BV=32, num_warps=4)


# ---- conv ------------------------------------------------------------------


@triton.jit
def _conv_gather_kernel(conv_ptrs, s_slot, s_tok, s_ch, slots, accepted, hist, DIM: tl.constexpr,
                        ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """hist[layer, ch, 1 + j] = conv[layer][slots[layer, 0], accepted - 1 + j, ch], j = 0..2 (2 the newest)."""
    layer = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    off = tl.load(accepted) - 1
    base = tl.load(conv_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    src = base + tl.load(slots + layer * ROWS).to(tl.int64) * s_slot + ch.to(tl.int64) * s_ch
    dst = hist + (layer * DIM + ch).to(tl.int64) * 4
    for j in tl.static_range(3):
        tl.store(dst + 1 + j, tl.load(src + (off + j) * s_tok))


@triton.jit
def _window_entry(h, x, idx: tl.constexpr, ROW: tl.constexpr):
    """Entry idx of [h1, h2, h3, x_0, ...]: the history (bf16) or a raw row."""
    if idx < 3:
        return tl.load(h + 1 + idx)
    return tl.load(x + (idx - 3) * ROW).to(tl.bfloat16)


@triton.jit
def _conv_commit_kernel(conv_ptrs, s_slot, s_tok, s_ch, slots, hist, raw, s_raw_layer, DIM: tl.constexpr,
                        ROW: tl.constexpr, LEN: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """vLLM's window after the launch: entry e = [h1, h2, h3, x_0 .. x_{rows-1}][e + 1], e < LEN = rows + 2."""
    layer = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(conv_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    dst = base + tl.load(slots + layer * ROWS).to(tl.int64) * s_slot + ch.to(tl.int64) * s_ch
    h = hist + (layer * DIM + ch).to(tl.int64) * 4
    x = raw + layer.to(tl.int64) * s_raw_layer + ch
    for e in tl.static_range(LEN):
        tl.store(dst + e * s_tok, _window_entry(h, x, e + 1, ROW))


def conv_gather(conv_ptrs, strides, slots, accepted, hist) -> None:
    """``conv_ptrs`` int64 [48] the layers' conv caches, ``strides`` (slot, entry, channel) in elements;
    ``slots`` int32 [48, 8] (each layer's window sits in its first slot); ``hist`` [48, 10240 * 4]."""
    _conv_gather_kernel[(conv_ptrs.numel(), d.CONV_DIM // 1024)](
        conv_ptrs, *strides, slots, accepted, hist, DIM=d.CONV_DIM, ROWS=slots.shape[1], BLOCK=1024, num_warps=4)


def conv_commit(conv_ptrs, strides, slots, hist, raw, window: int) -> None:
    """``raw`` the kernel's pre-conv rows [48, rows * PROJECTED]; ``window`` vLLM's entries a slot holds."""
    _conv_commit_kernel[(conv_ptrs.numel(), d.CONV_DIM // 1024)](
        conv_ptrs, *strides, slots, hist, raw, raw.stride(0),
        DIM=d.CONV_DIM, ROW=d.PROJECTED, LEN=window, ROWS=slots.shape[1], BLOCK=1024, num_warps=4)


# ---- KV ----------------------------------------------------------------------


@triton.jit
def _kv_import_kernel(k_ptrs, v_ptrs, s_blk, s_tok, s_head, block_rows, s_rows, block_size, start_pos, end_pos,
                      mirror_ptrs, HD: tl.constexpr, KVH: tl.constexpr, PROGRAMS: tl.constexpr, T: tl.constexpr):
    """Positions [start_pos, end_pos) of one (layer, head), vLLM's pages -> the mirror pages."""
    layer = tl.program_id(0)
    p = tl.program_id(1)
    h = tl.program_id(2)
    start = tl.load(start_pos)
    end = tl.load(end_pos)
    kc = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    vc = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    mc = tl.load(mirror_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    block_row = block_rows + layer * s_rows
    t = tl.arange(0, T)
    dd = tl.arange(0, HD)
    for c in range(start // T + p, (end + T - 1) // T, PROGRAMS):
        pos = c * T + t
        ok = (pos >= start) & (pos < end)
        blk = tl.load(block_row + pos // block_size, mask=ok, other=0).to(tl.int64)
        src = (blk * s_blk + (pos % block_size) * s_tok + h * s_head)[:, None] + dd[None, :]
        kv_k = tl.load(kc + src, mask=ok[:, None])
        kv_v = tl.load(vc + src, mask=ok[:, None])
        dst = mc + (((pos // 32).to(tl.int64) * KVH + h) * 2 * 32 + pos % 32)[:, None] * HD + dd[None, :]
        tl.store(dst, kv_k, mask=ok[:, None])
        tl.store(dst + 32 * HD, kv_v, mask=ok[:, None])


@triton.jit
def _kv_export_kernel(k_ptrs, v_ptrs, s_blk, s_tok, s_head, block_rows, s_rows, block_size, base, rows,
                      mirror_ptrs, HD: tl.constexpr, KVH: tl.constexpr):
    """This launch's first ``rows`` rows' K / V (mirror pages) -> vLLM's pages."""
    layer = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    if r < tl.load(rows):
        pos = (tl.load(base) + r).to(tl.int64)
        blk = tl.load(block_rows + layer * s_rows + pos // block_size).to(tl.int64)
        kc = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        vc = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        mc = tl.load(mirror_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        dd = tl.arange(0, HD)
        src = mc + ((pos // 32 * KVH + h) * 2 * 32 + pos % 32) * HD + dd
        dst = blk * s_blk + (pos % block_size) * s_tok + h * s_head + dd
        tl.store(kc + dst, tl.load(src))
        tl.store(vc + dst, tl.load(src + 32 * HD))


def kv_import(k_ptrs, v_ptrs, strides, block_rows, block_size: int, start_pos, end_pos, mirror_ptrs,
              programs=32) -> None:
    """``strides`` (block, position, head) of the K / V views in elements; ``block_rows`` int32 [16, blocks]
    each attention layer's block table for the request; [start, end) int32 [1]."""
    _kv_import_kernel[(len(d.ATTENTION_LAYERS), programs, d.KV_HEADS)](
        k_ptrs, v_ptrs, *strides, block_rows, block_rows.stride(0), block_size, start_pos, end_pos, mirror_ptrs,
        HD=d.HEAD_DIM, KVH=d.KV_HEADS, PROGRAMS=programs, T=16, num_warps=4)


def kv_export(k_ptrs, v_ptrs, strides, block_rows, block_size: int, base, rows, mirror_ptrs, max_rows) -> None:
    _kv_export_kernel[(len(d.ATTENTION_LAYERS), max_rows, d.KV_HEADS)](
        k_ptrs, v_ptrs, *strides, block_rows, block_rows.stride(0), block_size, base, rows, mirror_ptrs,
        HD=d.HEAD_DIM, KVH=d.KV_HEADS, num_warps=2)


# ---- a round's inputs ----------------------------------------------------------


@triton.jit
def _stage_kernel(ids_src, pos_src, acc_src, ids, base, acc, ROWS: tl.constexpr):
    """vLLM's step tensors -> the round's static buffers, in one launch."""
    r = tl.arange(0, ROWS)
    tl.store(ids + r, tl.load(ids_src + r).to(tl.int64))
    tl.store(base, tl.load(pos_src).to(tl.int64))
    tl.store(acc, tl.load(acc_src).to(tl.int32))


@triton.jit
def _stage_rows_kernel(src, n, dst, layers, s_dst, BLOCK: tl.constexpr):
    """One KV-cache group's row (state slots or block table, ``n`` entries) into the rows of its layers."""
    layer = tl.load(layers + tl.program_id(0))
    for i in range(0, n, BLOCK):
        o = i + tl.arange(0, BLOCK)
        tl.store(dst + layer * s_dst + o, tl.load(src + o, mask=o < n, other=0).to(tl.int32), mask=o < n)


def stage(ids_src, pos_src, acc_src, ids, base, acc) -> None:
    """Copy a step's token ids, first position and accepted count into the round's buffers."""
    _stage_kernel[(1,)](ids_src, pos_src, acc_src, ids, base, acc, ROWS=ids.numel(), num_warps=1)


def stage_rows(src, dst, layers) -> None:
    """``src`` a group's row; ``dst`` [layers, n] int32; ``layers`` int64 the group's rows of ``dst``."""
    n = min(dst.shape[1], src.numel())
    _stage_rows_kernel[(layers.numel(),)](src, n, dst, layers, dst.stride(0), BLOCK=1024, num_warps=4)
