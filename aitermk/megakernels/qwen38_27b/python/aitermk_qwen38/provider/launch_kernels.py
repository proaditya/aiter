# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A launch's bookkeeping before the megakernel runs, as one Triton launch: the recurrent state slot the
step reads into the pointer table, the KV mirror's import range, and the launch inputs (token ids,
positions and pages, a fresh control block, the command epoch).
"""

import triton
import triton.language as tl

from aitermk_qwen38.provider import dims as d

@triton.jit
def _launch_begin_kernel(
    # the recurrent state the step reads -> the table's initial_state cells
    TABLE, TABLE_STRIDE, DELTA_ROWS, STATE_PTR, S_LAYER, S_SLOT, STATE_SLOT,
    # the KV mirror
    MIRROR_SLOT, MIRROR_LEN, REQUEST, IMPORT_START, IMPORT_END,
    # the launch
    TOK_SRC, TOK_DST, BASE, POSITIONS, CONTEXT_BLOCKS, BLOCK_MAP, NATIVE, CONTROLS,
    COMMAND,
    STATE_COL: tl.constexpr, NDELTA: tl.constexpr, NDELTA_P2: tl.constexpr, ROWS: tl.constexpr, PAGE: tl.constexpr,
    NATIVE_N: tl.constexpr, NATIVE_TOTAL: tl.constexpr, CONTROL_N: tl.constexpr,
    COMMAND_WORD: tl.constexpr, MAX_EPOCH: tl.constexpr, BLOCK: tl.constexpr,
):
    j = tl.arange(0, NDELTA_P2)
    jm = j < NDELTA
    row = tl.load(DELTA_ROWS + j, mask=jm, other=0)
    slot = tl.load(STATE_SLOT).to(tl.int64)
    cell = STATE_PTR + j.to(tl.int64) * S_LAYER + slot * S_SLOT
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


def launch_begin(binding, delta_rows, state_col: int, pools, inp,
                 mirror_slot, mirror_len, import_start, import_end) -> None:
    state = pools.state
    nbytes = state.element_size()
    b = binding
    _launch_begin_kernel[(1,)](
        b.table, b.table.stride(0), delta_rows, state.data_ptr(), state.stride(0) * nbytes,
        state.stride(1) * nbytes, inp.state_slot,
        mirror_slot, mirror_len, inp.request, import_start, import_end,
        inp.token_ids, b.endpoints["token_ids"], inp.base, b.positions, b.context_blocks, b.block_map,
        b.native, b.controls, b.command,
        STATE_COL=state_col, NDELTA=delta_rows.numel(),
        NDELTA_P2=triton.next_power_of_2(delta_rows.numel()), ROWS=b.rows, PAGE=d.PAGE_TOKENS,
        NATIVE_N=d.NATIVE_WORDS, NATIVE_TOTAL=b.native.numel(), CONTROL_N=b.controls.numel(),
        COMMAND_WORD=d.COMMAND_WORD, MAX_EPOCH=d.MAX_COMMAND_EPOCH,
        BLOCK=4096, num_warps=8,
    )  # fmt: skip
