# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The speculative commit of the GDN recurrent state: the state after a verify's accepted prefix, replayed
from the launch's records (``kernels/gen_replay_records.py``) in closed form. Pools are ``[layer, slot, 48,
128 v, 128 k]`` fp32; the source and destination slot may be the same (one program per tile).
"""

import triton
import triton.language as tl

@triton.jit
def _state_commit_kernel(
    pool, s_layer, s_slot, slot_src, slot_dst, rec, s_rec_layer, path,
    accepted, pending, MAX_ROWS: tl.constexpr, BV: tl.constexpr,
):
    """The recurrent state after the pending launch's first k = accepted rows
    along ``path`` (rows in order; the identity for a chain): S <- d_t S +
    delta_t key_t^T, from the launch's initial state in ``slot_src`` into
    ``slot_dst``. One program a (layer, head, value block); nothing when no
    launch is pending."""
    if tl.load(pending) != 0:
        lh = tl.program_id(0)
        vb = tl.program_id(1)
        layer = lh // 48
        head = lh % 48
        k = tl.load(accepted)
        vs = vb * BV + tl.arange(0, BV)
        ks = tl.arange(0, 128)
        tile = head * 16384 + vs[:, None] * 128 + ks[None, :]
        base = pool + layer.to(tl.int64) * s_layer
        st = tl.load(base + tl.load(slot_src).to(tl.int64) * s_slot + tile)
        for t in tl.static_range(MAX_ROWS):
            if t < k:
                row = tl.load(path + t)
                r = rec + layer.to(tl.int64) * s_rec_layer + (row * 48 + head) * 260
                dk = tl.load(r + 256)
                dl = tl.load(r + vs)
                kk = tl.load(r + 128 + ks)
                st = dk * st + dl[:, None] * kk[None, :]
        tl.store(base + tl.load(slot_dst).to(tl.int64) * s_slot + tile, st)


def state_commit(pool, slot_src, slot_dst, rec, path, accepted, pending, max_rows) -> None:
    """``pool`` the recurrent pool [48, slots, 48, 128, 128] fp32 (VK);
    ``rec`` the records [48, max_rows * 48 * 260]; ``path`` int64 [max_rows]."""
    layers = pool.shape[0]
    _state_commit_kernel[(layers * 48, 4)](
        pool, pool.stride(0), pool.stride(1), slot_src, slot_dst, rec,
        rec.stride(0), path, accepted, pending, MAX_ROWS=max_rows, BV=32, num_warps=4,
    )  # fmt: skip
