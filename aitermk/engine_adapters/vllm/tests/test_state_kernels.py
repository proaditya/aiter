# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU conformance of ``aitermk_qwen38.provider.vllm_state_kernels`` against vLLM's own cache semantics.

Runs on one GPU with vLLM and ``aitermk_qwen38`` importable, no checkpoint: random caches in vLLM's layouts.

- conv: gathering the window at ``num_accepted - 1`` and committing a launch leaves the conv cache exactly as
  vLLM's speculative ``causal_conv1d_update`` (the GDN layer's conv step) leaves it for the same inputs;
- recurrent state: the commit replays the records from the slot the launch read into ``slots[n - 1]`` and
  touches no other slot;
- KV: import from / export to TRITON_ATTN's packed paged cache through a block table match a torch gather;
- every layer through its own slots / blocks (vLLM's KV-cache groups give layers different ones);
- launch bookkeeping: the state slot, the table's state cells and the import range.
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from aitermk_qwen38.provider import dims as d  # noqa: E402
from aitermk_qwen38.provider import vllm_state_kernels as vk  # noqa: E402

DEV = torch.device("cuda")
ROWS = 8
WINDOW = ROWS + 2


def _ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=DEV)


def _conv_caches(layers, slots, dim_first):
    """Per-layer conv caches inside one shared allocation, as vLLM's hybrid allocator lays them out."""
    page = WINDOW * d.CONV_DIM + 512  # a page wider than the state, as with the attention page size
    raw = torch.randn(layers, slots, page, device=DEV).to(torch.bfloat16)
    out = []
    for i in range(layers):
        v = raw[i, :, : WINDOW * d.CONV_DIM]
        out.append(v.view(slots, d.CONV_DIM, WINDOW) if dim_first else v.view(slots, WINDOW, d.CONV_DIM))
    return out


def _strides(c):
    ch = 1 if c.shape[1] == d.CONV_DIM else 2
    return c.stride(0), c.stride(3 - ch), c.stride(ch)


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("accepted", [1, 3, 8])
def test_conv_matches_vllm_spec_update(dim_first, accepted):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    layers, nslots = 3, 20
    convs = _conv_caches(layers, nslots, dim_first)
    slots = torch.stack([torch.randperm(nslots - 1, device=DEV)[:ROWS] + 1 for _ in range(layers)]).to(torch.int32)
    acc = torch.tensor([accepted], dtype=torch.int32, device=DEV)
    rawrows = torch.randn(layers, ROWS * d.PROJECTED, device=DEV).to(torch.bfloat16)
    hist = torch.zeros(layers, d.CONV_DIM * d.CONV_TAPS, dtype=torch.bfloat16, device=DEV)
    ptrs, strides = _ptrs(convs), _strides(convs[0])

    vk.conv_gather(ptrs, strides, slots, acc, hist)
    for i, c in enumerate(convs):
        sd = c if not dim_first else c.transpose(1, 2)  # [slot, entry, ch]
        want = sd[int(slots[i, 0]), accepted - 1 : accepted + 2].T  # [ch, 3]
        assert torch.equal(hist[i].view(d.CONV_DIM, 4)[:, 1:], want)

    ref = [c.clone() for c in convs]
    vk.conv_commit(ptrs, strides, slots, hist, rawrows, WINDOW)
    weight = torch.randn(d.CONV_DIM, d.CONV_TAPS, device=DEV).to(torch.bfloat16)
    qsl = torch.tensor([0, ROWS], dtype=torch.int32, device=DEV)
    for i in range(layers):
        x = rawrows[i].view(ROWS, d.PROJECTED)[:, : d.CONV_DIM].contiguous()
        state = ref[i] if dim_first else ref[i].transpose(-1, -2)  # (..., dim, state_len)
        causal_conv1d_update(x, state, weight, None, "silu", conv_state_indices=slots[i, :1],
                             num_accepted_tokens=acc, query_start_loc=qsl, max_query_len=ROWS)
        assert torch.equal(convs[i], ref[i]), f"layer {i}: conv cache differs from vLLM's update"


@pytest.mark.parametrize("prev,n", [(1, 1), (1, 8), (4, 4), (8, 2)])
def test_state_commit(prev, n):
    layers, nslots = 2, 12
    s_slot = d.V_HEADS * d.GDN_DIM * d.GDN_DIM + 64
    pool = torch.randn(layers, nslots, s_slot, device=DEV)
    states = [pool[i, :, : s_slot - 64].view(nslots, d.V_HEADS, d.GDN_DIM, d.GDN_DIM) for i in range(layers)]
    slots = torch.stack([torch.randperm(nslots, device=DEV)[:ROWS] for _ in range(layers)]).to(torch.int32)
    src = slots[:, prev - 1].contiguous()
    rec = torch.randn(layers, ROWS * d.V_HEADS * d.RECORD_FLOATS, device=DEV)
    acc = torch.tensor([n], dtype=torch.int32, device=DEV)
    before = pool.clone()
    vk.state_commit(_ptrs(states), s_slot, src, slots, rec, acc, ROWS)
    for i in range(layers):
        st = before[i, int(src[i]), : s_slot - 64].view(d.V_HEADS, d.GDN_DIM, d.GDN_DIM).clone()
        r = rec[i].view(ROWS, d.V_HEADS, d.RECORD_FLOATS)
        for t in range(n):
            st = r[t, :, 256, None, None] * st + r[t, :, :128, None] * r[t, :, None, 128:256]
        dst = int(slots[i, n - 1])
        torch.testing.assert_close(states[i][dst], st, rtol=1e-5, atol=1e-5)
        other = [s for s in range(nslots) if s != dst]
        assert torch.equal(pool[i, other], before[i, other])


def test_kv_import_export():
    nblocks, bs = 40, 48
    kvs = [torch.randn(nblocks, d.KV_HEADS, bs, 2 * d.HEAD_DIM, device=DEV).to(torch.bfloat16)
           for _ in d.ATTENTION_LAYERS]
    k_views = [kv.transpose(1, 2)[..., : d.HEAD_DIM] for kv in kvs]
    v_views = [kv.transpose(1, 2)[..., d.HEAD_DIM :] for kv in kvs]
    strides = k_views[0].stride()[:3]
    nA = len(d.ATTENTION_LAYERS)
    block_rows = torch.stack([torch.randperm(nblocks, device=DEV) for _ in range(nA)]).to(torch.int32)
    pages = 40
    mirror = [torch.zeros(pages * d.PAGE_ELEMENTS, dtype=torch.bfloat16, device=DEV) for _ in d.ATTENTION_LAYERS]
    mptrs = _ptrs(mirror)
    start, end = 37, 1000
    s_t = torch.tensor([start], dtype=torch.int32, device=DEV)
    e_t = torch.tensor([end], dtype=torch.int32, device=DEV)
    vk.kv_import(_ptrs(k_views), _ptrs(v_views), strides, block_rows, bs, s_t, e_t, mptrs)
    pos = torch.arange(start, end, device=DEV)
    for kview, vview, m, row in zip(k_views, v_views, mirror, block_rows):
        blk = row[pos // bs].long()
        mp = m.view(pages, d.KV_HEADS, 2, d.PAGE_TOKENS, d.HEAD_DIM)
        assert torch.equal(mp[pos // 32, :, 0, pos % 32], kview[blk, pos % bs])
        assert torch.equal(mp[pos // 32, :, 1, pos % 32], vview[blk, pos % bs])
    # export the first 5 of 8 rows at base 1000 back into the paged cache
    for m in mirror:
        m.view(pages, d.KV_HEADS, 2, d.PAGE_TOKENS, d.HEAD_DIM)[31:32].normal_()
    before = [kv.clone() for kv in kvs]
    base = torch.tensor([1000], dtype=torch.int32, device=DEV)
    rows = torch.tensor([5], dtype=torch.int32, device=DEV)
    vk.kv_export(_ptrs(k_views), _ptrs(v_views), strides, block_rows, bs, base, rows, mptrs, ROWS)
    pos = torch.arange(1000, 1005, device=DEV)
    for kv, b, kview, vview, m, row in zip(kvs, before, k_views, v_views, mirror, block_rows):
        blk = row[pos // bs].long()
        mp = m.view(pages, d.KV_HEADS, 2, d.PAGE_TOKENS, d.HEAD_DIM)
        assert torch.equal(kview[blk, pos % bs], mp[pos // 32, :, 0, pos % 32])
        assert torch.equal(vview[blk, pos % bs], mp[pos // 32, :, 1, pos % 32])
        # nothing but those 5 positions changed
        expected = torch.zeros(nblocks, bs, dtype=torch.bool, device=DEV)
        expected[blk, pos % bs] = True
        assert torch.equal((kv != b).any(-1).any(1), expected)


class _Binding:
    """The fields ``launch_begin`` touches."""

    def __init__(self, rows):
        self.rows = rows
        self.table = torch.zeros(d.TABLE_ROWS, d.TABLE_CELLS, dtype=torch.int64, device=DEV)
        self.endpoints = {"token_ids": torch.zeros(rows, dtype=torch.int64, device=DEV)}
        self.positions = torch.zeros(rows, dtype=torch.int32, device=DEV)
        self.context_blocks = torch.zeros(rows, dtype=torch.int32, device=DEV)
        self.block_map = torch.arange(64, dtype=torch.int32, device=DEV)
        self.native = torch.ones(d.NATIVE_WORDS + 128, dtype=torch.int32, device=DEV)
        self.controls = torch.ones(d.LAYERS, d.CONTROL_WORDS, dtype=torch.int32, device=DEV)
        self.command = torch.zeros(1, dtype=torch.int32, device=DEV)


class _Inp:
    def __init__(self):
        self.token_ids = torch.arange(ROWS, dtype=torch.int64, device=DEV) + 100
        self.base = torch.tensor([77], dtype=torch.int32, device=DEV)
        self.state_slot = torch.zeros(len(d.DELTA_LAYERS), dtype=torch.int32, device=DEV)
        self.request = torch.zeros(1, dtype=torch.int32, device=DEV)


def test_launch_begin():
    b, inp = _Binding(ROWS), _Inp()
    state_ptrs = torch.arange(len(d.DELTA_LAYERS), dtype=torch.int64, device=DEV) * 10_000_000 + 4096
    nL = len(d.DELTA_LAYERS)
    slots = (torch.arange(ROWS, device=DEV)[None, :] + 5 + 100 * torch.arange(nL, device=DEV)[:, None]).to(torch.int32)
    acc = torch.tensor([3], dtype=torch.int32, device=DEV)
    delta_rows = torch.tensor(d.DELTA_LAYERS, dtype=torch.int64, device=DEV)
    col = d.DELTA_SLOTS.index("initial_state")
    mirror_slot = torch.zeros(1, dtype=torch.int32, device=DEV)
    mirror_len = torch.tensor([50], dtype=torch.int32, device=DEV)
    start, end = torch.zeros_like(mirror_len), torch.zeros_like(mirror_len)
    vk.launch_begin(b, delta_rows, col, state_ptrs, 1000, slots, acc, inp, mirror_slot, mirror_len, start, end)
    assert torch.equal(inp.state_slot, slots[:, 2])
    assert torch.equal(b.table[delta_rows, col], state_ptrs + slots[:, 2].long() * 1000)
    assert (int(start), int(end)) == (50, 77)
    assert torch.equal(b.endpoints["token_ids"], inp.token_ids)
    assert torch.equal(b.positions, torch.arange(77, 77 + ROWS, dtype=torch.int32, device=DEV))
    assert int(b.native[d.COMMAND_WORD]) == 1 and int(b.controls.abs().sum()) == 0


def test_stage_rows():
    src = torch.arange(10, 30, dtype=torch.int64, device=DEV)
    dst = torch.zeros(6, 8, dtype=torch.int32, device=DEV)
    layers = torch.tensor([1, 4], dtype=torch.int64, device=DEV)
    vk.stage_rows(src, dst, layers)
    want = torch.zeros_like(dst)
    want[[1, 4]] = torch.arange(10, 18, dtype=torch.int32, device=DEV)
    assert torch.equal(dst, want)
