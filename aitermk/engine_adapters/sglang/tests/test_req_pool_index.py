# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``req_pool_index`` reads a request's ``req_to_token`` row on SGLang before and after its request-KV refactor."""

import types

import pytest

pytest.importorskip("sglang")

from aitermk_sglang.dflash_megakernel import req_pool_index  # noqa: E402


def test_before_refactor():
    # f825d72936: the row on the request; ``Req.kv`` is a different record (no req_pool_idx)
    req = types.SimpleNamespace(req_pool_idx=7, kv=types.SimpleNamespace(kv_allocated_len=0))
    assert req_pool_index(req) == 7


def test_after_refactor():
    # 59eb71a831: the row moved into ``Req.kv``
    req = types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=3))
    assert req_pool_index(req) == 3
