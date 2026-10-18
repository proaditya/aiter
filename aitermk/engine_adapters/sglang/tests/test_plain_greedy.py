# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which requests the megakernel may serve: plain greedy only. A penalty (repetition / frequency / presence, a
minimum length), logit bias, min-p, custom logit processor or watermark, on the request or in the batch's sampling
state, leaves the request to SGLang's own path. CPU only."""

import types

import pytest

pytest.importorskip("sglang")

from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

from aitermk_sglang.dflash_megakernel import plain_greedy  # noqa: E402


def req(custom_logit_processor=None, **kw):
    sp = SamplingParams(temperature=0.0, max_new_tokens=16, **kw)
    return types.SimpleNamespace(sampling_params=sp, custom_logit_processor=custom_logit_processor)


def batch(**kw):
    state = dict(penalizer_orchestrator=types.SimpleNamespace(is_required=False), logit_bias=None,
                 has_custom_logit_processor=False, need_min_p_sampling=False, has_watermark_candidates=False)
    state.update(kw)
    return types.SimpleNamespace(**state)


def test_plain_greedy_is_eligible():
    assert plain_greedy(req(), batch())
    assert plain_greedy(req(stop=["\n\n"], ignore_eos=True), batch())
    assert plain_greedy(req(), types.SimpleNamespace())  # an SGLang without the batch flags


@pytest.mark.parametrize(
    "kw",
    [
        dict(repetition_penalty=1.2),
        dict(frequency_penalty=0.5),
        dict(presence_penalty=-0.5),
        dict(min_new_tokens=4),
        dict(min_p=0.1),
        dict(logit_bias={"11": -100.0}),
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_request_falls_back(kw):
    assert not plain_greedy(req(**kw), batch())


def test_custom_logit_processor_falls_back():
    assert not plain_greedy(req(custom_logit_processor="{}"), batch())


@pytest.mark.parametrize(
    "kw",
    [
        dict(penalizer_orchestrator=types.SimpleNamespace(is_required=True)),
        dict(logit_bias=object()),
        dict(has_custom_logit_processor=True),
        dict(need_min_p_sampling=True),
        dict(has_watermark_candidates=True),
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_batch_state_falls_back(kw):
    assert not plain_greedy(req(), batch(**kw))
