# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which requests the megakernel may serve: plain greedy only. Anything that changes the logits or the token choice
(penalties, logit bias, banned / allowed tokens, a minimum length, min-p, logits-processor arguments, a thinking
budget, watermarking), and any engine-wide logits processor or watermark, leaves the request to vLLM's own path.
CPU only."""

import types

import pytest

pytest.importorskip("torch")
SamplingParams = pytest.importorskip("vllm").SamplingParams

from aitermk_vllm_plugin.dflash_megakernel import _engine_sampling_ok, _sampling_ok  # noqa: E402


def greedy(**kw):
    return SamplingParams(temperature=0.0, max_tokens=16, **kw)


def test_plain_greedy_is_eligible():
    assert _sampling_ok(greedy())
    # stop conditions and output options do not change the token choice
    assert _sampling_ok(greedy(stop=["\n\n"], ignore_eos=True, seed=3, skip_special_tokens=False))


@pytest.mark.parametrize(
    "kw",
    [
        dict(repetition_penalty=1.2),
        dict(frequency_penalty=0.5),
        dict(presence_penalty=-0.5),
        dict(logit_bias={11: -100.0}),
        dict(bad_words=["the"]),
        dict(allowed_token_ids=[1, 2, 3]),
        dict(min_tokens=4),
        dict(extra_args={"my_logits_processor": 1}),
        dict(thinking_token_budget=64),
        dict(logprobs=1),
        dict(prompt_logprobs=1),
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_falls_back(kw):
    assert not _sampling_ok(greedy(**kw))


def test_sampled_falls_back():
    assert not _sampling_ok(SamplingParams(temperature=0.7))
    assert not _sampling_ok(SamplingParams(temperature=0.7, n=2))
    assert not _sampling_ok(None)


def test_greedy_drops_min_p():
    # vLLM resets min-p (and top-p / top-k) for greedy requests: argmax is unchanged by them
    sp = greedy(min_p=0.1)
    assert sp.min_p == 0.0 and _sampling_ok(sp)
    assert not _sampling_ok(types.SimpleNamespace(**{**vars_of(greedy()), "min_p": 0.1}))


def vars_of(sp):
    return {k: getattr(sp, k) for k in ("temperature", "n", "logprobs", "prompt_logprobs", "structured_outputs")}


def engine(processors=None, watermark=None):
    return types.SimpleNamespace(model_config=types.SimpleNamespace(logits_processors=processors),
                                 watermark_config=watermark)


def test_engine_wide():
    assert _engine_sampling_ok(engine())
    assert _engine_sampling_ok(engine(processors=[]))
    assert not _engine_sampling_ok(engine(processors=["my.module:Processor"]))
    assert not _engine_sampling_ok(engine(watermark=object()))
