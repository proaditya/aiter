# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DFLASH single-sequence rounds on the AITerMK Qwen3.8-27B megakernel (ROCm, gfx950).

Opt-in: ``SGLANG_DFLASH_MEGAKERNEL_AOT=<dir>`` (the DFlash2Mono code objects + manifest) with the
``aitermk_qwen38`` package importable. A decode step of exactly one plain greedy request (no grammar, logprobs,
hidden-state returns, penalties, logit bias or other logit changes: ``plain_greedy``) then runs the megakernel's
draft + verify + commit; everything else stays on the regular DFLASH path. The megakernel reads and commits
SGLang's pools in place, so either path can take the next step except that the megakernel drafter keeps its own
context: a request it did not see from prefill on is left to the regular path.
"""

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

MEGAKERNEL_AOT_ENV = "SGLANG_DFLASH_MEGAKERNEL_AOT"
# check mode: the first N megakernel rounds also run SGLang's own verify of the same block and log both
MEGAKERNEL_CHECK_ENV = "SGLANG_DFLASH_MEGAKERNEL_CHECK"


def megakernel_requested() -> bool:
    return bool(os.environ.get(MEGAKERNEL_AOT_ENV))


def plain_greedy(req, sampling_info) -> bool:
    """Nothing changes the request's logits or token choice beyond its argmax: no penalty (repetition /
    frequency / presence, a minimum length), logit bias, min-p, custom logit processor or watermark. The
    megakernel round accepts drafts against its own argmax of the raw logits, so any of them would be ignored;
    such a request stays on SGLang's own path, which applies them. Host values only."""
    sp = req.sampling_params
    if (
        getattr(sp, "repetition_penalty", 1.0) != 1.0
        or getattr(sp, "frequency_penalty", 0.0) != 0.0
        or getattr(sp, "presence_penalty", 0.0) != 0.0
        or getattr(sp, "min_new_tokens", 0)
        or getattr(sp, "min_p", 0.0)
        or getattr(sp, "logit_bias", None)
        or getattr(sp, "watermark", None)
        or getattr(req, "custom_logit_processor", None) is not None
    ):
        return False
    # and the batch's own flags, where this SGLang has them
    penalizers = getattr(sampling_info, "penalizer_orchestrator", None)
    return not (
        (penalizers is not None and getattr(penalizers, "is_required", False))
        or getattr(sampling_info, "logit_bias", None) is not None
        or getattr(sampling_info, "has_custom_logit_processor", False)
        or getattr(sampling_info, "need_min_p_sampling", False)
        or getattr(sampling_info, "has_watermark_candidates", False)
    )


def req_pool_index(req) -> Optional[int]:
    """A request's ``req_to_token`` row (host value): ``Req.kv.req_pool_idx`` in SGLang main after its
    request-KV refactor (e.g. 59eb71a831, 2026-10-09), ``Req.req_pool_idx`` before it (e.g. f825d72936)."""
    # (older SGLang also has a ``Req.kv``, a different record: the request's own attribute wins)
    try:
        return req.req_pool_idx
    except AttributeError:
        return req.kv.req_pool_idx


class DFlashMegakernel:
    """Owns the AITerMK adapter for one DFLASH worker; built on first use (the pools exist by then)."""

    def __init__(self, *, server_args, target_worker, block_size: int) -> None:
        if block_size != 8:
            raise ValueError(f"megakernel DFLASH path needs block size 8, got {block_size}")
        self._server_args = server_args
        self._target_worker = target_worker
        self._adapter = None
        # the request whose context the megakernel drafter holds, from its prefill on
        self._context_req: Optional[int] = None
        self._context_len = 0
        self._checks_left = int(os.environ.get(MEGAKERNEL_CHECK_ENV, "0"))

    def wants_check(self) -> bool:
        return self._checks_left > 0

    def _get_adapter(self):
        if self._adapter is None:
            from aitermk_sglang.mk_dflash import MegakernelDFlash

            mr = self._target_worker.model_runner
            model = mr.model
            kv_pool = mr.token_to_kv_pool
            attention_layers = sorted(kv_pool.full_attention_layer_id_mapping)
            mamba = mr.req_to_token_pool.mamba_pool.mamba_cache
            if len(mamba.conv) != 1:
                raise ValueError(f"megakernel path: {len(mamba.conv)} conv state tensors, expected 1")
            self._adapter = MegakernelDFlash(
                hf_path=self._server_args.model_path,
                draft_path=self._server_args.speculative_draft_model_path,
                aot_dir=os.environ[MEGAKERNEL_AOT_ENV],
                max_model_len=mr.model_config.context_len,
                embed=model.get_input_embeddings().weight,
                lm_head=model.lm_head.weight,
                conv=mamba.conv[0],
                state=mamba.temporal,
                k_buffers=[kv_pool.get_key_buffer(i) for i in attention_layers],
                v_buffers=[kv_pool.get_value_buffer(i) for i in attention_layers],
            )
        return self._adapter

    def on_prefill(self, *, batch: ScheduleBatch, positions: torch.Tensor, aux: torch.Tensor) -> None:
        """A single request's prefill chunk: its target features into the megakernel drafter."""
        if len(batch.reqs) != 1:
            self._context_req = None
            return
        # host values only: reading the device tensors here would wait for the GPU
        req_index = req_pool_index(batch.reqs[0])
        start = int(batch.prefix_lens[0])
        continues = self._context_req == req_index and self._context_len == start
        if start != 0 and not continues:
            self._context_req = None  # a prefix the drafter did not see
            return
        self._context_req = req_index
        self._get_adapter().prefill_context(positions, aux, new_sequence=start == 0)
        self._context_len = start + int(batch.extend_lens[0])

    def note_regular_decode(self, batch: ScheduleBatch) -> None:
        """A decode step on the regular path: the drafter misses its rows, so it gives up that request."""
        if self._adapter is not None:
            self._adapter.invalidate_draft()
        if self._context_req is not None and any(req_pool_index(r) == self._context_req for r in batch.reqs):
            self._context_req = None

    def eligible(self, batch: ScheduleBatch, *, all_greedy: bool) -> bool:
        # host values only (``req_pool_index``): an ``int()`` of ``batch.req_pool_indices`` synchronized
        # every step with the previous round, which serialized SGLang's host work with the GPU
        return (
            len(batch.reqs) == 1
            and all_greedy
            and not batch.has_grammar
            and not batch.return_logprob
            and not any(r.return_hidden_states for r in batch.reqs)
            and plain_greedy(batch.reqs[0], batch.sampling_info)
            and self._context_req is not None
            and req_pool_index(batch.reqs[0]) == self._context_req
        )

    def round(self, *, batch: ScheduleBatch, bonus_tokens: torch.Tensor, verify_out_cache_loc: torch.Tensor,
              compute_accept, commit_accept, check=None):
        """Draft + verify + commit of one request; returns (out_tokens, commit_lens, bonus)."""
        req_pool = self._target_worker.model_runner.req_to_token_pool
        adapter = self._get_adapter()
        req_index = batch.req_pool_indices[:1]
        args = dict(
            anchor=bonus_tokens.view(1),
            prefix_len=batch.seq_lens[:1],
            mamba_slot=req_pool.get_mamba_indices(req_index).to(torch.int32),
            req_to_token=req_pool.req_to_token,
            req_index=req_index,
        )
        if check is None:
            return adapter.full_round(out_slots=verify_out_cache_loc, **args)
        adapter.invalidate_draft()
        candidates, target_predict = adapter.round(**args)
        self._log_check(adapter, candidates, target_predict, check(candidates))
        accept_len, bonus = compute_accept(candidates=candidates, target_predict=target_predict)
        out_tokens, commit_lens = commit_accept(candidates, accept_len, bonus)
        adapter.commit(commit_len=commit_lens, out_slots=verify_out_cache_loc)
        return out_tokens, commit_lens, bonus

    def finish(self) -> None:
        """The rest of the round ``round`` began (commit, context, the next block's draft): after SGLang's
        publish of the new length, so the scheduler's wait on it does not cover this work."""
        if self._adapter is not None:
            self._adapter.finish_round()

    def _log_check(self, adapter, candidates, target_predict, ref_logits) -> None:
        self._checks_left -= 1
        mk_logits = adapter.mono.binding.logits.float()
        ref = ref_logits.float()
        ref_top = ref.argmax(-1)
        gap = (ref.gather(-1, ref_top[:, None]) - ref.gather(-1, target_predict.view(-1, 1))).view(-1)
        logger.info(
            "megakernel check: block %s | mk %s | ref %s | ref logit gap of mk pick %s | max |dlogit| %.3f, rel %.4f",
            candidates.view(-1).tolist(), target_predict.view(-1).tolist(), ref_top.tolist(),
            [round(float(x), 3) for x in gap], float((mk_logits - ref).abs().max()),
            float((mk_logits - ref).norm() / ref.norm()),
        )

    @staticmethod
    def empty_logits_output() -> LogitsProcessorOutput:
        return LogitsProcessorOutput(next_token_logits=None)
