# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""SGLang general plugin (entry point group ``sglang.srt.plugins``): Qwen3.8-27B DFLASH rounds on the AITerMK megakernel.

Installed with this package and enabled by ``SGLANG_DFLASH_MEGAKERNEL_AOT=<DFlash2Mono export dir>``; stock SGLang
otherwise. ``register`` wraps three methods of SGLang's ``DFlashWorkerV2``:

- ``__init__``: attach a ``DFlashMegakernel`` (built lazily on first use, once the pools exist);
- ``_append_target_hidden_to_draft_kv_by_loc``: during a prefill, also hand the prompt's DFlash capture to the
  megakernel drafter;
- ``forward_batch_generation``: a decode step of one eligible request runs the megakernel round; every other step
  runs SGLang's own path unchanged.
"""

import logging

import torch

from aitermk_sglang.dflash_megakernel import DFlashMegakernel, megakernel_requested

logger = logging.getLogger(__name__)

_ATTR = "_aitermk_megakernel"
_PREFILL_BATCH = "_aitermk_prefill_batch"


def register() -> None:
    if not megakernel_requested():
        return
    from sglang.srt.speculative import dflash_worker_v2 as w

    cls = w.DFlashWorkerV2
    if getattr(cls, "_aitermk_patched", False):
        return
    _wrap_init(cls)
    _wrap_append(cls)
    _wrap_forward(cls, w)
    cls._aitermk_patched = True
    logger.info("AITerMK megakernel DFLASH plugin registered")


def _wrap_init(cls) -> None:
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        setattr(self, _ATTR, DFlashMegakernel(
            server_args=self.server_args, target_worker=self._target_worker, block_size=int(self.block_size)))
        setattr(self, _PREFILL_BATCH, None)

    cls.__init__ = __init__


def _wrap_append(cls) -> None:
    orig = cls._append_target_hidden_to_draft_kv_by_loc

    def _append_target_hidden_to_draft_kv_by_loc(self, *, target_hidden, cache_loc, positions, **kwargs):
        orig(self, target_hidden=target_hidden, cache_loc=cache_loc, positions=positions, **kwargs)
        batch = getattr(self, _PREFILL_BATCH)
        if batch is not None and "commit_lens" not in kwargs:
            getattr(self, _ATTR).on_prefill(batch=batch, positions=positions, aux=target_hidden)

    cls._append_target_hidden_to_draft_kv_by_loc = _append_target_hidden_to_draft_kv_by_loc


def _wrap_forward(cls, w) -> None:
    orig = cls.forward_batch_generation

    def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, **kwargs):
        # (``kwargs``: arguments newer SGLang passes through, e.g. ``pp_proxy_tensors``)
        mk = getattr(self, _ATTR)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            setattr(self, _PREFILL_BATCH, batch)
            try:
                return orig(self, batch, on_publish=on_publish, grammar_barrier=grammar_barrier, **kwargs)
            finally:
                setattr(self, _PREFILL_BATCH, None)
        if (
            not batch.forward_mode.is_idle()
            and isinstance(batch.spec_info, w.DFlashDraftInputV2)
            and mk.eligible(batch, all_greedy=w._is_all_greedy(batch.sampling_info))
        ):
            return _megakernel_round(self, w, batch, on_publish)
        if not batch.forward_mode.is_idle():
            mk.note_regular_decode(batch)
        return orig(self, batch, on_publish=on_publish, grammar_barrier=grammar_barrier, **kwargs)

    cls.forward_batch_generation = forward_batch_generation


def _megakernel_round(worker, w, batch, on_publish):
    """One megakernel round, returning what SGLang's own DFLASH decode step returns."""
    mk = getattr(worker, _ATTR)
    block_size = int(worker.block_size)
    batch.seq_lens.record_stream(torch.get_device_module(worker.device).current_stream())
    prefix_lens = batch.seq_lens
    verify_out_cache_loc = w.assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=worker.model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + block_size,
        batch_size=1,
        draft_token_num=block_size,
        device=worker.device,
    )
    check = _regular_verify_logits(worker, w, batch, verify_out_cache_loc) if mk.wants_check() else None
    out_tokens, commit_lens, bonus = mk.round(
        batch=batch,
        bonus_tokens=batch.spec_info.bonus_tokens,
        verify_out_cache_loc=verify_out_cache_loc,
        compute_accept=w.compute_dflash_correct_drafts_and_bonus,
        commit_accept=w._commit_accept,
        check=check,
    )
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    if on_publish is not None:
        on_publish(new_seq_lens)
    mk.finish()
    return w.GenerationBatchResult(
        logits_output=DFlashMegakernel.empty_logits_output(),
        next_token_ids=out_tokens.reshape(-1),
        accept_lens=commit_lens,
        can_run_cuda_graph=False,
        next_draft_input=worker._make_next_draft_input_decode(bonus_tokens=bonus, new_seq_lens=new_seq_lens),
        speculative_num_draft_tokens=block_size,
        new_seq_lens=new_seq_lens,
    )


def _regular_verify_logits(worker, w, batch, verify_out_cache_loc):
    """Check mode: a callable running SGLang's own target verify of given block tokens, returning its
    [block, vocab] logits. It writes the block's KV / intermediate states only, which the megakernel's
    commit then overwrites."""
    block_size = int(worker.block_size)

    def run(block_tokens):
        positions = batch.seq_lens[:1].unsqueeze(1) + worker._block_pos_offsets
        verify_input = w.DFlashVerifyInput(
            draft_token=block_tokens.reshape(-1),
            positions=positions.reshape(-1),
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=w.CaptureHiddenMode.NULL,
        )
        saved = (batch.out_cache_loc, batch.seq_lens_cpu, batch.seq_lens_sum)
        batch.out_cache_loc = verify_out_cache_loc
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = batch.seq_lens_cpu + block_size
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        fb, _ = verify_input.prepare_for_verify(batch, worker.target_worker)
        batch.out_cache_loc, batch.seq_lens_cpu, batch.seq_lens_sum = saved
        out = worker.target_worker.forward_batch_generation(
            batch=None, forward_batch=fb, is_verify=True, skip_attn_backend_init=True)
        return out.logits_output.next_token_logits.view(block_size, -1)

    return run
