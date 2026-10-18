# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DFlash 2 single-sequence rounds on the AITerMK Qwen3.8-27B megakernel inside vLLM (ROCm, gfx950).

Opt-in: ``AITERMK_VLLM_DFLASH_AOT=<dir>`` (the DFlash2Mono code objects + manifest) with the ``aitermk_qwen38``
package importable. A speculative step of exactly one plain greedy request (``_sampling_ok``) whose
drafter context the megakernel drafter holds then runs the megakernel's verify (vLLM samples from its logits)
and the megakernel drafter's commit + context + draft; everything else stays on vLLM's own path. The
megakernel reads and commits vLLM's caches in place, so vLLM's path can take any later step; the megakernel
drafter keeps its own context, so a request it did not see from its first prefill chunk on, or that took a step
on vLLM's path, is left to vLLM.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

AOT_ENV = "AITERMK_VLLM_DFLASH_AOT"
# check mode: the first N megakernel rounds also run vLLM's own target forward of the same block and log both
CHECK_ENV = "AITERMK_VLLM_DFLASH_CHECK"
ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
VOCAB = 248320


def megakernel_requested() -> bool:
    return bool(os.environ.get(AOT_ENV))


def _local_path(name: str, revision: str | None = None) -> str:
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(name, revision=revision, local_files_only=True)


def _sampling_ok(sp) -> bool:
    """A request the megakernel may serve: plain greedy, one sequence, no logprobs, no structured output, and
    nothing that changes the logits or the token choice (penalties, logit bias, banned / allowed tokens, a
    minimum length, min-p, per-request logits-processor arguments, a thinking budget, watermarking). The rest
    stay on vLLM's own path, which applies them."""
    if sp is None:
        return False
    return (
        sp.temperature < 1e-5
        and getattr(sp, "n", 1) == 1
        and sp.logprobs is None
        and sp.prompt_logprobs is None
        and getattr(sp, "structured_outputs", None) is None
        and getattr(sp, "repetition_penalty", 1.0) == 1.0
        and getattr(sp, "frequency_penalty", 0.0) == 0.0
        and getattr(sp, "presence_penalty", 0.0) == 0.0
        and not getattr(sp, "logit_bias", None)
        and not getattr(sp, "bad_words", None)
        and not getattr(sp, "_bad_words_token_ids", None)
        and getattr(sp, "allowed_token_ids", None) is None
        and not getattr(sp, "min_tokens", 0)
        and not getattr(sp, "min_p", 0.0)
        and not getattr(sp, "extra_args", None)  # where custom logits processors read their arguments
        and getattr(sp, "thinking_token_budget", None) is None
        and not getattr(sp, "watermarking", None)
    )


def _engine_sampling_ok(cfg) -> bool:
    """No engine-wide logits processor or watermark: either would act on every request's logits."""
    return not getattr(cfg.model_config, "logits_processors", None) and getattr(cfg, "watermark_config", None) is None


class DFlashMegakernel:
    """Owns the megakernel path of one vLLM (Model Runner V2) GPU model runner."""

    def __init__(self, runner) -> None:
        from aitermk_vllm_plugin.mk_dflash import MegakernelDFlash

        self.runner = runner
        cfg = runner.vllm_config
        spec = cfg.speculative_config
        mc = cfg.model_config
        refusal = []
        if spec is None or spec.method != "dflash" or spec.num_speculative_tokens != 7:
            refusal.append("--speculative-config method dflash with num_speculative_tokens 7")
        if not set(ARCHITECTURES) & set(mc.hf_config.architectures or ()):
            refusal.append(f"a Qwen3.8-27B checkpoint ({mc.hf_config.architectures})")
        if cfg.parallel_config.tensor_parallel_size != 1 or cfg.parallel_config.pipeline_parallel_size != 1:
            refusal.append("TP1 / PP1")
        if cfg.cache_config.mamba_cache_mode != "none":
            refusal.append(f"mamba_cache_mode none (got {cfg.cache_config.mamba_cache_mode}: --no-enable-prefix-caching)")
        if cfg.cache_config.mamba_ssm_cache_dtype != "float32":
            refusal.append("--mamba-ssm-cache-dtype float32")
        if refusal:
            raise ValueError("AITerMK megakernel DFlash path needs " + "; ".join(refusal))

        model = runner.get_model()
        lm = model.language_model if hasattr(model, "language_model") else model
        self.layers = lm.model.layers
        if len(self.layers) != 64:
            raise ValueError(f"{len(self.layers)} layers: not Qwen3.8-27B")
        self.attn = [self.layers[i].self_attn.attn for i in range(64) if i % 4 == 3]
        self.gdn = [self.layers[i].linear_attn for i in range(64) if i % 4 != 3]
        self.attn_name = self.attn[0].layer_name
        self.gdn_name = self.gdn[0].prefix
        self.adapter = MegakernelDFlash(
            hf_path=_local_path(mc.model, mc.revision),
            draft_path=_local_path(spec.draft_model_config.model, spec.draft_model_config.revision),
            aot_dir=os.environ[AOT_ENV],
            max_model_len=mc.max_model_len,
            embed=lm.model.embed_tokens.weight[:VOCAB],
            lm_head=lm.lm_head.weight[:VOCAB],
        )
        self._checks_left = int(os.environ.get(CHECK_ENV, "0"))
        self._engine_sampling_ok = _engine_sampling_ok(cfg)
        if not self._engine_sampling_ok:
            logger.info("AITerMK megakernel: engine-wide logits processors or watermarking; every request "
                        "stays on vLLM's own path")
        # per request: eligible sampling, prompt length
        self._requests: dict[str, tuple[bool, int]] = {}
        # the request whose context the megakernel drafter holds, from its first prefill chunk on
        self._context_req: str | None = None
        self._context_len = 0
        # this step: None, "prefill" (feed the drafter, vLLM runs it) or "round" (the megakernel runs it)
        self.mode: str | None = None
        self._new_sequence = False
        self._input_batch = None
        self._attn_metadata = None
        self._logits = None
        self._in_forward = False
        self._check = None
        self._dummy = None
        self._groups = None
        self.rounds = 0

    # ---- step routing (host values only) ----------------------------------------------------

    def begin_step(self, so) -> None:
        self.end_step()
        for r in so.scheduled_new_reqs:
            ok = self._engine_sampling_ok and _sampling_ok(r.sampling_params)
            self._requests[r.req_id] = (ok, len(r.prompt_token_ids or ()))
        for rid in so.finished_req_ids:
            self._requests.pop(rid, None)
            if rid == self._context_req:
                self._context_req = None
        sched = so.num_scheduled_tokens
        if len(sched) != 1:
            if self._context_req in sched:
                self._context_req = None
            return
        (rid, n), = sched.items()
        ok, prompt_len = self._requests.get(rid, (False, 0))
        drafts = so.scheduled_spec_decode_tokens.get(rid) or ()
        computed = self._num_computed(so, rid)
        if drafts:
            if (ok and rid == self._context_req and len(drafts) == self.adapter.rows - 1 and n == self.adapter.rows
                    and not getattr(so, "has_structured_output_requests", False)):
                self.mode = "round"
            elif rid == self._context_req:
                self._context_req = None  # a step the drafter misses
            return
        if computed is not None and computed < prompt_len and ok:
            if computed == 0:
                self._context_req, self._context_len, self._new_sequence = rid, 0, True
            elif rid != self._context_req or computed != self._context_len:
                self._context_req = None  # a prefix the drafter did not see
                return
            else:
                self._new_sequence = False
            self._context_len = computed + n
            self.mode = "prefill"
            return
        if rid == self._context_req:
            self._context_req = None

    @staticmethod
    def _num_computed(so, rid) -> int | None:
        for r in so.scheduled_new_reqs:
            if r.req_id == rid:
                return r.num_computed_tokens
        cached = so.scheduled_cached_reqs
        if rid in cached.req_ids:
            return cached.num_computed_tokens[cached.req_ids.index(rid)]
        return None

    def end_step(self) -> None:
        self.mode = None
        self._input_batch = self._attn_metadata = self._logits = self._check = None

    def note_input_batch(self, input_batch) -> None:
        if self.mode == "round":
            self._input_batch = input_batch

    def note_attn_metadata(self, attn_metadata) -> None:
        if self.mode == "round":
            self._attn_metadata = attn_metadata

    # ---- the round ----------------------------------------------------------------------------

    def forward(self, stock):
        """In place of vLLM's target forward on a round: the megakernel's verify. ``stock`` runs vLLM's own
        forward (check mode). Returns what the target forward returns: (hidden, aux) placeholders."""
        if self._in_forward or self.mode != "round":
            return stock()
        self._in_forward = True
        try:
            return self._forward(stock)
        finally:
            self._in_forward = False

    def _forward(self, stock):
        ib, md = self._input_batch, self._attn_metadata
        if ib is None or md is None:
            raise RuntimeError("megakernel round without vLLM's step inputs")
        self._ensure_bound()
        gdn, att = md[self.gdn_name], md[self.attn_name]
        if gdn.num_spec_decodes != 1 or gdn.num_prefills or gdn.num_decodes:
            raise RuntimeError(f"megakernel round on a step vLLM did not plan as one speculative decode: "
                               f"{gdn.num_spec_decodes} / {gdn.num_prefills} / {gdn.num_decodes}")
        if self._groups is None:
            self._groups = self._kv_groups(md)
        slot_groups = [(md[name].spec_state_indices_tensor[0], layers) for name, layers in self._groups[0]]
        block_groups = [(md[name].block_table[0], layers) for name, layers in self._groups[1]]
        self._logits = self.adapter.head(token_ids=ib.input_ids, positions=ib.positions,
                                         accepted=gdn.num_accepted_tokens, slot_groups=slot_groups,
                                         block_groups=block_groups)
        self.rounds += 1
        if self._checks_left > 0:
            self._checks_left -= 1
            return self._stock_forward(stock)
        n = ib.num_tokens_after_padding
        if self._dummy is None or self._dummy.shape[0] < n:
            self._dummy = torch.zeros(max(n, 64), 5120, dtype=torch.bfloat16, device=self.adapter.device)
        hidden = self._dummy[:n]
        return (hidden, [hidden] * 5) if self.runner.use_aux_hidden_state_outputs else hidden

    def _kv_groups(self, md):
        """The layers of each of vLLM's KV-cache groups (layers of a group share one metadata object), as
        (a layer name to read the group's rows from, its layer indices) for the DeltaNet and the attention
        layers."""
        out = []
        for names in ([g.prefix for g in self.gdn], [a.layer_name for a in self.attn]):
            groups: dict[int, tuple[str, list[int]]] = {}
            for i, name in enumerate(names):
                groups.setdefault(id(md[name]), (name, []))[1].append(i)
            out.append([(name, torch.tensor(layers, dtype=torch.int64, device=self.adapter.device))
                        for name, layers in groups.values()])
        logger.info("AITerMK megakernel: vLLM's KV-cache groups: %d of DeltaNet layers, %d of attention layers",
                    len(out[0]), len(out[1]))
        return out

    def _stock_forward(self, stock):
        """Check mode: vLLM's own forward of the same block. It writes the request's 8 recurrent state slots,
        so the slot each layer's commit replays from (the launch's ``state_slot``) is saved and restored."""
        src = self.adapter.mono.inp.state_slot.long()
        saved = [g.kv_cache[1].index_select(0, src[i:i + 1]) for i, g in enumerate(self.gdn)]
        out = stock()
        for i, (g, s) in enumerate(zip(self.gdn, saved)):
            g.kv_cache[1].index_copy_(0, src[i:i + 1], s)
        self._check = {"ids": self.adapter.s_ids.clone(), "aux": out[1] if isinstance(out, tuple) else None}
        return out

    def compute_logits(self, stock, hidden):
        """In place of vLLM's LM head on a round: the megakernel's logits (consumed once, by the sampler)."""
        logits, self._logits = self._logits, None
        if logits is None:
            return stock(hidden)
        if self._check is not None:
            self._log_check(logits[: hidden.shape[0]], stock(hidden).float())
        return logits[: hidden.shape[0]]

    def propose(self, stock, kwargs):
        """In place of vLLM's DFlash 2 drafter: the round's tail on a round; the prompt's features into the
        megakernel drafter on a prefill step (vLLM's drafter drafts the first block)."""
        if self.mode == "round":
            ib = kwargs["input_batch"]
            anchor = kwargs["last_sampled"].index_select(0, ib.idx_mapping[:1].long())
            drafts = self.adapter.tail(num_sampled=kwargs["num_sampled"], anchor=anchor)
            if self._check is not None and self._check["aux"] is not None:
                self._log_aux(self._check["aux"])
            return drafts
        if self.mode == "prefill":
            ib = kwargs["input_batch"]
            aux = kwargs["aux_hidden_states"]
            t = ib.num_tokens
            feats = torch.stack([h[:t] for h in aux], 1)
            self.adapter.prefill_context(ib.positions[:t], feats, new_sequence=self._new_sequence)
        return stock()

    # ---- caches -----------------------------------------------------------------------------

    def _ensure_bound(self) -> None:
        if self.adapter.bound:
            return
        k_views, v_views = [], []
        for a in self.attn:
            backend = a.attn_backend.get_name() if hasattr(a, "attn_backend") else a.get_attn_backend().get_name()
            kv = a.kv_cache
            if backend != "TRITON_ATTN" or kv.dim() != 4 or kv.shape[1] != 4 or kv.shape[3] != 512:
                raise ValueError(f"megakernel path reads the KV cache of --attention-backend TRITON_ATTN "
                                 f"((blocks, 4, block_size, 512) bf16); got {backend} {tuple(kv.shape)}")
            t = kv.transpose(1, 2)
            k_views.append(t[..., :256])
            v_views.append(t[..., 256:])
        conv = [g.kv_cache[0] for g in self.gdn]
        ssm = [g.kv_cache[1] for g in self.gdn]
        self.adapter.bind(k_views=k_views, v_views=v_views, conv_states=conv, ssm_states=ssm)

    # ---- check mode ---------------------------------------------------------------------------

    def _log_check(self, mk: torch.Tensor, ref: torch.Tensor) -> None:
        ids = self._check["ids"]
        mk_top, ref_top = mk.argmax(-1), ref.argmax(-1)
        gap = (ref.gather(-1, ref_top[:, None]) - ref.gather(-1, mk_top[:, None])).view(-1)
        logger.info(
            "megakernel check: block %s | mk %s | ref %s | ref logit gap of mk pick %s | max |dlogit| %.3f, rel %.4f",
            ids.tolist(), mk_top.tolist(), ref_top.tolist(), [round(float(x), 3) for x in gap],
            float((mk - ref).abs().max()), float((mk - ref).norm() / ref.norm()),
        )

    def _log_aux(self, aux) -> None:
        mk = self.adapter.aux().float()
        rel = [float((mk[:, i] - a[: mk.shape[0]].float()).norm() / a[: mk.shape[0]].float().norm())
               for i, a in enumerate(aux)]
        logger.info("megakernel check: aux streams (layers 5/19/33/47/61) rel diff vs vLLM %s",
                    [round(x, 4) for x in rel])
