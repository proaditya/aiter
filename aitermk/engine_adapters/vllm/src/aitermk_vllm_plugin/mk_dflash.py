# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""vLLM adapter: Qwen3.8-27B single-sequence DFlash 2 rounds on the AITerMK megakernel.

vLLM keeps everything it owns: scheduling, prefill (its BF16 model), the KV and GDN caches, sampling. A
speculative step of one greedy request becomes:

1. the head (in place of vLLM's target forward): the megakernel (MXFP4, one launch) verifies the 8 rows
   vLLM assembled (the previous round's bonus token and 7 drafts) over vLLM's caches; vLLM's rejection sampler
   samples from its logits;
2. the tail (in place of vLLM's DFlash 2 drafter): the accepted prefix is committed into vLLM's caches the
   way vLLM's own step leaves them, the 8 rows' target features go into the drafter's context, and
   DFlash2Mono (persistent FlyDSL drafter, MXFP8, MXFP4-screened LM head) drafts the next 7 tokens after the
   bonus token, which vLLM then schedules.

The prompt's context reaches the drafter from vLLM's prefill (its DFlash aux hidden states).
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger("aitermk_qwen38")


def _drafter_module():
    import aitermk_qwen38.provider

    path = str(Path(aitermk_qwen38.provider.__file__).resolve().parent / "drafter")
    if path not in sys.path:
        sys.path.insert(0, path)
    import dflash2_mono

    return dflash2_mono


class MegakernelDFlash:
    """One vLLM target + DFlash 2 draft checkpoint; serves one request at a time."""

    def __init__(self, *, hf_path: str, draft_path: str, aot_dir: str, max_model_len: int, embed: torch.Tensor,
                 lm_head: torch.Tensor, variant: str = "m8") -> None:
        from aitermk_qwen38.provider import weights as W
        from aitermk_qwen38.provider.vllm_engine import VllmMonoRound

        dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self.max_model_len = max_model_len
        weights = W.convert_model(W.safetensors_source(hf_path, dev))
        self.mono = VllmMonoRound(weights, max_model_len, variant=variant, device=dev)
        ps = json.loads((Path(aot_dir) / "manifest.json").read_text())["abi"]["Ps"]
        fit = [p for p in ps if p >= max_model_len + 8]
        if not fit:
            raise ValueError(f"DFlash2Mono exports max_pos {ps}; max_model_len {max_model_len} needs >= {max_model_len + 8}")
        self.drafter = _drafter_module().DFlash2Mono(
            draft_path, embed, lm_head, min(fit), prec=dict(layers="aot8", lm="mx4r128", ctx="aot8"), aot=aot_dir)
        self.rows = self.mono.rows
        # the round's inputs, staged from vLLM's step tensors into static buffers (graph inputs)
        self.s_ids = torch.zeros(self.rows, dtype=torch.int64, device=dev)
        self.s_base = torch.zeros(1, dtype=torch.int64, device=dev)
        # per layer (vLLM's KV-cache groups give layers different state slots and blocks)
        self.s_slots = torch.zeros(48, self.rows, dtype=torch.int32, device=dev)
        self.s_accepted = torch.zeros(1, dtype=torch.int32, device=dev)
        self.s_block_rows = None  # [16, blocks], sized at ``bind`` (vLLM's block size)
        self.s_sampled = torch.ones(1, dtype=torch.int32, device=dev)
        self.s_anchor = torch.zeros(1, dtype=torch.int64, device=dev)
        # the drafter's block: the anchor, then the 7 drafts
        self.ids = torch.zeros(self.rows, dtype=torch.int64, device=dev)
        self.drafts = torch.zeros(1, self.rows - 1, dtype=torch.int64, device=dev)
        self.ar = torch.arange(self.rows, dtype=torch.int64, device=dev)
        self.logits = None
        # a round is two graphs: the head (verify), whose logits vLLM samples, and the tail (commit, context,
        # the next round's draft), which runs after vLLM has started copying the step's tokens out
        self._graphs = {}
        self._warm = set()
        self._eager = os.environ.get("AITERMK_VLLM_EAGER") == "1"
        # diagnostics: propose token 0 every time, so each round keeps only the verified token -- the
        # megakernel's own greedy decode, the reference a speculative run must reproduce
        self._draft_off = os.environ.get("AITERMK_VLLM_DRAFT_OFF") == "1"
        logger.info("AITerMK megakernel DFlash 2 path ready (variant %s, drafter max_pos %d)", variant, min(fit))

    def bind(self, **caches) -> None:
        self.mono.bind(**caches)
        blocks = (self.max_model_len + self.rows + self.mono.block_size - 1) // self.mono.block_size
        self.s_block_rows = torch.zeros(16, blocks, dtype=torch.int32, device=self.device)

    @property
    def bound(self) -> bool:
        return self.mono.bound

    def prefill_context(self, positions: torch.Tensor, feats: torch.Tensor, *, new_sequence: bool) -> None:
        """A prefill chunk's target features ([n, 5, HIDDEN] bf16, vLLM's DFlash aux hidden states) into the
        drafter; ``new_sequence`` on its first chunk."""
        if new_sequence:
            self.mono.invalidate_mirror()
        n = positions.numel()
        for i in range(0, n, 1024):
            self.drafter.context(positions[i:i + 1024].long(), feats[i:i + 1024].contiguous())

    def head(self, *, token_ids, positions, accepted, slot_groups, block_groups) -> torch.Tensor:
        """The megakernel's verify of a step's 8 rows: ``token_ids`` / ``positions`` vLLM's inputs,
        ``accepted`` the request's last step's accepted count (int [1]); ``slot_groups`` (row, layers) per
        KV-cache group of DeltaNet layers: the request's 8 state slots there and the layers (int64, 0..47) it
        holds; ``block_groups`` the same for the attention layers (0..15) and the request's block-table row.
        Returns the logits [rows, VOCAB] fp32."""
        from aitermk_qwen38.provider.vllm_state_kernels import stage, stage_rows

        stage(token_ids, positions, accepted, self.s_ids, self.s_base, self.s_accepted)
        for row, layers in slot_groups:
            stage_rows(row, self.s_slots, layers)
        for row, layers in block_groups:
            stage_rows(row, self.s_block_rows, layers)
        self._run("head", self._head)
        return self.logits

    def tail(self, *, num_sampled: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """The rest of the round ``head`` began: commit the step's ``num_sampled`` rows (int [1]: accepted
        drafts + the bonus token), the rows' context, and the next block's draft after ``anchor`` (the bonus
        token, int [1]). Returns the drafts [1, 7] int64."""
        self.s_sampled.copy_(num_sampled[:1])
        self.s_anchor.copy_(anchor.view(-1)[:1])
        self._run("tail", self._tail)
        return self.drafts

    def aux(self) -> torch.Tensor:
        return self.mono.aux()

    # ---- graph pieces: every input in a static buffer, no host sync -------------------------

    def _run(self, name: str, body) -> None:
        """``body`` replayed from its graph after a warm-up call (eager with ``AITERMK_VLLM_EAGER=1``)."""
        graph = self._graphs.get(name)
        if graph is not None:
            graph.replay()
        elif self._eager or name not in self._warm:
            body()
            self._warm.add(name)
        else:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                body()
            torch.cuda.current_stream().wait_stream(stream)
            self._graphs[name] = graph
            logger.info("AITerMK megakernel DFlash 2 round %s captured", name)
            # the capture recorded the body without running it
            graph.replay()

    def _head(self) -> None:
        self.logits = self.mono.verify(self.s_ids, self.s_base, self.s_slots, self.s_accepted, self.s_block_rows)

    def _tail(self) -> None:
        self.mono.commit(self.s_sampled, self.s_slots, self.s_block_rows)
        self.drafter.context(self.s_base + self.ar, self.mono.aux())
        # the next block: its anchor is this round's bonus token, at the committed length
        self.s_base.add_(self.s_sampled.to(self.s_base.dtype))
        self.ids.zero_()
        self.ids[:1].copy_(self.s_anchor)
        # the mailbox tag is (base, step): a repeated base must not read an earlier draft's mailboxes
        self.drafter.step.add_(1)
        self.drafter.draft_into(self.ids, self.s_base)
        self.drafts.copy_(self.ids[1:].view(1, -1))
        if self._draft_off:
            self.drafts.zero_()
