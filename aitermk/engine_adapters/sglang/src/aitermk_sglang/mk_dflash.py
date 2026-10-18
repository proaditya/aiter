# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""SGLang adapter: Qwen3.8-27B single-sequence DFlash 2 rounds on the AITerMK megakernel.

SGLang keeps everything it owns: scheduling, prefill (its BF16 model), the KV / GDN pools, the radix cache
off. For a decode step of one greedy request, ``MegakernelDFlash.round`` replaces SGLang's draft + target
verify + commit with:

1. DFlash2Mono (persistent FlyDSL drafter, MXFP8, MXFP4-screened LM head) drafts 7 tokens after the
   anchor (the previous round's bonus token);
2. the megakernel (MXFP4, one launch) verifies the 8 rows over SGLang's pools and picks each row's greedy
   token;
3. the accepted prefix is committed into SGLang's pools (state in place, conv window, KV rows), and the
   8 rows' target features go into the drafter's context.

The prompt's context reaches the drafter from SGLang's prefill (its DFlash aux capture).
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
    """One SGLang target + DFlash 2 draft checkpoint; serves one request at a time."""

    def __init__(self, *, hf_path: str, draft_path: str, aot_dir: str, max_model_len: int, embed: torch.Tensor,
                 lm_head: torch.Tensor, conv: torch.Tensor, state: torch.Tensor, k_buffers, v_buffers,
                 variant: str = "m8") -> None:
        from aitermk_qwen38.provider import weights as W
        from aitermk_qwen38.provider.sgl_engine import SglMonoRound

        dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        weights = W.convert_model(W.safetensors_source(hf_path, dev))
        self.mono = SglMonoRound(weights, max_model_len, conv=conv, state=state, k_buffers=k_buffers,
                                 v_buffers=v_buffers, variant=variant, device=dev)
        ps = json.loads((Path(aot_dir) / "manifest.json").read_text())["abi"]["Ps"]
        fit = [p for p in ps if p >= max_model_len + 8]
        if not fit:
            raise ValueError(f"DFlash2Mono exports max_pos {ps}; max_model_len {max_model_len} needs >= {max_model_len + 8}")
        self.drafter = _drafter_module().DFlash2Mono(
            draft_path, embed, lm_head, min(fit), prec=dict(layers="aot8", lm="mx4r128", ctx="aot8"), aot=aot_dir)
        self.rows = self.mono.rows
        self.ids = torch.zeros(self.rows, dtype=torch.int64, device=dev)
        self.base = torch.zeros(1, dtype=torch.int64, device=dev)
        self.request = torch.zeros(1, dtype=torch.int64, device=dev)
        self.ar = torch.arange(self.rows, dtype=torch.int64, device=dev)
        self.s_anchor = torch.zeros(1, dtype=torch.int64, device=dev)
        self.s_slot = torch.zeros(1, dtype=torch.int32, device=dev)
        self.s_out_slots = torch.zeros(self.rows, dtype=torch.int64, device=dev)
        self.s_out_tokens = torch.zeros(1, self.rows, dtype=torch.int64, device=dev)
        self.s_commit = torch.zeros(1, dtype=torch.int32, device=dev)
        self.s_bonus = torch.zeros(1, dtype=torch.int64, device=dev)
        # a round is two graphs: the head (verify + greedy accept), after which SGLang publishes the new
        # sequence length (its scheduler waits on that), and the tail (commit, context, the next round's
        # draft), which then overlaps SGLang's host work for the next step
        self._graphs = {}
        self._warm = set()
        self._eager = os.environ.get("AITERMK_SGL_EAGER") == "1"
        # whether ``ids`` holds this request's next block, drafted by the previous round's tail
        self._drafted = False
        self._tail_pending = None
        logger.info("AITerMK megakernel DFlash 2 path ready (variant %s, drafter max_pos %d)", variant, min(fit))

    def prefill_context(self, positions: torch.Tensor, aux, *, new_sequence: bool) -> None:
        """The prompt's target features (SGLang's DFlash capture: packed [n, 5 * HIDDEN] or a list of five
        [n, HIDDEN]) into the drafter; ``new_sequence`` on its first chunk."""
        if new_sequence:
            self.mono.invalidate_mirror()
        self._drafted = False
        packed = aux if isinstance(aux, torch.Tensor) else torch.cat(list(aux), -1)
        n = positions.numel()
        feats = packed.view(n, 5, -1)
        for i in range(0, n, 1024):
            self.drafter.context(positions[i:i + 1024].long(), feats[i:i + 1024].contiguous())

    def invalidate_draft(self) -> None:
        """A step this adapter did not run: the block drafted ahead belongs to no round."""
        self._drafted = False
        self._tail_pending = None

    def round(self, *, anchor: torch.Tensor, prefix_len: torch.Tensor, mamba_slot: torch.Tensor,
              req_to_token: torch.Tensor, req_index: torch.Tensor):
        """Draft + verify of one request (eager; check mode). ``anchor`` int64 [1] (the bonus token at
        ``prefix_len``). Returns (draft tokens [1, rows], target greedy [1, rows])."""
        self._stage(anchor, prefix_len, mamba_slot, req_index)
        self._draft()
        self.mono.verify(self.ids, self.base, self.s_slot, req_to_token, self.request)
        self._drafted = False
        return self.ids.view(1, -1), self.mono.target.view(1, -1)

    def commit(self, *, commit_len: torch.Tensor, out_slots: torch.Tensor) -> None:
        """Settle the round's first ``commit_len`` rows (accepted drafts + the anchor) into SGLang's pools and
        hand all rows' features to the drafter (rows past the commit are overwritten next round)."""
        self.mono.commit(commit_len.view(1).to(torch.int32), out_slots.view(-1).to(torch.int64))
        self.drafter.context(self.base + self.ar, self.mono.aux())

    def full_round(self, *, anchor: torch.Tensor, prefix_len: torch.Tensor, mamba_slot: torch.Tensor,
                   req_to_token: torch.Tensor, req_index: torch.Tensor, out_slots: torch.Tensor):
        """The head of a round: (the draft, unless the previous round's tail drafted it,) verify and greedy
        accept. Returns (out_tokens [1, rows] int64, commit_lens [1] int32, bonus [1] int64), fresh tensors.
        ``finish_round`` must follow, once SGLang has published the new length."""
        self.finish_round()  # (a tail left pending: settle it before this round reads the pools)
        self._stage(anchor, prefix_len, mamba_slot, req_index)
        self.s_out_slots.copy_(out_slots.view(-1))
        if not self._drafted:
            self._run("draft", self._draft)
        self._run("head", lambda: self._head(req_to_token))
        self._tail_pending = req_to_token
        return self._results()

    def finish_round(self) -> None:
        """The tail of the round ``full_round`` began: commit, the rows' context, the next block's draft."""
        if self._tail_pending is None:
            return
        self._tail_pending = None
        self._run("tail", self._tail)
        self._drafted = True

    # ---- graph pieces: every input in a static buffer, no host sync -------------------------

    def _run(self, name: str, body) -> None:
        """``body`` replayed from its graph after a warm-up call (eager with ``AITERMK_SGL_EAGER=1``)."""
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

    # ---- graph pieces: every input in a static buffer, no host sync -------------------------

    def _stage(self, anchor, prefix_len, mamba_slot, req_index) -> None:
        self.s_anchor.copy_(anchor.view(1))
        self.base.copy_(prefix_len.view(1))
        self.s_slot.copy_(mamba_slot.view(1))
        self.request.copy_(req_index.view(1))

    def _draft(self) -> None:
        """The block after the anchor at ``base``."""
        dm = self.drafter
        self.ids.zero_()
        self.ids[:1].copy_(self.s_anchor)
        # the mailbox tag is (base, step): a repeated base must not read an earlier draft's mailboxes
        dm.step.add_(1)
        dm.draft_into(self.ids, self.base)

    def _head(self, req_to_token) -> None:
        self.mono.verify(self.ids, self.base, self.s_slot, req_to_token, self.request)
        cand, tgt = self.ids.view(1, -1), self.mono.target.view(1, -1)
        match = (cand[:, 1:] == tgt[:, :-1]).to(torch.int32)
        accept = torch.cumprod(match, dim=1).sum(dim=1)  # accepted drafts, [1]
        self.s_bonus.copy_(tgt.gather(1, accept[:, None].to(torch.int64)).view(1))
        self.s_commit.copy_(accept + 1)
        # SGLang's committed block: the drafts shifted left, the bonus at the accept boundary
        self.s_out_tokens[:, :-1].copy_(cand[:, 1:])
        self.s_out_tokens[:, -1].fill_(0)
        self.s_out_tokens.scatter_(1, accept[:, None].to(torch.int64), self.s_bonus.view(1, 1))

    def _tail(self) -> None:
        self.mono.commit(self.s_commit, self.s_out_slots)
        self.drafter.context(self.base + self.ar, self.mono.aux())
        # the next round's block: its anchor is this round's bonus, at the committed length
        self.base.add_(self.s_commit.to(self.base.dtype))
        self.s_anchor.copy_(self.s_bonus)
        self._draft()

    def _results(self):
        return self.s_out_tokens.clone(), self.s_commit.clone(), self.s_bonus.clone()
