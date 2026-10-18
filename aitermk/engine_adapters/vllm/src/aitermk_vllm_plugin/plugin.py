# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""vLLM general plugin (entry point group ``vllm.general_plugins``): Qwen3.8-27B DFlash 2 rounds on the AITerMK
megakernel.

Installed with this package and enabled by ``AITERMK_VLLM_DFLASH_AOT=<DFlash2Mono export dir>``; stock vLLM
otherwise. ``register`` wraps methods of vLLM's Model Runner V2 (``vllm.v1.worker.gpu.model_runner``):

- ``load_model``: attach a ``DFlashMegakernel`` (weights converted next to vLLM's own, before its memory
  profile) and wrap, on the instances, the target model's ``forward`` and ``compute_logits``, the speculator's
  ``propose``, the model state's ``prepare_attn`` and (once it exists) the cudagraph manager's replays;
- ``execute_model`` / ``prepare_inputs`` / ``sample_tokens``: route each step (host values only) and keep the
  step's inputs for the megakernel.

A round (a speculative step of one eligible request) then runs the megakernel's verify in place of the target
forward and its LM head, vLLM's rejection sampler on its logits, and the megakernel drafter's commit + context
+ draft in place of vLLM's drafter; every other step runs vLLM's own path unchanged.
"""

import functools
import inspect
import logging

logger = logging.getLogger(__name__)

_ATTR = "_aitermk_megakernel"


def register() -> None:
    from aitermk_vllm_plugin.dflash_megakernel import megakernel_requested

    if not megakernel_requested():
        return
    _route_logs()
    from vllm.v1.worker.gpu import model_runner as mr

    cls = mr.GPUModelRunner
    if getattr(cls, "_aitermk_patched", False):
        return
    _wrap_load_model(cls)
    _wrap_execute_model(cls)
    _wrap_prepare_inputs(cls)
    _wrap_sample_tokens(cls)
    cls._aitermk_patched = True
    logger.info("AITerMK megakernel DFlash plugin registered (Model Runner V2)")


def _route_logs() -> None:
    """vLLM configures only its own loggers: send ours through its handlers."""
    engine = logging.getLogger("vllm")
    for name in ("aitermk_qwen38", "aitermk_vllm_plugin"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        for handler in engine.handlers:
            if handler not in log.handlers:
                log.addHandler(handler)
        log.propagate = False


def _mk(runner):
    return getattr(runner, _ATTR, None)


def _wrap_load_model(cls) -> None:
    orig = cls.load_model

    @functools.wraps(orig)
    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        from aitermk_vllm_plugin.dflash_megakernel import DFlashMegakernel

        mk = DFlashMegakernel(self)
        setattr(self, _ATTR, mk)
        _wrap_instances(self, mk)
        return out

    cls.load_model = load_model


def _wrap_instances(runner, mk) -> None:
    model = runner.get_model()
    forward = model.forward
    model.forward = functools.wraps(forward)(lambda *a, **k: mk.forward(lambda: forward(*a, **k)))
    compute_logits = model.compute_logits
    model.compute_logits = functools.wraps(compute_logits)(lambda h: mk.compute_logits(compute_logits, h))

    speculator = runner.speculator
    propose = speculator.propose
    signature = inspect.signature(propose)

    @functools.wraps(propose)
    def _propose(*a, **k):
        return mk.propose(lambda: propose(*a, **k), signature.bind(*a, **k).arguments)

    speculator.propose = _propose

    state = runner.model_state
    prepare_attn = state.prepare_attn

    @functools.wraps(prepare_attn)
    def _prepare_attn(*a, **k):
        out = prepare_attn(*a, **k)
        if not k.get("for_capture", False):
            mk.note_attn_metadata(out)
        return out

    state.prepare_attn = _prepare_attn


def _wrap_graph_manager(runner, mk) -> None:
    """The target's cudagraph manager (built with the KV cache): a round replays none of its graphs."""
    cgm = getattr(runner, "cudagraph_manager", None)
    if cgm is None or getattr(cgm, "_aitermk_patched", False):
        return
    run_fullgraph, run_pw_graph = cgm.run_fullgraph, cgm.run_pw_graph
    cgm.run_fullgraph = lambda desc: mk.forward(lambda: run_fullgraph(desc))
    cgm.run_pw_graph = lambda model, inputs: mk.forward(lambda: run_pw_graph(model, inputs))
    cgm._aitermk_patched = True


def _wrap_execute_model(cls) -> None:
    orig = cls.execute_model

    @functools.wraps(orig)
    def execute_model(self, scheduler_output, *args, **kwargs):
        mk = _mk(self)
        if mk is None or kwargs.get("dummy_run", False):
            return orig(self, scheduler_output, *args, **kwargs)
        _wrap_graph_manager(self, mk)
        mk.begin_step(scheduler_output)
        try:
            return orig(self, scheduler_output, *args, **kwargs)
        except BaseException:
            mk.end_step()
            raise

    cls.execute_model = execute_model


def _wrap_prepare_inputs(cls) -> None:
    orig = cls.prepare_inputs

    @functools.wraps(orig)
    def prepare_inputs(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        mk = _mk(self)
        if mk is not None:
            mk.note_input_batch(out)
        return out

    cls.prepare_inputs = prepare_inputs


def _wrap_sample_tokens(cls) -> None:
    orig = cls.sample_tokens

    @functools.wraps(orig)
    def sample_tokens(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        finally:
            mk = _mk(self)
            if mk is not None:
                mk.end_step()

    cls.sample_tokens = sample_tokens
