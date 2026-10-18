"""vLLM side of a native fleet MPK provider (AITERMK_FLEET_MPK_PROVIDER set).

The provider owns prefill, its KV cache and its weights; vLLM keeps its API server, scheduler, detokenizer
and metrics:

* FleetMPKGptOssForCausalLM: an attention-free, weightless stand-in for vLLM's model init and memory profiling
  (serve with --load-format dummy). Its constructor checks the serving configuration, then loads the provider:
  AITERMK_FLEET_MPK_PROVIDER (the DSO) and AITERMK_FLEET_MPK_STATE (its exported device state).
* worker.FleetMPKWorker (--worker-cls fleet_vllm.native.worker.FleetMPKWorker): execute_model hands the
  scheduled request to the session and returns the tokens produced so far, so vLLM's model runner and sampler
  never run while serving.
* A step can return several tokens; the scheduler hook advances num_computed_tokens past all of them.
* Requests the provider cannot serve are rejected with HTTP 400 by the API server (limits.sampling_errors), and
  vLLM rejects prompt + max_tokens above --max-model-len, which may not exceed the build's capacity.

Limits: one request at a time (--max-num-seqs 1), greedy, no logprobs, no async scheduling, prompt plus
max_tokens within the provider's capacity. A request vLLM finishes early (EOS, a stop string, an abort) ends
its launch through ProviderSession.stop().
"""

import contextlib
import os
from typing import ClassVar, Literal

import torch
from torch import nn

from .limits import request_capacity, sampling_errors, serving_config_errors

SUPPORTED_VLLM = ("0.27.", "0.31.")  # tested; the hooked internals are the same in both

_DRIVER = None


class _Driver:
    def __init__(self, session):
        self.sess = session
        self.reqs = {}  # req_id -> [prompt ids, max_tokens, prompt tokens scheduled so far]
        self.active = None

    def execute(self, so):
        """{req_id: new token ids} for this engine step (at most one scheduled request)."""
        for nr in so.scheduled_new_reqs:
            sp = nr.sampling_params
            errors = ["no sampling parameters"] if sp is None else sampling_errors(sp)
            if errors:  # the API server rejects these; reaching the engine is a bug
                raise RuntimeError(
                    f"fleet MPK provider cannot serve request {nr.req_id}: {'; '.join(errors)}"
                )
            self.reqs[nr.req_id] = [list(nr.prompt_token_ids), int(sp.max_tokens), 0]
        for rid in so.finished_req_ids:
            self.reqs.pop(rid, None)
            if rid == self.active:
                self.sess.stop()
                self.active = None
        sched = so.num_scheduled_tokens
        if not sched:
            return {}
        if len(sched) != 1:
            raise RuntimeError(
                "fleet MPK provider: one request at a time (run with --max-num-seqs 1)"
            )
        rid, n = next(iter(sched.items()))
        st = self.reqs[rid]
        if rid != self.active:
            self.sess.start(st[0], st[1])
            self.active = rid
        st[2] += n
        if st[2] < len(st[0]):
            return {rid: []}  # vLLM is still feeding the prompt in chunks
        toks = self.sess.poll()
        if not toks:
            raise RuntimeError(
                f"fleet MPK provider: the launch ended before request {rid} had its tokens"
            )
        return {rid: toks}


@contextlib.contextmanager
def _plain_defaults():
    """CPU tensors and float32 defaults instead of vLLM's model-init context."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("cpu"):
            yield
    finally:
        torch.set_default_dtype(old)


def _check_serving_config(vllm_config, state_dir):
    """Fail at startup, before the state loads, if this vLLM configuration cannot serve the provider."""
    from .state import read_manifest

    arena = int(read_manifest(state_dir)["init_args"]["max_seq_length"])
    mc, sc = vllm_config.model_config, vllm_config.scheduler_config
    load_format = vllm_config.load_config.load_format
    errors = serving_config_errors(
        max_num_seqs=sc.max_num_seqs,
        max_model_len=mc.max_model_len,
        capacity=request_capacity(arena),
        enable_prefix_caching=bool(vllm_config.cache_config.enable_prefix_caching),
        enforce_eager=bool(mc.enforce_eager),
        async_scheduling=bool(getattr(sc, "async_scheduling", False)),
        load_format=str(getattr(load_format, "value", load_format)),
        watermark=getattr(vllm_config, "watermark_config", None) is not None,
        logits_processors=bool(getattr(mc, "logits_processors", None)),
    )
    if errors:
        raise RuntimeError(
            "fleet MPK provider cannot serve with this vLLM configuration:\n  - "
            + "\n  - ".join(errors)
        )


def driver(vllm_config=None):
    global _DRIVER
    if _DRIVER is None:
        import logging

        from .session import ProviderSession

        log = logging.getLogger("fleet_vllm").warning
        so = os.environ["AITERMK_FLEET_MPK_PROVIDER"]
        state = os.environ["AITERMK_FLEET_MPK_STATE"]
        if vllm_config is not None:
            _check_serving_config(vllm_config, state)
        with _plain_defaults():
            _DRIVER = _Driver(ProviderSession(so, state, log=log))
    return _DRIVER


class FleetMPKGptOssForCausalLM(nn.Module):
    is_attention_free: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        self.vocab_size = mc.get_vocab_size()
        dev = torch.device("cuda", torch.cuda.current_device())
        sc = vllm_config.scheduler_config
        self._hidden = torch.zeros(max(sc.max_num_batched_tokens, 1), 1, dtype=mc.dtype, device=dev)
        self._logits = torch.zeros(
            max(sc.max_num_seqs, 1), self.vocab_size, dtype=torch.float32, device=dev
        )
        driver(vllm_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._hidden[: input_ids.shape[0]]

    get_input_embeddings = embed_input_ids

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        n = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if n > self._hidden.shape[0]:
            self._hidden = torch.zeros(n, 1, dtype=self._hidden.dtype, device=self._hidden.device)
        return self._hidden[:n]

    def compute_logits(self, hidden_states: torch.Tensor, *args) -> torch.Tensor:
        n = hidden_states.shape[0]
        if n > self._logits.shape[0]:
            self._logits = torch.zeros(
                n, self.vocab_size, dtype=torch.float32, device=self._logits.device
            )
        return self._logits[:n]

    def load_weights(self, weights):
        for _ in weights:
            pass
        return set()


def _install_scheduler_hook():
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_fleet_mpk_hook", False):
        return
    orig = Scheduler._update_request_with_output

    def update(self, request, new_token_ids, *args, **kwargs):
        accepted, stopped = orig(self, request, new_token_ids, *args, **kwargs)
        stale = kwargs.get("is_stale", args[0] if args else False)
        if not stale and len(accepted) > 1:
            # schedule() counted one token; the launch already holds KV for the others.
            request.num_computed_tokens += len(accepted) - 1
        return accepted, stopped

    Scheduler._update_request_with_output = update
    Scheduler._fleet_mpk_hook = True


def _install_request_validation():
    """Reject what the provider cannot serve in the API server (HTTP 400), before it reaches the engine."""
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    if getattr(InputProcessor, "_fleet_mpk_validation", False):
        return
    orig = InputProcessor._validate_params

    def validate(self, params, *args, **kwargs):
        orig(self, params, *args, **kwargs)
        if isinstance(params, SamplingParams):
            errors = sampling_errors(params)
            if errors:
                raise VLLMValidationError("fleet MPK provider: " + "; ".join(errors))

    InputProcessor._validate_params = validate
    InputProcessor._fleet_mpk_validation = True


def register():
    import vllm
    from vllm import ModelRegistry

    if not vllm.__version__.startswith(SUPPORTED_VLLM):
        raise RuntimeError(
            "fleet_vllm's native fleet MPK provider path needs vLLM "
            + " or ".join(v + "x" for v in SUPPORTED_VLLM)
            + f", found {vllm.__version__}"
        )
    ModelRegistry.register_model("GptOssForCausalLM", FleetMPKGptOssForCausalLM)
    _install_scheduler_hook()
    _install_request_validation()
