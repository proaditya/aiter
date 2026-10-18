"""One fleet MPK provider in this process: its state, its init, and one free-running launch per request.

Each request is one launch on a stream of its own (only that stream waits for the kernel), on a background
thread. The provider is built with MPK_TOK_HOST=1: its scheduler publishes request 0's position and token id
into pinned host memory every iteration, so tokens stream to the host without a GPU call.

A launch that stops producing tokens is not waited for forever: after FLEET_MPK_STALL_S seconds (default 120)
without a token the session raises, which stops the engine so that a supervisor can restart the server.

A request vLLM finishes early (EOS, a stop string, an abort or a client disconnect) ends its launch: stop() asks
the kernel to end it (fleet_mpk_request_stop), which it does at its next 16-iteration check.
"""

import json
import os
import threading
import time

import torch

from .limits import request_capacity
from .provider import Provider
from .state import ProviderState

STALL_WARN_S = 10.0
_STALL_HINT = (
    "AMD SMI metric queries (amd-smi metric, GPU monitoring agents) stall the kernel while they run; "
    "otherwise the launch is hung and the server must be restarted"
)


class ProviderSession:
    def __init__(self, so_path, state_dir, log=print):
        self.log = log
        self.stall_s = float(os.environ.get("FLEET_MPK_STALL_S", "120"))
        self.state = st = ProviderState(state_dir, log=log)
        env = st.env()
        if env.get("MPK_TOK_HOST") != "1":
            raise RuntimeError(f"fleet MPK provider {so_path} was not built with MPK_TOK_HOST=1")
        os.environ.update(
            env
        )  # the runtime reads MPK_* (and the drafter pointer lists) in fleet_mpk_init
        self.prov = prov = Provider(so_path)
        if st.build_id and st.build_id != prov.build_id:
            raise RuntimeError(
                f"state {state_dir} is for {st.build_id}, provider {so_path} is {prov.build_id}"
            )
        names = [p["name"] for p in sorted(st.man["ptrs"], key=lambda p: p["idx"])]
        if names != prov.ptr_names:
            raise RuntimeError(f"state {state_dir} does not match the provider's pointer table")
        t0 = time.time()
        prov.init(st.ptrs(), st.meta(), st.init_args)
        prov.set_rope_tables(*st.rope())
        log(
            f"fleet_vllm: provider {os.path.basename(so_path)} ({prov.build_id}) initialized "
            f"in {time.time() - t0:.1f} s"
        )
        self.tokens = st.meta_view("tokens")
        self.prompt_lengths = st.meta_view("prompt_lengths")
        self.step = st.meta_view("step")
        self.num_new_tokens = st.meta_view("num_new_tokens")
        self.resets = st.reset_views()
        self.max_seq = min(self.tokens.size(1), int(st.init_args["max_seq_length"]))
        self.capacity = request_capacity(self.max_seq)
        self.launch_stream = torch.cuda.Stream()
        self.thread = None
        self.error = None
        self.prompt_len = 0
        self.n_expected = 0
        self.emitted = 0
        self.n_requests = 0
        self.streamed = []
        self.dump_dir = os.environ.get("FLEET_NATIVE_DUMP")

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def wait_idle(self):
        """Wait for the current launch to end (at its max_tokens, or soon after stop())."""
        if self.thread is not None:
            self.thread.join(self.stall_s)
            if self.thread.is_alive():
                raise RuntimeError(
                    f"fleet MPK provider: launch {self.n_requests} still running after {self.stall_s:.0f} s "
                    f"({_STALL_HINT})"
                )
            self.thread = None
        if self.error is not None:
            err, self.error = self.error, None
            raise err

    def start(self, prompt_ids, max_tokens):
        """Begin one request: reset the per-request state, then launch on a background thread."""
        n = len(prompt_ids)
        if n + max_tokens > self.capacity:
            raise ValueError(
                f"prompt {n} + max_tokens {max_tokens} exceeds the provider's capacity of {self.capacity} tokens"
            )
        self.wait_idle()
        if self.dump_dir and self.n_requests > 0:
            end = self.prompt_len + self.n_expected
            arena = self.tokens[0, self.prompt_len : end].cpu().tolist()
            with open(os.path.join(self.dump_dir, f"req{self.n_requests - 1}.json"), "w") as f:
                json.dump(dict(streamed=self.streamed, arena=arena), f)
        prov = self.prov
        prov.set_max_seq_length(n + max_tokens + 1)
        t = self.tokens
        t.zero_()
        t[0, :n] = torch.tensor(prompt_ids, dtype=t.dtype, device=t.device)
        for r in range(1, t.size(0)):
            t[r, :n] = t[0, :n]
        self.prompt_lengths.fill_(n)
        self.step.zero_()
        self.num_new_tokens.fill_(1)
        prov.init_request()
        for b in self.resets:  # monotonic barrier counters: zero between launches, never during one
            b.zero_()
        prov.reset_decode_progress()
        torch.cuda.synchronize()
        self.prompt_len, self.n_expected, self.emitted = n, max_tokens, 0
        self.error = None
        self.streamed = []
        self.n_requests += 1
        stream = self.launch_stream

        def run():
            try:
                prov.launch(stream.cuda_stream)
                stream.synchronize()
            except BaseException as e:  # surfaced by poll() / wait_idle()
                self.error = e

        self.thread = threading.Thread(target=run, name="fleet-mpk-launch", daemon=True)
        self.thread.start()

    def _available(self):
        pos = self.prov.decode_progress(0)
        if pos < self.prompt_len:
            return 0
        return min(pos - self.prompt_len + 1, self.n_expected)

    def poll(self, timeout=None, sleep_s=20e-6):
        """Wait until at least one new token exists (or the launch ended); return the new token ids.

        With a timeout, return [] when it expires. Without one, warn after STALL_WARN_S and raise after
        FLEET_MPK_STALL_S seconds without a token.
        """
        t0 = time.perf_counter()
        warned = False
        while True:
            avail = self._available()
            if avail > self.emitted:
                break
            if not self.busy():
                avail = self._available()
                if avail > self.emitted:
                    break
                if self.error is not None:
                    self.wait_idle()
                return []
            waited = time.perf_counter() - t0
            if timeout is not None:
                if waited > timeout:
                    return []
            elif waited > self.stall_s:
                raise RuntimeError(
                    f"fleet MPK provider: no token for {self.stall_s:.0f} s in request {self.n_requests} after "
                    f"{self.emitted} tokens ({_STALL_HINT})"
                )
            elif not warned and waited > STALL_WARN_S:
                warned = True
                self.log(
                    f"fleet_vllm: no token for {STALL_WARN_S:.0f} s in request {self.n_requests} after "
                    f"{self.emitted} tokens ({_STALL_HINT}); giving up after {self.stall_s:.0f} s"
                )
            time.sleep(sleep_s)
        base = self.prompt_len + self.emitted
        ids = [self.prov.decode_token(base + i) for i in range(avail - self.emitted)]
        if any(i < 0 for i in ids):
            raise RuntimeError(f"position published without its token id: {ids}")
        self.emitted = avail
        self.streamed += ids
        return ids

    def stop(self):
        """End the running launch early: vLLM finished its request (EOS, a stop string, an abort).

        The kernel ends the launch at its next 16-iteration check. A DSO without fleet_mpk_request_stop runs the
        launch to its max_tokens.
        """
        if self.busy() and self.prov.request_stop is not None:
            self.prov.request_stop()

    def done(self):
        return self.emitted >= self.n_expected
