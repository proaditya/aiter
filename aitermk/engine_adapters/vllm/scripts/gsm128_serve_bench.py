"""GSM8K-128 single-request benchmark of GPT-OSS-120B through `vllm serve` with a fleet MPK provider.

usage: gsm128_serve_bench.py --provider SO --state DIR --refs JSON --model DIR --out JSON [options]

Starts the OpenAI server with the fleet_vllm plugin in provider mode (AITERMK_FLEET_MPK_PROVIDER / _STATE,
--worker-cls fleet_vllm.native.worker.FleetMPKWorker), then sends each question's prompt token ids to
/v1/completions (streaming, greedy, ignore_eos, --new tokens), one request at a time.

Per request: TTFT, and decode tok/s = (tokens - 1) / (last chunk - first chunk), as serving benchmarks compute TPOT.
Pooled over the questions: all decode tokens / all decode time.
Correctness: FLEET_NATIVE_DUMP makes the session write each request's streamed token ids and the kernel's own
token arena; both are compared with the reference run's token ids (--refs, from the published run).
Host stalls: a request whose largest inter-chunk gap exceeds --stall-ms is sent again; if the rerun shows the same
gap (within 25%) it comes from the text (special tokens the detokenizer holds back) and the first timing is kept;
otherwise the attempt with the smaller gap is kept. A request without a token for --hang-s restarts the server
and is retried (counted in the summary's server restarts).
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument(
    "--provider",
    required=True,
    help="provider DSO (libaitermk_provider_gpt_oss_gfx950_fleet_mpk_*.so)",
)
ap.add_argument("--state", required=True, help="the export's state directory (state/manifest.json)")
ap.add_argument("--refs", required=True, help="reference token ids: gsm128_ref_<config>.json")
ap.add_argument(
    "--model", required=True, help="gpt-oss-120b checkpoint (config and tokenizer only)"
)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--new", type=int, default=256)
ap.add_argument("--port", type=int, default=8061)
ap.add_argument("--util", default="0.9")
ap.add_argument(
    "--dump", default="", help="directory for the streamed-vs-arena dumps (enables the token check)"
)
ap.add_argument("--stall-ms", type=float, default=25.0)
ap.add_argument("--hang-s", type=float, default=30.0)
ap.add_argument(
    "--pidfile", default="", help="write this process's pid here (to stop it from outside)"
)
a = ap.parse_args()
with open(a.refs) as f:
    REF = json.load(f)
QS = {int(x["q"]): x for x in REF["questions"]}
N, NEW = min(a.n, len(QS)), a.new
if a.pidfile:
    with open(a.pidfile, "w") as f:
        f.write(f"{os.getpid()}\n")
# SIGTERM ends the run through the finally below, which stops the server's whole process group.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


def capacity(state_dir):
    """The build's longest prompt + max_tokens (fleet_vllm.native.limits.request_capacity)."""
    with open(os.path.join(state_dir, "manifest.json")) as f:
        args = json.load(f).get("init_args")
    if not args:
        with open(
            os.path.join(os.path.dirname(os.path.abspath(state_dir)), "src", "init_args.json")
        ) as f:
            args = json.load(f)
    return int(args["max_seq_length"]) - 1


def summarize(res):
    """The kept attempt per question, pooled decode tok/s over the kept attempts that are not host stalls."""
    by_q = {}
    for r in res["requests"]:
        by_q.setdefault(r["q"], []).append(r)
    kept = []
    for q in sorted(by_q):
        att = [r for r in by_q[q] if r.get("tokens")]
        if not att:
            continue
        first = att[0]
        if first["max_gap_ms"] <= a.stall_ms:
            k = dict(first, stall=False)
        elif len(att) > 1 and all(
            abs(x["max_gap_ms"] - first["max_gap_ms"]) <= 0.25 * first["max_gap_ms"]
            for x in att[1:]
        ):
            k = dict(
                first, stall=False, gap_reproduced=True
            )  # deterministic gap: the text, not the host
        else:
            best = min(att, key=lambda x: x["max_gap_ms"])
            k = dict(
                best,
                stall=best["max_gap_ms"] > a.stall_ms
                and len(att) > 1
                and not any(
                    abs(x["max_gap_ms"] - best["max_gap_ms"]) <= 0.25 * best["max_gap_ms"]
                    for x in att
                    if x is not best
                ),
            )
        kept.append(k)
    clean = [r for r in kept if not r["stall"]]
    tok = sum(r["tokens"] - 1 for r in clean)
    sec = sum(r["decode_s"] for r in clean)
    res["kept"] = kept
    res["summary"] = dict(
        questions=len(kept),
        clean=len(clean),
        requests=len(res["requests"]),
        restarts=res.get("restarts", 0),
        gap_reproduced=sum(1 for r in kept if r.get("gap_reproduced")),
        decode_tok_s=tok / sec if sec else 0.0,
        ttft_ms_mean=1e3 * sum(r["ttft_s"] for r in kept) / len(kept) if kept else 0.0,
        tokens_same=sum(1 for r in kept if r.get("tokens_same")),
        tokens_checked_min=min((r.get("tokens_checked", NEW) for r in kept), default=0),
        streamed_eq_arena=sum(1 for r in kept if r.get("streamed_eq_arena")),
        text_same=sum(1 for r in kept if r.get("text_same")),
    )
    return res


CAP = capacity(a.state)
env0 = dict(os.environ)
env0.update(
    AITER_MK_FLEET_VLLM="1",
    VLLM_PLUGINS="fleet_mk",
    AITERMK_FLEET_MPK_PROVIDER=a.provider,
    AITERMK_FLEET_MPK_STATE=a.state,
)
for k in ("FLEET_MK_NATIVE", "FLEET_NATIVE_DEMO", "FLEET_NATIVE_ARGS", "REDLINE_VLLM_ENABLE"):
    env0.pop(k, None)
cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model,
    "--served-model-name", "gpt-oss-120b", "--load-format", "dummy", "--enforce-eager",
    "--max-model-len", str(CAP), "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
    "--no-enable-prefix-caching", "--gpu-memory-utilization", a.util, "--port", str(a.port),
    "--no-async-scheduling", "--worker-cls", "fleet_vllm.native.worker.FleetMPKWorker",
]  # fmt: skip
base = f"http://127.0.0.1:{a.port}"
if a.dump:
    shutil.rmtree(a.dump, ignore_errors=True)
    os.makedirs(a.dump, exist_ok=True)


class Server:
    def __init__(self, k):
        self.k, self.n_req = k, 0
        env = dict(env0)
        if a.dump:
            env["FLEET_NATIVE_DUMP"] = os.path.join(a.dump, f"srv{k}")
            os.makedirs(env["FLEET_NATIVE_DUMP"], exist_ok=True)
        print(f"[GSM128] server {k}: {' '.join(cmd)}", flush=True)
        self.p = subprocess.Popen(
            cmd, env=env, start_new_session=True
        )  # own process group: stop() kills all
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise SystemExit(f"server exited with {self.p.returncode}")
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except Exception:
                if time.time() - t0 > 1500:
                    raise SystemExit("server not ready after 1500 s") from None
                time.sleep(2)
        print(f"[GSM128] server {k} ready after {time.time() - t0:.0f} s", flush=True)

    def stop(self):
        # the engine-core child holds the GPU (and, after a launch hang, the stuck kernel): end the whole group
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 60)):
            try:
                os.killpg(self.p.pid, sig)
                self.p.wait(timeout=wait)
            except Exception:
                pass
        time.sleep(
            5
        )  # let the driver tear down the dead process's GPU context before the next server starts


def request(prompt, n):
    body = json.dumps(
        {
            "model": "gpt-oss-120b",
            "prompt": prompt,
            "max_tokens": n,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    stamps, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=a.hang_s) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if ch.get("text"):
                    stamps.append(time.perf_counter())
                    text.append(ch["text"])
    if not stamps or not usage:
        return None
    gaps = [y - x for x, y in zip(stamps, stamps[1:], strict=False)]
    return dict(
        tokens=usage["completion_tokens"],
        chunks=len(stamps),
        ttft_s=stamps[0] - t0,
        decode_s=stamps[-1] - stamps[0],
        max_gap_ms=1e3 * max(gaps) if gaps else 0.0,
        text="".join(text),
    )


res = {
    "config": REF.get("config"),
    "n": N,
    "new": NEW,
    "provider": a.provider,
    "requests": [],
    "restarts": 0,
}
srv = Server(0)
try:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)

    def one(q, attempt):
        global srv
        prompt, ref = QS[q]["prompt_ids"], QS[q]["token_ids"]
        r = None
        for _ in range(3):
            try:
                r = request(prompt, NEW)
            except (TimeoutError, OSError) as e:
                r = None
                print(
                    f"[GSM128] q{q:03d}: no token for {a.hang_s:.0f} s ({type(e).__name__}): restarting the server",
                    flush=True,
                )
            if r is not None:
                break
            srv.stop()
            res["restarts"] += 1
            srv = Server(res["restarts"])
        if r is None:
            raise SystemExit(f"q{q:03d} failed 3 times")
        nc = min(len(ref), NEW)
        ref_text = tok.decode(ref[:nc], skip_special_tokens=True)
        r.update(
            q=q,
            attempt=attempt,
            prompt_len=len(prompt),
            srv=srv.k,
            idx=srv.n_req,
            tokens_checked=nc,
            text_same=(r["text"] == ref_text) if nc == NEW else r["text"].startswith(ref_text[:-8]),
        )
        srv.n_req += 1
        del r["text"]
        res["requests"].append(r)
        tps = (r["tokens"] - 1) / r["decode_s"] if r["decode_s"] > 0 else 0
        print(
            f"[GSM128] q{q:03d} try{attempt} prompt {len(prompt)} tokens {r['tokens']} ttft "
            f"{1e3 * r['ttft_s']:.1f} ms decode {tps:.0f} tok/s max gap {r['max_gap_ms']:.1f} ms "
            f"text_same {r['text_same']}",
            flush=True,
        )
        return r

    for q in sorted(QS)[:N]:
        r = one(q, 0)
        k = 1
        while (
            r["max_gap_ms"] > a.stall_ms and k < 3
        ):  # a host stall, or a gap the text produces (reproduces)?
            r2 = one(q, k)
            k += 1
            if (
                abs(r2["max_gap_ms"] - r["max_gap_ms"]) <= 0.25 * r["max_gap_ms"]
                or r2["max_gap_ms"] <= a.stall_ms
            ):
                break
            r = r2
    # Closing 1-token requests: the first makes the session dump the last real request (for the token check). Two
    # more follow before the server is stopped: with only one, the last question's [SPEC] counter line was missing
    # from the server log.
    body = json.dumps(
        {
            "model": "gpt-oss-120b",
            "prompt": QS[sorted(QS)[0]]["prompt_ids"],
            "max_tokens": 1,
            "temperature": 0.0,
        }
    ).encode()
    for _ in range(3):
        urllib.request.urlopen(
            urllib.request.Request(
                base + "/v1/completions", data=body, headers={"Content-Type": "application/json"}
            ),
            timeout=600,
        ).read()
        time.sleep(1.0)
    if a.dump:
        for r in res["requests"]:
            p = os.path.join(a.dump, f"srv{r['srv']}", f"req{r['idx']}.json")
            if not os.path.exists(p):
                r["tokens_same"] = None
                continue
            with open(p) as f:
                d = json.load(f)
            ref = QS[r["q"]]["token_ids"][: r["tokens_checked"]]
            s, arena = d["streamed"][:NEW], d["arena"][:NEW]
            r["streamed_eq_arena"] = s == arena
            r["tokens_same"] = s[: len(ref)] == ref
            if not r["tokens_same"]:
                r["first_token_diff"] = next(
                    (j for j in range(min(len(s), len(ref))) if s[j] != ref[j]), len(ref)
                )
finally:
    srv.stop()

summarize(res)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(res, f, indent=1)
s, nq = res["summary"], res["summary"]["questions"]
print(
    f"[GSM128] SUMMARY {res['config']}: {nq} questions ({s['clean']} clean, {s['requests']} requests, "
    f"{s['restarts']} server restarts): decode {s['decode_tok_s']:.0f} tok/s pooled, TTFT {s['ttft_ms_mean']:.1f} ms "
    f"mean; tokens identical to the reference {s['tokens_same']}/{nq}, streamed == kernel arena "
    f"{s['streamed_eq_arena']}/{nq}, text identical {s['text_same']}/{nq} -> {a.out}",
    flush=True,
)
