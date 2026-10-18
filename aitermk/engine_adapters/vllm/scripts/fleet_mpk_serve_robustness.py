"""Robustness tests of a fleet MPK provider served through `vllm serve` (run inside the vLLM container).

usage: fleet_mpk_serve_robustness.py --provider SO --state DIR --refs JSON --model DIR [--only GROUPS] [--soak N] [--port P]

Groups (--only, comma-separated; default all):
  live     requests the kernel cannot decode get HTTP 400 and leave the server serving; edge requests (prompt +
           max_tokens exactly at capacity, max_tokens 1, stop strings, client disconnects, concurrent requests,
           chat, a random burst) complete; a probe reproduces the reference tokens after each group
  soak     N requests, every answer checked against the references; then SIGTERM must stop the server
  restart  a new server serves right after; SIGTERM and SIGKILL while a request decodes stop the server, and a
           new server serves exactly after each
  startup  each misconfiguration makes the server exit with its fix in the log, never hang
Prints one [ROBUST] PASS/FAIL line per check and a summary; exits 1 on any failure.
"""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--provider", required=True)
ap.add_argument("--state", required=True)
ap.add_argument("--refs", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=8071)
ap.add_argument("--only", default="live,soak,restart,startup")
ap.add_argument("--soak", type=int, default=256)
ap.add_argument("--log-dir", default="/tmp/robust")
a = ap.parse_args()
GROUPS = ("live", "soak", "restart", "startup")
unknown = set(a.only.split(",")) - set(GROUPS)
if unknown:
    ap.error(
        f"--only: unknown group(s) {', '.join(sorted(unknown))}; choose from {','.join(GROUPS)}"
    )
os.makedirs(a.log_dir, exist_ok=True)
with open(a.refs) as f:
    REF = json.load(f)
QS = [
    REF["questions"][i]
    for i in sorted(range(len(REF["questions"])), key=lambda i: REF["questions"][i]["q"])
]
with open(os.path.join(a.state, "manifest.json")) as f:
    CAP = int(json.load(f)["init_args"]["max_seq_length"]) - 1
BASE = f"http://127.0.0.1:{a.port}"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(
        f"[ROBUST] {'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True
    )
    return ok


def server_cmd(max_model_len=CAP, extra=(), drop=()):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model,
        "--served-model-name", "gpt-oss-120b", "--load-format", "dummy", "--enforce-eager",
        "--max-model-len", str(max_model_len), "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
        "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.9", "--port", str(a.port),
        "--no-async-scheduling", "--worker-cls", "fleet_vllm.native.worker.FleetMPKWorker",
    ]  # fmt: skip
    for flag in drop:
        i = cmd.index(flag)
        del cmd[i : i + (1 if flag.startswith("--no-") or flag == "--enforce-eager" else 2)]
    return cmd + list(extra)


def start_server(tag, cmd, env_extra=None, state=None, ready_timeout=1500):
    env = dict(os.environ)
    env.update(AITER_MK_FLEET_VLLM="1", VLLM_PLUGINS="fleet_mk", AITERMK_FLEET_MPK_PROVIDER=a.provider,
               AITERMK_FLEET_MPK_STATE=state or a.state)  # fmt: skip
    env.update(env_extra or {})
    log = open(os.path.join(a.log_dir, f"{tag}.log"), "w")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < ready_timeout:
        if p.poll() is not None:
            return p, False, time.time() - t0
        try:
            urllib.request.urlopen(BASE + "/health", timeout=2).read()
            return p, True, time.time() - t0
        except Exception:
            time.sleep(2)
    return p, False, time.time() - t0


def stop_server(p, wait=60):
    t0 = time.time()
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=wait)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=60)
        except Exception:
            pass
    time.sleep(5)
    return time.time() - t0


def live_in_group(pgid):
    """Processes of the group that have not exited (zombies do not count)."""
    n = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                state, _ppid, pgrp = f.read().rsplit(")", 1)[1].split()[:3]
        except (OSError, ValueError):
            continue
        n += state != "Z" and int(pgrp) == pgid
    return n


def kill_mid_request(p, sig):
    """Send sig to the server once a long streamed request has 5 chunks; wait until its processes have exited."""
    q = QS[0]
    body = {"model": "gpt-oss-120b", "prompt": q["prompt_ids"], "max_tokens": CAP - len(q["prompt_ids"]),
            "temperature": 0.0, "ignore_eos": True, "stream": True}  # fmt: skip
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    n = 0
    r = urllib.request.urlopen(req, timeout=120)
    for raw in r:
        if raw.startswith(b"data:") and b"[DONE]" not in raw:
            n += 1
            if n >= 5:
                break
    t0 = time.time()
    os.killpg(p.pid, sig)
    while time.time() - t0 < 90 and live_in_group(p.pid):
        p.poll()
        time.sleep(0.5)
    p.poll()
    r.close()
    return n, time.time() - t0, live_in_group(p.pid) == 0


def post(path, body, timeout=120):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def healthy():
    try:
        return urllib.request.urlopen(BASE + "/health", timeout=10).status == 200
    except Exception:
        return False


def complete(prompt, max_tokens, **kw):
    body = {
        "model": "gpt-oss-120b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        **kw,
    }
    return post("/v1/completions", body)


def stream_ids(prompt, max_tokens, disconnect_after=None, **kw):
    """Streamed completion; returns the text (or None after a deliberate disconnect)."""
    body = {"model": "gpt-oss-120b", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, **kw}  # fmt: skip
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    text, n = [], 0
    r = urllib.request.urlopen(req, timeout=120)
    try:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices", []):
                text.append(ch.get("text", ""))
                n += 1
            if disconnect_after is not None and n >= disconnect_after:
                return None
    finally:
        r.close()
    return "".join(text)


def probe_ok(tok, name):
    """q0 greedy, 64 tokens: must equal the reference's first 64 tokens."""
    q = QS[0]
    st, out = complete(q["prompt_ids"], 64, ignore_eos=True)
    if st != 200:
        return check(name, False, f"HTTP {st}")
    ref = tok.decode(q["token_ids"][:64], skip_special_tokens=True)
    return check(
        name,
        out["choices"][0]["text"] == ref,
        "matches the reference" if out["choices"][0]["text"] == ref else "TEXT DIFFERS",
    )


def live_tests(tok):
    q = QS[0]
    p0 = q["prompt_ids"]
    probe_ok(tok, "probe before the tests")
    bad = {
        "temperature 0.7": dict(temperature=0.7),
        "logprobs 5": dict(logprobs=5),
        "n 2": dict(n=2),
        "presence_penalty 0.5": dict(presence_penalty=0.5),
        "repetition_penalty 1.2": dict(repetition_penalty=1.2),
        "logit_bias": dict(logit_bias={"0": 5}),
        "min_tokens 5": dict(min_tokens=5),
        "structured output (choice)": dict(structured_outputs={"choice": ["yes", "no"]}),
        "thinking_token_budget 8": dict(thinking_token_budget=8),
        "logprob_token_ids": dict(logprob_token_ids=[0]),
    }
    for name, kw in bad.items():
        st, body = complete(p0, 16, **kw)
        check(f"rejected with 400: {name}", st == 400, f"HTTP {st} {str(body)[:120]}")
    st, body = complete(p0, CAP - len(p0) + 1)
    check(
        "rejected with 400: prompt + max_tokens = capacity + 1",
        st == 400,
        f"HTTP {st} {str(body)[:100]}",
    )
    st, body = complete(list(p0) * (CAP // len(p0) + 1), 1)
    check(
        "rejected with 400: prompt longer than the capacity",
        st == 400,
        f"HTTP {st} {str(body)[:100]}",
    )
    check("healthy after the rejected requests", healthy())
    probe_ok(tok, "probe after the rejected requests")

    n_edge = CAP - len(p0)
    st, out = complete(p0, n_edge, ignore_eos=True)
    ok = st == 200 and out["usage"]["completion_tokens"] == n_edge
    check(
        f"prompt + max_tokens = capacity ({len(p0)} + {n_edge})",
        ok,
        f"HTTP {st}" + (f", {out['usage']['completion_tokens']} tokens" if st == 200 else ""),
    )
    if ok:
        ref = tok.decode(q["token_ids"][:64], skip_special_tokens=True)
        check(
            "  its first 64 tokens match the reference", out["choices"][0]["text"].startswith(ref)
        )
    st, out = complete(p0, 1)
    check("max_tokens 1", st == 200 and out["usage"]["completion_tokens"] == 1, f"HTTP {st}")
    st, out = complete(p0, 128, stop=["\n"])
    check("stop string", st == 200 and out["usage"]["completion_tokens"] <= 128, f"HTTP {st}")
    st, out = complete(p0, 64)  # EOS honoured (ignore_eos off)
    check("EOS honoured (ignore_eos off)", st == 200, f"HTTP {st}")
    for k in (1, 5, 30):
        try:
            stream_ids(p0, 256, disconnect_after=k)
        except Exception as e:
            print(f"[ROBUST] (disconnect after {k} chunks: {type(e).__name__})", flush=True)
    time.sleep(2)
    check("healthy after client disconnects", healthy())
    probe_ok(tok, "probe after client disconnects")

    # A request vLLM finishes early must free the GPU at once: the adapter stops the launch (fleet_mpk_request_stop)
    # instead of letting it decode to its max_tokens (a full-capacity launch decodes for tens of seconds).
    long_mt = CAP - len(p0)
    for name, kw in (("stop string", dict(stop=["\n"])), ("EOS", {}), ("client disconnect", None)):
        if kw is None:
            try:
                stream_ids(p0, long_mt, disconnect_after=5)
            except Exception as e:
                print(f"[ROBUST] (disconnect of the long request: {type(e).__name__})", flush=True)
        else:
            complete(p0, long_mt, **kw)
        t1 = time.time()
        st, out = complete(p0, 8, ignore_eos=True)
        free_s = time.time() - t1
        check(
            f"{name} with max_tokens {long_mt}: the next request completes in {free_s:.1f} s (< 5 s)",
            st == 200 and free_s < 5.0,
            f"HTTP {st}",
        )
    probe_ok(tok, "probe after the early stops")

    outs = [None, None]

    def run(i):
        outs[i] = complete(QS[i]["prompt_ids"], 64, ignore_eos=True)

    ths = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(300)
    same = all(
        o is not None and o[0] == 200
        and o[1]["choices"][0]["text"] == tok.decode(QS[i]["token_ids"][:64], skip_special_tokens=True)
        for i, o in enumerate(outs)
    )  # fmt: skip
    check("two concurrent requests both served, both match their references", same)
    st, out = post("/v1/chat/completions", {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "What is 2+2?"}],
                                            "max_tokens": 32, "temperature": 0})  # fmt: skip
    check("chat completion", st == 200, f"HTTP {st}")
    rng = random.Random(7)
    burst_ok = True
    for i in range(30):
        qq = QS[rng.randrange(len(QS))]
        mt = rng.randint(1, max(1, min(200, CAP - len(qq["prompt_ids"]))))
        try:
            if rng.random() < 0.3:
                stream_ids(qq["prompt_ids"], mt, disconnect_after=rng.randint(1, 5))
            else:
                st, _ = complete(qq["prompt_ids"], mt, ignore_eos=rng.random() < 0.5)
                burst_ok &= st == 200
        except Exception as e:
            burst_ok = False
            print(f"[ROBUST] burst request {i}: {type(e).__name__} {e}", flush=True)
    check("random burst of 30 (random max_tokens, 30% disconnects)", burst_ok and healthy())
    probe_ok(tok, "probe after the burst")


def soak(tok, n):
    bad, t0 = 0, time.time()
    for i in range(n):
        q = QS[i % len(QS)]
        nc = min(len(q["token_ids"]), 256)
        st, out = complete(q["prompt_ids"], nc, ignore_eos=True)
        if st != 200 or out["choices"][0]["text"] != tok.decode(
            q["token_ids"][:nc], skip_special_tokens=True
        ):
            bad += 1
            print(
                f"[ROBUST] soak request {i} (q{q['q']}): HTTP {st} {'text differs' if st == 200 else ''}",
                flush=True,
            )
    check(
        f"soak: {n} requests, every answer equal to its reference",
        bad == 0,
        f"{bad} bad, {time.time() - t0:.0f} s",
    )


def startup_tests(tok):
    """Each misconfiguration must make the server exit with its fix in the log, within the timeout.

    Prefix caching may instead be switched off by vLLM itself (it does that for the attention-free
    stand-in model); then the server must say so in its engine config and serve exactly.
    """
    tmp = tempfile.mkdtemp(prefix="robust_state_")
    with open(os.path.join(a.state, "manifest.json")) as f:
        files = [x["file"] for x in json.load(f)["allocations"]]

    def state_copy(name, mutate):  # symlinks to the real files, then break one
        d = os.path.join(tmp, name, "state")
        os.makedirs(d)
        shutil.copy(os.path.join(a.state, "manifest.json"), d)
        for fn in files:
            os.symlink(os.path.join(a.state, fn), os.path.join(d, fn))
        mutate(d)
        return d

    def drop_one(d):
        os.unlink(os.path.join(d, files[-1]))

    def short_one(d):
        os.unlink(os.path.join(d, files[-1]))
        with open(os.path.join(d, files[-1]), "wb") as f:
            f.write(b"\0" * 4096)

    no_pc = ["--no-enable-prefix-caching"]
    cases = [
        ("--max-num-seqs 2", server_cmd(extra=["--max-num-seqs", "2"]), {}, None, "--max-num-seqs 1", False),
        (f"--max-model-len {CAP + 1}", server_cmd(max_model_len=CAP + 1), {}, None, f"--max-model-len {CAP}", False),
        ("prefix caching flag left out", server_cmd(drop=no_pc), {}, None, no_pc[0], True),
        ("--enable-prefix-caching", server_cmd(drop=no_pc, extra=["--enable-prefix-caching"]), {}, None, no_pc[0], True),
        ("CUDA graphs on", server_cmd(drop=["--enforce-eager"]), {}, None, "--enforce-eager", False),
        ("partitions of two GPUs", server_cmd(), {"HIP_VISIBLE_DEVICES": "1,2"}, None, "not the partitions of one GPU", False),
        ("a state file missing", server_cmd(), {}, state_copy("missing", drop_one), "the export is incomplete", False),
        ("a state file truncated", server_cmd(), {}, state_copy("short", short_one), "the export is incomplete", False),
    ]  # fmt: skip
    for i, (name, cmd, env_extra, state, expect, may_come_up) in enumerate(cases):
        p, ready, dt = start_server(f"startup{i}", cmd, env_extra, state, ready_timeout=600)
        if ready:
            with open(os.path.join(a.log_dir, f"startup{i}.log")) as f:
                off = "enable_prefix_caching=False" in f.read()
            if may_come_up and off:
                probe_ok(tok, f"startup: {name}: vLLM keeps prefix caching off, probe")
            else:
                check(f"startup refused: {name}", False, "the server came up")
            stop_server(p)
            continue
        exited = p.poll() is not None
        if not exited:
            stop_server(p)
        with open(os.path.join(a.log_dir, f"startup{i}.log")) as f:
            log = f.read()
        check(
            f"startup refused: {name}",
            exited and expect in log,
            f"exited={exited} after {dt:.0f} s, fix named: {expect in log}",
        )
        time.sleep(3)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    groups = set(a.only.split(","))
    if groups & {"live", "soak"}:
        p, ready, dt = start_server("live", server_cmd())
        if not check("server ready", ready, f"{dt:.0f} s"):
            stop_server(p)
            finish()
        try:
            if "live" in groups:
                live_tests(tok)
            if "soak" in groups and a.soak:
                soak(tok, a.soak)
        finally:
            dt = stop_server(p)
        check("SIGTERM stops the server", p.poll() is not None, f"{dt:.0f} s")
    if "restart" in groups:
        restart_tests(tok)
    if "startup" in groups:
        startup_tests(tok)
    finish()


def restart_tests(tok):
    p, ready, dt = start_server("restart", server_cmd())
    if check("a new server starts right after", ready, f"{dt:.0f} s"):
        probe_ok(tok, "probe on the new server")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            n, dt, gone = kill_mid_request(p, sig)
            check(
                f"{sig.name} while a request decodes stops the server",
                gone,
                f"after {n} streamed chunks, {dt:.0f} s",
            )
            p, ready, dt = start_server(f"after_{sig.name.lower()}", server_cmd())
            if not check(f"a new server starts after {sig.name}", ready, f"{dt:.0f} s"):
                break
            probe_ok(tok, f"probe after {sig.name}")
    stop_server(p)


def finish():
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"[ROBUST] SUMMARY {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    sys.exit(1 if n_fail or not RESULTS else 0)


if __name__ == "__main__":
    main()
