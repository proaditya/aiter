#!/usr/bin/env python3
"""geval_run.py: one freshly started `vllm serve` endpoint, then vLLM's GSM8K evaluator against it.

Stage 3 of libAiterMK's libaitermk-vllm-evaluation skill: tests/evals/gsm8k/gsm8k_eval.py from the matching vLLM fork,
5 shots, at most 256 tokens, temperature 0, seed 42, one request at a time.
  --mode stock     the default vLLM engine with the real checkpoint; no general plugin is loaded (the libAiterMK adapter
                   is installed in this container)
  --mode provider  a fleet MPK provider through libaitermk-vllm, started as gsm128_serve_bench.py starts it
The run is invalid (exit 2) if any request failed or the endpoint is not healthy afterwards; the caller renames such a
run *.failed and its accuracy must not be quoted (skill rule)."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("stock", "provider"), required=True)
ap.add_argument("--provider", default="")
ap.add_argument("--state", default="")
ap.add_argument("--model", required=True)
ap.add_argument("--dev", required=True, help="HIP_VISIBLE_DEVICES of the server")
ap.add_argument("--port", type=int, default=8131)
ap.add_argument("--n", type=int, default=1319)
ap.add_argument("--out", required=True, help="this run's results directory")
ap.add_argument("--evaluator", required=True, help="tests/evals/gsm8k/gsm8k_eval.py of the vLLM fork")
ap.add_argument("--datasets", required=True, help="cache directory for the evaluator's GSM8K files")
ap.add_argument("--stock-args", default="")
ap.add_argument("--plugins", default="", help="stock mode: VLLM_PLUGINS (default none)")
ap.add_argument("--conc", type=int, default=1, help="requests in flight (gsm8k: the evaluator's --max-concurrency)")
ap.add_argument("--eval", choices=("gsm8k", "chat", "mmlupro"), default="gsm8k",
                help="gsm8k = vLLM's evaluator (5-shot completions); chat = gchat_eval.py (0-shot chat, reasoning low)")
a = ap.parse_args()

G = os.path.join(a.out, "gsm8k")
os.makedirs(G, exist_ok=True)
os.makedirs(a.datasets, exist_ok=True)
BASE = f"http://127.0.0.1:{a.port}"


def log(*x):
    print("[GEVAL]", time.strftime("%H:%M:%S", time.gmtime()), *x, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read().decode()


env = dict(os.environ)
for k in list(env):
    if k.startswith(("AITER_MK", "AITERMK", "FLEET_", "REDLINE_")) or k == "VLLM_PLUGINS":
        env.pop(k)
env["HIP_VISIBLE_DEVICES"] = a.dev
meta = {"mode": a.mode, "n": a.n, "dev": a.dev, "port": a.port, "model": a.model}
if a.mode == "provider":
    with open(os.path.join(a.state, "manifest.json")) as f:
        man = json.load(f)
    init = man.get("init_args")
    if not init:
        with open(os.path.join(os.path.dirname(os.path.abspath(a.state)), "src", "init_args.json")) as f:
            init = json.load(f)
    cap = int(init["max_seq_length"]) - 1
    env.update(AITER_MK_FLEET_VLLM="1", VLLM_PLUGINS="fleet_mk", AITERMK_FLEET_MPK_PROVIDER=a.provider,
               AITERMK_FLEET_MPK_STATE=a.state)
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model,
           "--served-model-name", "gpt-oss-120b", "--load-format", "dummy", "--enforce-eager",
           "--max-model-len", str(cap), "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
           "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.9", "--port", str(a.port),
           "--no-async-scheduling", "--worker-cls", "fleet_vllm.native.worker.FleetMPKWorker"]  # fmt: skip
    meta.update(provider=a.provider, provider_sha256=sha256(a.provider), state=a.state,
                state_manifest_sha256=sha256(os.path.join(a.state, "manifest.json")), max_model_len=cap)
else:
    env["VLLM_PLUGINS"] = a.plugins  # default "": no general plugin, vLLM's default engine
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model,
           "--served-model-name", "gpt-oss-120b", "--port", str(a.port)] + a.stock_args.split()  # fmt: skip
meta["server_cmd"] = cmd
meta["server_env"] = {k: v for k, v in env.items()
                      if k.startswith(("AITER", "FLEET", "VLLM", "HIP", "MPK", "HSA", "GPU_", "FQ_", "PYTHONPATH"))}
meta["versions"] = {p: ver(p) for p in ("vllm", "libaitermk-vllm", "torch", "triton", "aiter", "amd-aiter", "transformers")}
try:
    meta["versions"]["rocm"] = open("/opt/rocm/.info/version").read().strip()
except OSError:
    pass
meta["general_plugins_installed"] = [(e.name, e.value) for e in md.entry_points(group="vllm.general_plugins")]
meta["python"] = sys.version.split()[0]

slog = open(os.path.join(a.out, "server.log"), "w")
log("server:", " ".join(cmd))
t0 = time.time()
srv = subprocess.Popen(cmd, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)


def stop_server():
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 60)):
        try:
            os.killpg(srv.pid, sig)
            srv.wait(timeout=wait)
            break
        except Exception:
            pass
    time.sleep(5)


rc = 1
try:
    while True:
        if srv.poll() is not None:
            raise SystemExit(f"server exited with {srv.returncode} before it was ready")
        try:
            if get("/health", 2)[0] == 200:
                break
        except Exception:
            pass
        if time.time() - t0 > 1800:
            raise SystemExit("server not ready after 1800 s")
        time.sleep(3)
    meta["startup_s"] = round(time.time() - t0, 1)
    meta["v1_models"] = json.loads(get("/v1/models")[1])
    log(f"ready after {meta['startup_s']} s; /v1/models ok")

    tag = {"chat": f"gsm8k-chat-n{a.n}", "mmlupro": f"mmlupro-n{a.n}"}.get(a.eval, f"gsm8k-n{a.n}")
    items = os.path.join(G, f"{tag}-items.json")
    summ = os.path.join(G, f"{tag}.json")
    elog = os.path.join(G, f"{tag}.log")
    here = os.path.dirname(os.path.abspath(__file__))
    if a.eval == "chat":
        ecmd = [sys.executable, os.path.join(here, "gchat_eval.py"), "--port", str(a.port), "--n", str(a.n),
                "--test", os.path.join(a.datasets, "test.jsonl"), "--items", items, "--summary", summ,
                "--conc", str(a.conc)]  # fmt: skip
        meta["concurrency"] = a.conc
        eenv = dict(os.environ)
        meta["evaluator"] = ecmd[1]
        meta["evaluator_sha256"] = sha256(ecmd[1])
        meta["evaluator_cmd"] = ecmd[1:]
    elif a.eval == "mmlupro":
        ecmd = [sys.executable, os.path.join(here, "mmlupro_eval.py"), "--port", str(a.port), "--n", str(a.n),
                "--test", os.path.join(a.datasets, "mmlupro_test.jsonl"), "--items", items, "--summary", summ,
                "--conc", str(a.conc)]  # fmt: skip
        meta["concurrency"] = a.conc
        eenv = dict(os.environ)
        meta["evaluator"] = ecmd[1]
        meta["evaluator_sha256"] = sha256(ecmd[1])
        meta["evaluator_cmd"] = ecmd[1:]
        meta["dataset_sha256"] = sha256(os.path.join(a.datasets, "mmlupro_test.jsonl"))
    else:
        ecmd = [sys.executable, os.path.join(here, "geval_eval.py"),
                "--host", "http://127.0.0.1", "--port", str(a.port), "--num-questions", str(a.n), "--num-shots", "5",
                "--max-tokens", "256", "--temperature", "0", "--seed", "42", "--max-concurrency", str(a.conc),
                "--save-results", summ]  # fmt: skip
        meta["concurrency"] = a.conc
        eenv = dict(os.environ, GEVAL_EVALUATOR=a.evaluator, GEVAL_ITEMS=items, TMPDIR=a.datasets, GEVAL_TIMEOUT="28800")
        meta["evaluator_request_timeout_s"] = 28800  # the evaluator's 600 s also covers queueing at concurrency 1
        meta["evaluator"] = a.evaluator
        meta["evaluator_sha256"] = sha256(a.evaluator)
        meta["evaluator_cmd"] = [a.evaluator] + ecmd[2:]
    meta["eval"] = a.eval
    log("evaluator:", " ".join(meta["evaluator_cmd"]))
    t1 = time.time()
    with open(elog, "w") as f:
        erc = subprocess.call(ecmd, env=eenv, stdout=f, stderr=subprocess.STDOUT)
    meta["eval_wall_s"] = round(time.time() - t1, 1)
    meta["datasets"] = {n: sha256(os.path.join(a.datasets, n)) for n in ("train.jsonl", "test.jsonl")
                        if os.path.exists(os.path.join(a.datasets, n))}
    with open(elog) as f:
        etext = f.read()
    errors = etext.count("Error calling vLLM")
    if a.eval in ("chat", "mmlupro") and os.path.exists(summ):
        errors += int(json.load(open(summ)).get("errors", 0))
    try:
        health = get("/health", 10)[0]
    except Exception as e:
        health = f"unreachable ({type(e).__name__})"
    res = json.load(open(summ)) if os.path.exists(summ) else {}
    rows = json.load(open(items)) if os.path.exists(items) else []
    first = rows[:128]
    meta.update(evaluator_rc=erc, request_errors=errors, health_after=health,
                accuracy=res.get("accuracy"), invalid_rate=res.get("invalid_rate"),
                first128_accuracy=sum(r["correct"] for r in first) / len(first) if first else None,
                first128_invalid=sum(r["pred"] == -9999999 for r in first) / len(first) if first else None,
                at_cap=res.get("at_cap", sum(r["completion_tokens"] >= 256 for r in rows)),
                empty=sum(not r.get("text", r.get("content")) for r in rows))
    slog.flush()
    with open(os.path.join(a.out, "server.log"), errors="replace") as f:
        stext = f.read()
    meta["server_fwd_pass_total_lines"] = stext.count("[FWD_PASS_TOTAL]")
    ok = erc == 0 and errors == 0 and health == 200 and len(rows) == a.n
    meta["valid"] = ok
    rc = 0 if ok else 2
    log(f"RESULT mode={a.mode} n={a.n} accuracy={meta['accuracy']} invalid={meta['invalid_rate']} "
        f"first128={meta['first128_accuracy']} at_cap={meta['at_cap']} empty={meta['empty']} errors={errors} "
        f"health_after={health} fwd_pass_total_lines={meta['server_fwd_pass_total_lines']} "
        f"eval_s={meta['eval_wall_s']} valid={ok}")
finally:
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    stop_server()
sys.exit(rc)
