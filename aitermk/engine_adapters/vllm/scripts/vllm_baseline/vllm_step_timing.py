#!/usr/bin/env python3
"""Stock vLLM on one GPU: decode-step time on the GPU timeline, split into verify / draft / overhead, for no
speculation, EAGLE3 and DFlash, on the prompts the megakernel's GSM8K-128 numbers use (same prompt token ids,
greedy, NEW tokens, EOS ignored). No vLLM code is changed: the worker's step methods are wrapped with hip events.

Per decode step:
  verify   = step start -> drafter start  (target forward over the K+1 tokens, logits, rejection sampling)
  draft    = drafter start -> drafter end
  overhead = drafter end -> next step start (bookkeeping, scheduling, input preparation, idle)
  total    = step start -> next step start
Without speculation (K=0) verify runs to the end of sampling and draft is 0.

Model runners: V1 (`_model_forward` exists: the step starts at the forward; vLLM <= 0.27) and V2 (default from 0.31:
the step is stamped at execute_model entry, the end of sample_tokens and around speculator.propose, with events on
the runner's main stream). `total` is the step period in both.

env:
  MODEL       target checkpoint (required)
  K           drafts per step, 0 = no speculation (default 0)
  METHOD      eagle3 | dflash (with K > 0)
  DRAFT       drafter checkpoint (required with K > 0)
  DRAFT_ATTN  drafter attention backend (default ROCM_AITER_FA for DFlash, vLLM's choice for EAGLE3)
  REF         a gsm128_ref_*.json from ../data (prompt ids + the megakernel's tokens per question), or
  PROMPTS     a comma list of json files that each hold prompt_ids (and optionally token_ids)
  N           first N questions of REF (default all)
  NEW         new tokens per prompt (default 256)
  KV_DTYPE    vLLM kv_cache_dtype (default: the checkpoint's; bfloat16 matches the megakernel)
  TARGET_ATTN target attention backend (default ROCM_AITER_UNIFIED_ATTN)
  BD_CC       1 = the SemiAnalysis compilation config (fuse_rope_kvcache, inductor graph partition); default 1
  GPU_UTIL, MAX_LEN, MAX_SEQS   vLLM engine arguments (defaults 0.90, 4096, 8)
  OUT         output json (default ./vllm_step_timing_k<K>.json)
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch  # noqa: E402,F401
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = os.environ.get("MODEL", "")
K = int(os.environ.get("K", "0"))
METHOD = os.environ.get("METHOD", "dflash")
DRAFT = os.environ.get("DRAFT", "")
DRAFT_ATTN = os.environ.get("DRAFT_ATTN", "ROCM_AITER_FA" if METHOD == "dflash" else "")
NEW = int(os.environ.get("NEW", "256"))
OUT = os.environ.get("OUT", f"vllm_step_timing_k{K}.json")


def load_prompts():
    """[(name, prompt_ids, reference token ids)] from REF or PROMPTS."""
    if os.environ.get("REF"):
        ref = json.load(open(os.environ["REF"]))
        qs = sorted(ref["questions"], key=lambda x: int(x["q"]))
        n = int(os.environ.get("N", str(len(qs))))
        return [("q%03d" % int(x["q"]), x["prompt_ids"], x.get("token_ids", [])) for x in qs[:n]]
    out = []
    for path in [p for p in os.environ.get("PROMPTS", "").split(",") if p]:
        d = json.load(open(path))
        out.append((os.path.basename(path), d["prompt_ids"], d.get("token_ids", [])))
    return out


def install(w):
    """Runs on the worker: wrap the runner's step methods with hip-event stamps."""
    import torch as T

    r = getattr(w, "model_runner", None) or w.worker.model_runner
    st = {"steps": []}
    r._bd = st
    if not hasattr(r, "_model_forward"):
        return install_v2(r, st)

    def ev():
        e = T.cuda.Event(enable_timing=True)
        e.record()
        return e

    o_exec, o_fwd, o_samp = r.execute_model, r._model_forward, r.sample_tokens
    o_prop = r.propose_draft_token_ids

    def execute_model(scheduler_output, *a, **k):
        st["steps"].append({"n": int(scheduler_output.total_num_scheduled_tokens), "x0": ev(),
                            "wall": time.perf_counter()})
        return o_exec(scheduler_output, *a, **k)

    def model_forward(*a, **k):
        s = st["steps"][-1] if st["steps"] else None
        if s is not None and "f0" not in s:
            s["f0"] = ev()
        out = o_fwd(*a, **k)
        if s is not None and "f1" not in s:
            s["f1"] = ev()
        return out

    def propose(*a, **k):
        s = st["steps"][-1]
        s["d0"] = ev()
        out = o_prop(*a, **k)
        s["d1"] = ev()
        return out

    def sample_tokens(*a, **k):
        out = o_samp(*a, **k)
        if st["steps"]:
            st["steps"][-1]["s1"] = ev()
        return out

    r.execute_model, r._model_forward, r.sample_tokens = execute_model, model_forward, sample_tokens
    r.propose_draft_token_ids = propose
    return "model runner V1, drafter: " + (type(r.drafter).__name__ if getattr(r, "drafter", None) is not None
                                           else "none")


def install_v2(r, st):
    """Model Runner V2: no separate forward method, drafting in r.speculator. No forward-only time (fwd)."""
    import torch as T

    stream = getattr(r, "main_stream", None)

    def ev():
        e = T.cuda.Event(enable_timing=True)
        if stream is not None:
            e.record(stream)
        else:
            e.record()
        return e

    o_exec, o_samp = r.execute_model, r.sample_tokens

    def execute_model(scheduler_output, *a, **k):
        e = ev()
        st["steps"].append({"n": int(scheduler_output.total_num_scheduled_tokens), "x0": e, "f0": e,
                            "wall": time.perf_counter()})
        return o_exec(scheduler_output, *a, **k)

    def sample_tokens(*a, **k):
        out = o_samp(*a, **k)
        if st["steps"]:
            st["steps"][-1]["s1"] = ev()
        return out

    r.execute_model, r.sample_tokens = execute_model, sample_tokens
    spec = getattr(r, "speculator", None)
    wrapped = "none"
    if spec is not None and hasattr(spec, "propose"):
        o_prop = spec.propose

        def propose(*a, **k):
            s = st["steps"][-1] if st["steps"] else {}
            s["d0"] = ev()
            out = o_prop(*a, **k)
            s["d1"] = ev()
            return out

        spec.propose = propose
        wrapped = type(spec).__name__ + ".propose"
    return f"model runner V2 (stream {'main' if stream is not None else 'current'}), drafter hook: {wrapped}"


def reset(w):
    r = getattr(w, "model_runner", None) or w.worker.model_runner
    r._bd["steps"] = []
    return True


def collect(w):
    """Runs on the worker: turn the events into per-step ms on the GPU timeline."""
    import torch as T

    r = getattr(w, "model_runner", None) or w.worker.model_runner
    T.cuda.synchronize()
    steps = [s for s in r._bd["steps"] if "f0" in s]
    rows = []
    for i, s in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        row = {"n": s["n"], "wall": s["wall"]}
        row["fwd"] = s["f0"].elapsed_time(s["f1"]) if "f1" in s else None
        if "d0" in s:
            row["verify"] = s["f0"].elapsed_time(s["d0"])
            row["draft"] = s["d0"].elapsed_time(s["d1"])
            end = s["d1"]
        else:
            row["verify"] = s["f0"].elapsed_time(s["s1"]) if "s1" in s else None
            row["draft"] = 0.0
            end = s.get("s1")
        if nxt is not None and end is not None:
            row["overhead"] = end.elapsed_time(nxt["f0"])
            row["total"] = s["f0"].elapsed_time(nxt["f0"])
            row["prep"] = nxt["x0"].elapsed_time(nxt["f0"])
        rows.append(row)
    return rows


def summarize(rows, k):
    dec = [r for r in rows if r["n"] == k + 1 and r.get("total") is not None]
    dec = dec[2:]  # the first decode steps still warm caches / graphs
    if not dec:
        return {"decode_steps": 0}

    def med(key):
        v = [r[key] for r in dec if r.get(key) is not None]
        return statistics.median(v) if v else None

    def avg(key):
        v = [r[key] for r in dec if r.get(key) is not None]
        return sum(v) / len(v) if v else None

    keys = ("verify", "fwd", "draft", "overhead", "prep", "total")
    return {"decode_steps": len(dec), "median": {x: med(x) for x in keys}, "mean": {x: avg(x) for x in keys}}


def main():
    if not MODEL or (K > 0 and not DRAFT):
        sys.exit("set MODEL (and DRAFT with K > 0); see the docstring")
    prompts = load_prompts()
    if not prompts:
        sys.exit("set REF (a gsm128_ref_*.json) or PROMPTS")
    kw = dict(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
        max_model_len=int(os.environ.get("MAX_LEN", "4096")),
        block_size=64,
        enable_prefix_caching=False,
        disable_log_stats=False,
        max_num_seqs=int(os.environ.get("MAX_SEQS", "8")),
        attention_backend=os.environ.get("TARGET_ATTN", "ROCM_AITER_UNIFIED_ATTN"),
    )
    if os.environ.get("KV_DTYPE"):
        kw["kv_cache_dtype"] = os.environ["KV_DTYPE"]
    if os.environ.get("BD_CC", "1") == "1":
        kw["compilation_config"] = {"pass_config": {"fuse_rope_kvcache": True}, "use_inductor_graph_partition": True}
    if K > 0:
        kw["speculative_config"] = {"method": METHOD, "model": DRAFT, "num_speculative_tokens": K}
        if DRAFT_ATTN:
            kw["speculative_config"]["attention_backend"] = DRAFT_ATTN
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[BD] engine ready in {time.time() - t0:.0f} s, K={K}", flush=True)
    print("[BD] hooks:", llm.collective_rpc(install), flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=NEW, ignore_eos=True)
    # warm-up (graphs, compile caches, first touch of the KV cache)
    llm.generate([TokensPrompt(prompt_token_ids=prompts[0][1])],
                 SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True), use_tqdm=False)
    res = {"K": K, "method": METHOD if K > 0 else "none", "draft": DRAFT if K > 0 else None, "model": MODEL,
           "env": {x: os.environ.get(x) for x in ("VLLM_ROCM_USE_AITER", "TARGET_ATTN", "DRAFT_ATTN", "BD_CC",
                                                  "KV_DTYPE")},
           "vllm": __import__("vllm").__version__, "prompts": []}
    for name, ids, ref in prompts:
        llm.collective_rpc(reset)
        w0 = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        wall = time.perf_counter() - w0
        rows = llm.collective_rpc(collect)[0]
        toks = list(out.outputs[0].token_ids)
        same = 0
        while same < min(len(toks), len(ref)) and toks[same] == ref[same]:
            same += 1
        s = summarize(rows, K)
        ndec = len([r for r in rows if r["n"] <= K + 1])
        entry = {"prompt": name, "prompt_len": len(ids), "new": len(toks), "wall_s": wall, "decode_steps_all": ndec,
                 "tokens_per_step": (len(toks) - 1) / ndec if ndec else None, "summary": s,
                 "same_prefix_as_megakernel": same, "rows": rows}
        m = s.get("median", {})
        print(f"[BD] {name} K={K}: steps {ndec}, tokens/step {entry['tokens_per_step']:.3f}, median verify "
              f"{m.get('verify')} draft {m.get('draft')} overhead {m.get('overhead')} total {m.get('total')} ms, "
              f"wall {wall:.3f} s, same prefix as megakernel {same}", flush=True)
        res["prompts"].append(entry)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(f"[BD] wrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
