#!/usr/bin/env python3
"""Summarize vllm_step_timing.py results: one row per json (config), pooled decode tok/s over the prompts.

Per prompt: tokens = new - 1 (the first token comes from the prefill step), decode time = decode steps x the median
step total (a few warm-up / tail steps are excluded from the median). Pooled tok/s = all tokens / all decode time,
95% CI = bootstrap over prompts. Also the mean per-step split (verify / draft / overhead, medians averaged over
prompts) and the wall-clock check (all new tokens / all generate() wall time, prefill included).

usage: gsm128_vllm_summary.py RESULT.json [RESULT.json ...] [--out summary.json]
"""
import json
import os
import random
import statistics as st
import sys


def boot(rows, n=2000, seed=1):
    rnd = random.Random(seed)
    v = []
    for _ in range(n):
        s = [rows[rnd.randrange(len(rows))] for _ in rows]
        v.append(1000.0 * sum(r[0] for r in s) / sum(r[1] for r in s))
    v.sort()
    return v[int(0.025 * n)], v[int(0.975 * n) - 1]


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


def main(argv):
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    out = {}
    print("%-14s %4s %8s %15s %8s %8s %7s %7s %7s | %7s %7s %7s | %s" % (
        "config", "n", "pooled", "95% CI", "tok/step", "ms/step", "p10", "median", "p90", "verify", "draft", "overh",
        "wall tok/s"))
    for f in argv:
        r = json.load(open(f))
        tag = os.path.basename(f)[:-5]
        rows, per, ver, dra, ovh, same = [], [], [], [], [], []
        toks = wall = 0.0
        for p in r["prompts"]:
            n = p["decode_steps_all"]
            s = p["summary"]
            if not s.get("median"):
                continue
            med = s["median"]["total"]
            t = p["new"] - 1
            rows.append((t, n * med, n))
            per.append(1000.0 * t / (n * med))
            ver.append(s["median"]["verify"] or 0.0)
            dra.append(s["median"]["draft"] or 0.0)
            ovh.append(s["median"]["overhead"] or 0.0)
            toks += p["new"]
            wall += p["wall_s"]
            same.append(p.get("same_prefix_as_megakernel", 0))
        if not rows:
            print("%-14s no data" % tag)
            continue
        pooled = 1000.0 * sum(x[0] for x in rows) / sum(x[1] for x in rows)
        lo, hi = boot(rows)
        steps = sum(x[2] for x in rows)
        o = out[tag] = dict(K=r["K"], method=r["method"], model=r.get("model"), vllm=r.get("vllm"), n=len(rows),
                            pooled=pooled, lo=lo, hi=hi, tok_per_step=sum(x[0] for x in rows) / steps,
                            ms_per_step=sum(x[1] for x in rows) / steps, p10=pct(per, 0.1), med=st.median(per),
                            p90=pct(per, 0.9), verify=st.mean(ver), draft=st.mean(dra), overhead=st.mean(ovh),
                            wall_tps=toks / wall, same_prefix_median=st.median(same))
        print("%-14s %4d %8.0f %7.0f-%-7.0f %8.3f %8.3f %7.0f %7.0f %7.0f | %7.3f %7.3f %7.3f | %.0f  (same prefix as "
              "megakernel: median %d tok)" % (tag, o["n"], pooled, lo, hi, o["tok_per_step"], o["ms_per_step"], o["p10"],
                                              o["med"], o["p90"], o["verify"], o["draft"], o["overhead"], o["wall_tps"],
                                              o["same_prefix_median"]))
    if out_path:
        json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
