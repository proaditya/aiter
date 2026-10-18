#!/usr/bin/env python3
"""geval_sum.py: GSM8K results of geval_run.py runs, one row per arm, and each arm paired with the first, question by
question.

usage: [ARMS="stock ns e3 df8"] geval_sum.py [RESULTS_DIR]   (default results/vllm-evaluation)
A run is a directory <RESULTS_DIR>/<date-time id>_gptoss_<arm> with a meta.json; the newest valid one per arm is used.
Paired statistics on the same questions: b = baseline right and arm wrong, c = arm right and baseline wrong;
difference = (c - b) / n with a 95% interval from the paired standard error, exact two-sided McNemar p from
Binomial(b + c, 1/2)."""
import glob
import json
import math
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "results/vllm-evaluation"
INVALID = -9999999


def newest(arm):
    runs = sorted(d for d in glob.glob(os.path.join(R, f"2*_gptoss_{arm}"))
                  if not d.endswith(".failed") and os.path.exists(os.path.join(d, "meta.json")))
    return runs[-1] if runs else None


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    items = glob.glob(os.path.join(d, "gsm8k", "gsm8k-*n*-items.json"))
    rows = json.load(open(items[0])) if items else []
    return meta, rows


def ci(p, n):
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return 1.96 * se


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


ARMS = os.environ.get("ARMS", "stock ns e3 df8").split()  # the first one is the baseline
arms = {}
for arm in ARMS:
    d = newest(arm)
    if d:
        arms[arm] = (d,) + load(d)

print(f"{'arm':6} {'n':>5} {'accuracy':>16} {'first 128':>10} {'invalid':>8} {'at 256 cap':>10} {'eval s':>7} "
      f"{'q/s':>6} {'out tok/s':>9}  run")
for arm, (d, meta, rows) in arms.items():
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else float("nan")
    f128 = rows[:128]
    a128 = sum(r["correct"] for r in f128) / len(f128) if f128 else float("nan")
    inv = sum(r["pred"] == INVALID for r in rows) / n if n else float("nan")
    summ = glob.glob(os.path.join(d, "gsm8k", "gsm8k-*n*[0-9].json"))
    s = json.load(open(summ[0])) if summ else {}
    cap = s.get("at_cap", sum(r["completion_tokens"] >= 256 for r in rows))
    print(f"{arm:6} {n:5d} {100 * acc:6.2f}% +-{100 * ci(acc, n):4.2f} {100 * a128:9.2f}% {100 * inv:7.2f}% {cap:10d} "
          f"{s.get('latency', 0):7.0f} {s.get('questions_per_second', 0):6.2f} {s.get('tokens_per_second', 0):9.1f}  "
          f"{os.path.basename(d)}")
    if abs(acc - (meta.get("accuracy") or acc)) > 1e-9:
        print(f"   (evaluator's own accuracy {meta.get('accuracy')} differs from the per-question record)")

BASE = ARMS[0]
if BASE in arms:
    _, _, srows = arms[BASE]
    print(f"\npaired with {BASE} (same questions, same prompts):")
    print(f"{'arm':6} {'both':>5} {'neither':>7} {'stock only':>10} {'arm only':>8} {'difference':>20} {'McNemar p':>9} "
          f"{'same answer':>11} {'same text':>9}")
    for arm, (d, meta, rows) in arms.items():
        if arm == BASE or len(rows) != len(srows):
            continue
        n = len(rows)
        both = sum(a["correct"] and b["correct"] for a, b in zip(srows, rows))
        neither = sum(not a["correct"] and not b["correct"] for a, b in zip(srows, rows))
        bb = sum(a["correct"] and not b["correct"] for a, b in zip(srows, rows))
        cc = sum(not a["correct"] and b["correct"] for a, b in zip(srows, rows))
        diff = (cc - bb) / n
        se = math.sqrt(max(bb + cc - (cc - bb) ** 2 / n, 0.0)) / n
        same_pred = sum(a["pred"] == b["pred"] for a, b in zip(srows, rows))
        same_text = sum(a.get("text", a.get("content")) == b.get("text", b.get("content")) for a, b in zip(srows, rows))
        print(f"{arm:6} {both:5d} {neither:7d} {bb:10d} {cc:8d} {100 * diff:+7.2f} pts +-{100 * 1.96 * se:4.2f} "
              f"{mcnemar(bb, cc):9.3f} {same_pred:11d} {same_text:9d}")

