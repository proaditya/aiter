"""pstats.py: paired, question-by-question comparisons of the Oct 17-18 accuracy runs (exact two-sided McNemar).
Usage: python3 pstats.py. RUNS lists the run directories of the comparisons in PRECISION_MATCHED_COMPARISON.md
(sections 3-3b): MK_EVAL_ROOT = the megakernel sessions' SESS_EVAL_OUT root, GEVAL_RESULTS = geval_run.py's
results root. A pair is skipped while either file is missing."""
import json
import math
import os

R = os.environ.get("MK_EVAL_ROOT", "results/mk_eval")
V = os.environ.get("GEVAL_RESULTS", "results/vllm-evaluation")
RUNS = {
    # 5-shot GSM8K (vLLM's evaluator), keyed by question index
    "g5_stock1": V + "/20261017T204200Z_gptoss_stock/gsm8k/gsm8k-n1319-items.json",
    "g5_vbfa": V + "/20261018T003420Z_gptoss_vbfa/gsm8k/gsm8k-n1319-items.json",
    "g5_w4a8": V + "/20261018T003416Z_gptoss_vw4a8/gsm8k/gsm8k-n1319-items.json",
    "g5_mx_prov": V + "/20261017T211210Z_gptoss_ns/gsm8k/gsm8k-n1319-items.json",
    "g5_mx_def": R + "/mkd_mx_s/gsm8k5/items.json",
    "g5_bf_def": R + "/mkd_bf_s/gsm8k5/items.json",
    # chat GSM8K
    "gc_stock1": V + "/20261017T214152Z_gptoss_stockchat/gsm8k/gsm8k-chat-n1319-items.json",
    "gc_mx_prov": V + "/20261017T215620Z_gptoss_nschat/gsm8k/gsm8k-chat-n1319-items.json",
    "gc_bf_def": R + "/mkd_bf_c/chat/items.json",
    # MMLU-Pro (1,400), keyed by question id
    "mp_stock": V + "/20261018T063842Z_gptoss_vbfammlu/gsm8k/mmlupro-n1400-items.json",
    "mp_w4a8": V + "/20261018T063840Z_gptoss_vw4a8mmlu/gsm8k/mmlupro-n1400-items.json",
    "mp_mx_def": R + "/mkd_mx_s/mmlupro/items.json",
    "mp_bf_def": R + "/mkd_bf_s/mmlupro/items.json",
}
PAIRS = [("g5_mx_prov", "g5_mx_def"), ("g5_mx_def", "g5_bf_def"), ("g5_stock1", "g5_bf_def"), ("g5_vbfa", "g5_bf_def"),
         ("g5_w4a8", "g5_bf_def"), ("gc_stock1", "gc_bf_def"), ("gc_mx_prov", "gc_bf_def"),
         ("mp_stock", "mp_w4a8"), ("mp_stock", "mp_mx_def"), ("mp_stock", "mp_bf_def"), ("mp_mx_def", "mp_bf_def"),
         ("mp_w4a8", "mp_mx_def")]


def find(path):
    if os.path.exists(path):
        return path
    d = os.path.dirname(path)  # tolerate the evaluator's own file names
    if os.path.isdir(d):
        c = [f for f in os.listdir(d) if f.endswith("items.json")]
        if len(c) == 1:
            return os.path.join(d, c[0])
    return None


def load(name):
    p = find(RUNS[name])
    if not p:
        return None
    rows = json.load(open(p))
    key = "id" if "id" in rows[0] else "i"
    return {r[key]: r for r in rows}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


cache = {}
for name in RUNS:
    d = load(name)
    if d is not None:
        cache[name] = d
        print("%-11s n %4d  accuracy %.4f  (%d correct)" % (name, len(d), sum(r["correct"] for r in d.values()) / len(d),
                                                          sum(r["correct"] for r in d.values())))
print()
for a, b in PAIRS:
    if a not in cache or b not in cache:
        print("%-11s vs %-11s : missing" % (a, b))
        continue
    A, B = cache[a], cache[b]
    ks = sorted(set(A) & set(B))
    only_a = sum(1 for k in ks if A[k]["correct"] and not B[k]["correct"])
    only_b = sum(1 for k in ks if B[k]["correct"] and not A[k]["correct"])
    same_text = sum(1 for k in ks if (A[k].get("text") or A[k].get("content")) == (B[k].get("text") or B[k].get("content")))
    d = 100.0 * (only_b - only_a) / len(ks)
    print("%-11s vs %-11s : n %d, %s right only %d, %s right only %d, diff %+.1f pts, McNemar p %.3g, identical text %d" % (
        a, b, len(ks), a, only_a, b, only_b, d, mcnemar(only_a, only_b), same_text))
