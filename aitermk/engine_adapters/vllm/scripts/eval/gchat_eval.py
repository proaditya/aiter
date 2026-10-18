#!/usr/bin/env python3
"""gchat_eval.py: GSM8K in gpt-oss's chat format against a running OpenAI-compatible server.

The same GSM8K test file and order as vLLM's evaluator (its cached test.jsonl); 0-shot: the question, plus "Give the
final answer as a single number.", as the user message of /v1/chat/completions; reasoning effort low; temperature 0;
up to --max-tokens; one request at a time. Score: the last integer in the final answer (message.content), extracted
as the vLLM evaluator extracts it. Writes every question's record to --items and the summary to --summary; exits 2 if
any request failed."""
import argparse
import ast
import json
import re
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--n", type=int, default=1319)
ap.add_argument("--test", required=True, help="GSM8K test.jsonl (vLLM evaluator's cache)")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--effort", default="low")
ap.add_argument("--items", required=True)
ap.add_argument("--summary", required=True)
ap.add_argument("--conc", type=int, default=1, help="requests in flight (default 1, one at a time)")
a = ap.parse_args()
INVALID = -9999999


def value(s):
    nums = re.findall(r"\d+", (s or "").replace(",", ""))
    if not nums:
        return INVALID
    try:
        return ast.literal_eval(nums[-1])
    except SyntaxError:
        return INVALID


test = [json.loads(line) for line in open(a.test) if not line.startswith("#")][: a.n]
t0 = time.time()


def ask(i, ex):
    body = {"model": "gpt-oss-120b", "temperature": 0, "max_tokens": a.max_tokens, "reasoning_effort": a.effort,
            "messages": [{"role": "user", "content": ex["question"] + "\n\nGive the final answer as a single number."}]}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t1 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            res = json.load(r)
        ch = res["choices"][0]
        msg = ch["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        rec = {"finish": ch.get("finish_reason"), "completion_tokens": res.get("usage", {}).get("completion_tokens", 0),
               "content": content, "reasoning_chars": len(reasoning), "reasoning_head": reasoning[:300]}
    except Exception as e:  # noqa: BLE001
        rec = {"error": f"{type(e).__name__}: {e}", "content": "", "completion_tokens": 0}
    lab, pred = value(ex["answer"]), value(rec["content"])
    rec.update(i=i, label=lab, pred=pred, correct=pred == lab, latency_s=round(time.time() - t1, 3))
    return rec


if a.conc > 1:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(a.conc) as pool:
        rows = list(pool.map(ask, range(len(test)), test))
else:
    rows = []
    for i, ex in enumerate(test):
        rows.append(ask(i, ex))
        if (i + 1) % 50 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"[CHAT] {i + 1}/{len(test)} accuracy so far {acc:.3f} {time.time() - t0:.0f} s", flush=True)
errors = sum("error" in r for r in rows)
lat = time.time() - t0
n = len(rows)
summ = {"num_questions": n, "accuracy": sum(r["correct"] for r in rows) / n,
        "invalid_rate": sum(r["pred"] == INVALID for r in rows) / n, "errors": errors, "latency": lat,
        "questions_per_second": n / lat, "total_output_tokens": sum(r["completion_tokens"] for r in rows),
        "at_cap": sum(r["completion_tokens"] >= a.max_tokens for r in rows),
        "no_final": sum(not r["content"] for r in rows), "max_tokens": a.max_tokens, "effort": a.effort,
        "format": "0-shot chat, question + 'Give the final answer as a single number.'"}
json.dump(rows, open(a.items, "w"))
json.dump(summ, open(a.summary, "w"), indent=1)
print(f"Accuracy: {summ['accuracy']:.4f} invalid {summ['invalid_rate']:.4f} errors {errors} no_final {summ['no_final']} "
      f"at_cap {summ['at_cap']} latency {lat:.0f} s", flush=True)
sys.exit(2 if errors else 0)
