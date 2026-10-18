#!/usr/bin/env python3
"""mmlupro_eval.py: a multiple-choice benchmark (MMLU-Pro, or GPQA in the same JSONL form) in gpt-oss's chat format
against a running OpenAI-compatible server.

Each JSONL record: {"id", "category", "question", "options": [...], "answer": "<letter>"}. The user message is OpenAI
simple-evals' multiple-choice template with as many letters as the question has options; reasoning effort low;
temperature 0; up to --max-tokens. Score: the last "Answer: X" in the final answer (message.content), simple-evals'
pattern. Writes every question's record to --items and the summary to --summary; exits 2 if any request failed.
Same interface as gchat_eval.py."""
import argparse
import json
import re
import string
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--n", type=int, default=0, help="first N records (0 = all)")
ap.add_argument("--test", required=True, help="JSONL with id, category, question, options, answer")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--effort", default="low")
ap.add_argument("--items", required=True)
ap.add_argument("--summary", required=True)
ap.add_argument("--conc", type=int, default=1)
a = ap.parse_args()

TEMPLATE = ("Answer the following multiple choice question. The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before "
            "answering.\n\n{question}\n\n{options}")
ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\(?([A-J])\)?\$?")

recs = [json.loads(l) for l in open(a.test) if l.strip()]
if a.n:
    recs = recs[: a.n]


def ask(r):
    L = string.ascii_uppercase[: len(r["options"])]
    msg = TEMPLATE.format(letters=L, question=r["question"],
                          options="\n".join("%s) %s" % (x, o) for x, o in zip(L, r["options"])))
    body = {"model": "gpt-oss-120b", "temperature": 0, "max_tokens": a.max_tokens, "reasoning_effort": a.effort,
            "messages": [{"role": "user", "content": msg}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % a.port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=28800) as f:
            j = json.load(f)
    except Exception as e:  # noqa: BLE001
        return dict(id=r["id"], category=r.get("category", ""), label=r["answer"], pred="", correct=False,
                    completion_tokens=0, content="", error=repr(e)[:300], s=round(time.time() - t, 2))
    m = j["choices"][0]["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    found = ANSWER.findall(content)
    pred = found[-1].upper() if found else ""
    return dict(id=r["id"], category=r.get("category", ""), label=r["answer"], pred=pred, correct=pred == r["answer"],
                completion_tokens=j.get("usage", {}).get("completion_tokens", 0), content=content,
                reasoning_chars=len(reasoning), reasoning_head=reasoning[:300], s=round(time.time() - t, 2))


t0 = time.time()
with ThreadPoolExecutor(max_workers=a.conc) as ex:
    rows = list(ex.map(ask, recs))
n = len(rows)
errors = sum(1 for r in rows if r.get("error"))
cats = sorted({r["category"] for r in rows})
summary = dict(accuracy=sum(r["correct"] for r in rows) / n if n else 0.0, num_questions=n, errors=errors,
               invalid_rate=sum(not r["pred"] for r in rows) / n if n else 0.0,
               at_cap=sum(r["completion_tokens"] >= a.max_tokens for r in rows), max_tokens=a.max_tokens,
               effort=a.effort, wall_s=round(time.time() - t0, 1),
               per_category={c: round(sum(r["correct"] for r in rows if r["category"] == c) /
                                      max(1, sum(1 for r in rows if r["category"] == c)), 4) for c in cats})
json.dump(rows, open(a.items, "w"), indent=0)
json.dump(summary, open(a.summary, "w"), indent=1)
print("accuracy %.4f over %d, invalid %.3f, at cap %d, errors %d, %.0f s" % (
    summary["accuracy"], n, summary["invalid_rate"], summary["at_cap"], errors, summary["wall_s"]))
sys.exit(2 if errors else 0)
