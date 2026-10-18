#!/usr/bin/env python3
"""geval_eval.py: vLLM's GSM8K evaluator (tests/evals/gsm8k/gsm8k_eval.py) plus a per-question record.

Runs the evaluator's own main() with the given command line. Its prompts, stop strings and scoring are used as they
are. Two additions:
- _score_gsm8k also writes each question's label, extracted answer and completion text to $GEVAL_ITEMS (the evaluator
  itself saves only the summary);
- the evaluator's request_timeout_seconds (600, not on its command line) becomes $GEVAL_TIMEOUT. It is one aiohttp
  total timeout per request, and with --max-concurrency 1 the time a request waits for the single connection counts
  against it, so every request still queued 600 s into a run fails with TimeoutError."""
import functools
import json
import os
import sys

EV = os.environ["GEVAL_EVALUATOR"]
ITEMS = os.environ["GEVAL_ITEMS"]
sys.path.insert(0, os.path.dirname(EV))
import gsm8k_eval as ge  # noqa: E402

_score = ge._score_gsm8k


def score(states, output_tokens, labels, num_shots, max_tokens, latency):
    preds = [ge.get_answer_value(s) for s in states]
    rows = [
        {"i": i, "label": int(lab), "pred": int(p), "correct": bool(p == lab), "completion_tokens": int(t), "text": s}
        for i, (s, t, lab, p) in enumerate(zip(states, output_tokens, labels, preds))
    ]
    with open(ITEMS, "w") as f:
        json.dump(rows, f)
    return _score(states, output_tokens, labels, num_shots, max_tokens, latency)


ge._score_gsm8k = score
ge.evaluate_gsm8k = functools.partial(
    ge.evaluate_gsm8k, request_timeout_seconds=float(os.environ.get("GEVAL_TIMEOUT", "28800"))
)
sys.argv = [EV] + sys.argv[1:]
ge.main()

