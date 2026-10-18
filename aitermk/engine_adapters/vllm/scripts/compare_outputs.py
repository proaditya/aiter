# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare the greedy outputs of benchmark runs (``single_request_bench.py --save-text``), prompt by prompt.

    python compare_outputs.py --tokenizer MODEL REF.json RUN.json [RUN2.json ...]

For every run against REF: how many outputs are identical, and where the others first differ (in tokens of the
reference). The megakernel is not bit-reproducible (atomic split-K sums, MXFP8 re-quantization), so judge a
pair of different paths against the same statistics for two runs of one path: a path is lossless when its
divergence from the reference is no earlier than that self noise.
"""

import argparse
import json
import statistics


def _texts(path):
    rows = json.load(open(path))["rows"]
    if not rows or "text" not in rows[0]:
        raise SystemExit(f"{path}: no output text (run the bench with --save-text)")
    return [r["text"] for r in rows]


def _first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokenizer", help="report positions in this tokenizer's tokens (default: characters)")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    unit = "tokens" if tok else "chars"

    def length(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False)) if tok else len(s)

    ref = _texts(a.ref)
    for run in a.runs:
        out = _texts(run)
        n = min(len(ref), len(out))
        same, diverged = 0, []
        for r, o in zip(ref[:n], out[:n]):
            if r == o:
                same += 1
                continue
            at = _first_diff(r, o)
            diverged.append((length(r[:at]), length(r)))
        line = f"{run}: {same}/{n} identical"
        if diverged:
            pos = [x for x, _ in diverged]
            frac = [x / max(t, 1) for x, t in diverged]
            line += (f"; the other {len(diverged)} first differ at {unit} median {statistics.median(pos):.0f} "
                     f"(min {min(pos)}), median {100 * statistics.median(frac):.0f}% of the reference output")
        print(line)


if __name__ == "__main__":
    main()
