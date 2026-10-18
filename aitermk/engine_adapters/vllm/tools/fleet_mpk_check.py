"""Drive a fleet MPK provider without vLLM and compare its tokens with reference runs.

usage: fleet_mpk_check.py --provider SO --state DIR --refs 'DIR/tok_%03d.json' --q 127,0,1 [--new 256]

Each reference file is a fleet `demo.py --save-tokens` JSON (prompt_ids, token_ids). Prints, per question, the
token count, whether the provider's tokens equal the reference (over the reference's length), the first
differing position, and decode tok/s as a serving benchmark computes it: (tokens - 1) / (last - first token).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fleet_vllm.native.session import ProviderSession  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--q", default="127")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--out")
    ap.add_argument("--hang-s", type=float, default=30.0)
    a = ap.parse_args()
    t0 = time.time()
    sess = ProviderSession(a.provider, a.state)
    print(
        f"[CHECK] session ready in {time.time() - t0:.0f} s, token arena {sess.max_seq}", flush=True
    )
    rows, ntok, tdec = [], 0, 0.0
    for q in [int(x) for x in a.q.split(",")]:
        with open(a.refs % q) as f:
            d = json.load(f)
        prompt = [int(x) for x in d["prompt_ids"]]
        ref = d["token_ids"]
        ref = [int(x) for x in (ref[0] if ref and isinstance(ref[0], list) else ref)]
        sess.start(prompt, a.new)
        got, first, last = [], None, None
        while not sess.done():
            ids = sess.poll(timeout=a.hang_s)
            if not ids:
                if (
                    sess.busy()
                ):  # the launch is alive but produced nothing: a hung launch never ends
                    print(
                        f"[CHECK] HANG q{q:03d} (request {sess.n_requests}): no token for {a.hang_s:.0f} s "
                        f"after {len(got)} tokens",
                        flush=True,
                    )
                    os._exit(3)
                break
            now = time.perf_counter()
            first = now if first is None else first
            last = now
            got += ids
        sess.wait_idle()
        m = min(len(ref), len(got))
        diff = next((i for i in range(m) if got[i] != ref[i]), None)
        tps = (len(got) - 1) / (last - first) if last and last > first else 0.0
        if len(got) > 1:
            ntok += len(got) - 1
            tdec += last - first
        same = diff is None and m > 0
        rows.append(
            dict(
                q=q,
                prompt=len(prompt),
                tokens=len(got),
                checked=m,
                same=same,
                first_diff=diff,
                tok_s=round(tps, 1),
            )
        )
        print(
            f"[CHECK] q{q:03d} prompt {len(prompt)} tokens {len(got)} checked {m} same {same} first_diff {diff} "
            f"decode {tps:.1f} tok/s",
            flush=True,
        )
    pooled = ntok / tdec if tdec else 0.0
    nsame = sum(r["same"] for r in rows)
    print(
        f"[CHECK] SUMMARY {len(rows)} questions, {nsame}/{len(rows)} identical, pooled decode {pooled:.1f} tok/s",
        flush=True,
    )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(rows=rows, pooled_tok_s=pooled), f, indent=1)


if __name__ == "__main__":
    main()
