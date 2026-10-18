#!/bin/bash
# prep_mmlupro.sh: download MMLU-Pro's test split (TIGER-Lab/MMLU-Pro on Hugging Face, open) and write the fixed
# stratified sample of the precision comparison: 100 questions per category (14 categories), random.Random(42) per
# category over the questions sorted by question_id. Output in $D: mmlupro_test.jsonl (the sample, SHA-256
# 12a1b87c9443a3dca831243f08cd9863ae643a1a60c999e8f6a49f42fb40e103) and mmlupro_full.jsonl (every question).
# The conversion needs pandas + pyarrow: by default it runs in the container $CONTAINER, which must mount $D at $DC.
#   D=<dir> [CONTAINER=<name> DC=<dir as the container sees it>] bash prep_mmlupro.sh
#   CONTAINER= bash prep_mmlupro.sh   # convert with the host's python3 instead
set -eu
D=${D:-$PWD/data/mmlupro}
CONTAINER=${CONTAINER-}
DC=${DC:-$D}
mkdir -p "$D"
U=https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet
[ -s "$D/test.parquet" ] || curl -sSL -o "$D/test.parquet" "$U"
sha256sum "$D/test.parquet" | tee "$D/test.parquet.sha256"
PY='
import json, random, sys
from collections import Counter
import pandas as pd
D = sys.argv[1]
df = pd.read_parquet(D + "/test.parquet")
print(len(df), "questions")
L = "ABCDEFGHIJ"
out, full = [], []
for cat, g in sorted(df.groupby("category"), key=lambda x: x[0]):
    g = g.sort_values("question_id")
    rows = [dict(id=int(r.question_id), category=cat, question=r.question, options=[str(o) for o in r.options],
                 answer=str(r.answer)) for r in g.itertuples()]
    for r in rows:
        assert r["answer"] in L[: len(r["options"])], r
    full += rows
    out += random.Random(42).sample(rows, min(100, len(rows)))
with open(D + "/mmlupro_full.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in full)
with open(D + "/mmlupro_test.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in out)
print("sample:", len(out), dict(Counter(r["category"] for r in out)))
'
if [ -n "$CONTAINER" ]; then
  docker exec -i "$CONTAINER" python3 - "$DC" <<< "$PY"
else
  python3 - "$D" <<< "$PY"
fi
sha256sum "$D/mmlupro_test.jsonl" | tee "$D/mmlupro_test.jsonl.sha256"
