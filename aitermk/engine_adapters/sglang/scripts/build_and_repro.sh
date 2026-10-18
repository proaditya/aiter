#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# One command, from this checkout on one AMD Instinct MI355X with SGLang installed: build and reproduce the
# Qwen3.8-27B single-request results in SGLang with the aitermk plugin (DFlash2 speculative decoding, the MXFP4
# megakernel verifying, one request at a time). Needs SGLang main (f825d72936 or later, checked through
# 59eb71a831), which has DFLASH:
#   1. install the plugin and the Qwen3.8-27B kernels package (skipped when they already import),
#   2. fetch the model and the DFlash2 draft (unless local paths are given),
#   3. set up the FlyDSL compiler used to build the kernels (pinned frontend and wheel),
#   4. build the megakernel and export the DFlash2 drafter code objects (skipped when already built),
#   5. export the prompt sets: GSM8K 128, HumanEval 128, MT-Bench 80,
#   6. serve SGLang with the plugin and benchmark it,
#   7. print the results next to the published ones.
#
# Usage, from the aitermk checkout:  bash aitermk/engine_adapters/sglang/scripts/build_and_repro.sh
#
# Optional environment:
#   WORK           everything goes here  (default ./aitermk_blog_benchmark)
#   MODEL, DRAFT   local checkpoint directories or Hugging Face ids  (default Qwen/Qwen3.8-27B, incoai/Qwen3.8-27B-DFlash2)
#   FLYDSL_SRC     a FlyDSL checkout or source tarball at commit 19418894  (default: cloned from github.com/ROCm/FlyDSL)
#   FLYDSL_WHEEL   the flydsl 0.4.0.dev963 wheel for this Python             (default: downloaded from PyPI)
#   FLYDSL_DIR     where step 3 sets FlyDSL up                               (default $WORK/flydsl)
#   DRAFTER_OBJECTS  where step 4 exports the drafter code objects           (default $WORK/drafter-objects)
#   AITERMK_QWEN38_CACHE  the megakernel code-object cache                    (default $WORK/kernel-cache)
#   SETUP_ONLY=1   steps 1, 3 and 4 only: no GPU, checkpoint or server (a container build prebuilds with it)
#   PORT           server port                                               (default 30000)

set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPTS/../../../.." && pwd)"
PKG="$ROOT/aitermk/megakernels/qwen38_27b"
PY="${PY:-python3}"
WORK="${WORK:-$PWD/aitermk_blog_benchmark}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
DRAFT="${DRAFT:-incoai/Qwen3.8-27B-DFlash2}"
PORT="${PORT:-30000}"
FLYDSL_COMMIT=19418894
FLYDSL_VERSION=0.4.0.dev963
SETS="gsm8k:128 humaneval:128 mt-bench:80"
BENCH="$SCRIPTS/single_request_bench.py"
SETUP_ONLY="${SETUP_ONLY:-0}"
mkdir -p "$WORK"
step() { printf '\n== %s\n' "$*"; }

step "0. checks"
if [ "$SETUP_ONLY" != 1 ] && command -v rocminfo > /dev/null; then rocminfo 2> /dev/null | grep gfx950 > /dev/null || { echo "needs an MI355X (gfx950)" >&2; exit 1; }; fi
"$PY" -c "import sglang" 2> /dev/null || { echo "needs SGLang main (f825d72936 or later) with DFLASH speculative decoding" >&2; exit 1; }

step "1. plugin"
"$PY" -c "import aitermk_sglang, aitermk_qwen38" 2> /dev/null || {
    # setuptools-scm (in these images) runs git without GIT_* variables, so a checkout that another user owns
    # (root in a container, on a bind mount) gets safe.directory from a temporary XDG git config instead
    SAFE="$(mktemp -d)" && mkdir -p "$SAFE/git" && printf '[safe]\n\tdirectory = %s\n' "$ROOT" > "$SAFE/git/config"
    XDG_CONFIG_HOME="$SAFE" "$PY" -m pip install --no-build-isolation "$PKG" "$ROOT/aitermk/engine_adapters/sglang"
    rm -rf "$SAFE"
}

step "2. checkpoints"
fetch() {  # fetch ID_OR_DIR -> prints a local directory
    if [ -d "$1" ]; then echo "$1"; return; fi
    local dst="$WORK/models/$(basename "$1")"
    [ -f "$dst/config.json" ] || "$PY" -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$1" "$dst" > /dev/null
    echo "$dst"
}
if [ "$SETUP_ONLY" = 1 ]; then
    echo "skipped (SETUP_ONLY)"
else
    MODEL=$(fetch "$MODEL")
    DRAFT=$(fetch "$DRAFT")
    echo "model $MODEL"
    echo "draft $DRAFT"
fi

step "3. FlyDSL (build time only; serving loads the code objects)"
FLY="${FLYDSL_DIR:-$WORK/flydsl}"
if [ ! -f "$FLY/READY" ]; then
    rm -rf "$FLY" && mkdir -p "$FLY/src" "$FLY/wheel" "$FLY/python"
    if [ -d "${FLYDSL_SRC:-}" ]; then
        cp -r "$FLYDSL_SRC"/. "$FLY/src/"
    elif [ -f "${FLYDSL_SRC:-}" ]; then
        tar xzf "$FLYDSL_SRC" -C "$FLY/src"
    else
        git clone -q https://github.com/ROCm/FlyDSL "$FLY/src" && git -C "$FLY/src" checkout -q "$FLYDSL_COMMIT"
    fi
    if [ -f "${FLYDSL_WHEEL:-}" ]; then
        cp "$FLYDSL_WHEEL" "$FLY/wheel/"
    else
        "$PY" -m pip download -q --no-deps --only-binary=:all: "flydsl==$FLYDSL_VERSION" -d "$FLY/wheel"
    fi
    # the frontend at the pinned commit, with the wheel's compiled MLIR bindings and libraries
    "$PY" -m zipfile -e "$FLY"/wheel/flydsl-*.whl "$FLY/wheel/x"
    cp -r "$FLY/src/python/flydsl" "$FLY/python/"
    cp -r "$FLY/wheel/x/flydsl/_mlir" "$FLY/python/flydsl/"
    cp -r "$FLY/wheel/x/flydsl.libs" "$FLY/python/"
    # the drafter's MXFP8 quantizer rounds E8M0 scales up (ceil), as the megakernel does; upstream rounds to nearest
    "$PY" - "$FLY/src/kernels/monokernel/mxfp8_linear.py" << 'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """            round_up = ((working & fx.Int32(0x400000)) != 0) & (
                ((working & fx.Int32(0x200000)) != 0) | ((working & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
            )"""
new = """            round_up = (working & fx.Int32(0x7FFFFF)) != 0  # ceil: never clip at the FP8 maximum"""
assert s.count(old) == 1, "unexpected FlyDSL mxfp8_linear.py: is the checkout at the pinned commit?"
open(p, "w").write(s.replace(old, new))
PYEOF
    touch "$FLY/READY"
fi
echo "flydsl $FLY"

step "4. code objects"
export AITERMK_QWEN38_CACHE="${AITERMK_QWEN38_CACHE:-$WORK/kernel-cache}"
export AITERMK_QWEN38_FLYDSL="$FLY/python"
"$PY" -c "from aitermk_qwen38.provider.kernels.build import build; print('megakernel', build('m8'))"
AOT="${DRAFTER_OBJECTS:-$WORK/drafter-objects}"
DRAFTER="$("$PY" -c "import os, aitermk_qwen38.provider.drafter as d; print(os.path.dirname(d.__file__))")"
if [ ! -f "$AOT/manifest.json" ]; then
    PYTHONPATH="$FLY/python${PYTHONPATH:+:$PYTHONPATH}" FLY_KERNELS_ROOT="$FLY/src" PROTO_ROOT="$DRAFTER/kernels_src" \
        "$PY" "$DRAFTER/aot_runtime.py" export --out "$AOT" --P 40968
fi
echo "drafter $AOT"
[ "$SETUP_ONLY" = 1 ] && exit 0

step "5. prompt sets"
PROMPTS="$WORK/prompts"
mkdir -p "$PROMPTS"
for s in $SETS; do
    name=${s%%:*}
    [ -s "$PROMPTS/${name}_130.jsonl" ] || "$PY" "$BENCH" --dataset "$name" --num-prompts 130 \
        --export-prompts "$PROMPTS/${name}_130.jsonl"
done

step "6. SGLang with the plugin"
OUT="$WORK/results/sglang"; mkdir -p "$OUT"
# TEMP is Python's temp-dir variable: never let a caller's TEMP reach the server
env -u TEMP SGLANG_DFLASH_MEGAKERNEL_AOT="$AOT" "$PY" -m sglang.launch_server --model-path "$MODEL" \
    --tokenizer-path "$MODEL" --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT" \
    --speculative-num-draft-tokens 8 --trust-remote-code --mem-fraction-static 0.6 --max-running-requests 1 \
    --context-length 40960 --port "$PORT" --attention-backend triton --mm-attention-backend sdpa \
    --skip-server-warmup --mamba-ssm-dtype float32 --disable-radix-cache > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2> /dev/null || true; wait "$SERVER_PID" 2> /dev/null || true' EXIT
for _ in $(seq 1 240); do
    curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" > /dev/null && break
    kill -0 "$SERVER_PID" 2> /dev/null || { echo "server exited; see $OUT/server.log" >&2; tail -20 "$OUT/server.log" >&2; exit 1; }
    sleep 5
done
for s in $SETS; do
    name=${s%%:*}; n=${s#*:}
    "$PY" "$BENCH" --base-url "http://127.0.0.1:${PORT}" --prompts-file "$PROMPTS/${name}_130.jsonl" \
        --num-prompts "$n" --max-tokens 2048 --stats none --chat-template-kwargs '{"reasoning_effort": "xhigh"}' \
        --save-text --out "$OUT/$name.json" > "$OUT/$name.log"
    echo "  $name done"
done

step "7. results (tokens/s, mean of each request's decode speed)"
"$PY" - "$OUT" << 'PYEOF'
import json, os, sys
published = {"gsm8k": 1102, "humaneval": 925, "mt-bench": 843}
print(f"{'set':10s} {'SGLang + aitermk':>17s}   published")
for name, pub in published.items():
    s = json.load(open(os.path.join(sys.argv[1], name + ".json")))["summary"]
    print(f"{name:10s} {s['decode_tok_s_per_request_mean']:17.0f}   {pub}")
print("\nThe megakernel is not bit-reproducible run to run, so speeds move by a few percent with the acceptance length.")
PYEOF
