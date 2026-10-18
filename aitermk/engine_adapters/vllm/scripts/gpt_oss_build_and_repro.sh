#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# One command, on a host with an AMD Instinct MI355X in DPX+NPS2 and a vLLM 0.31.0 (or 0.27.1) container: build and
# reproduce the GPT-OSS-120B GSM8K-128 single-request results served through `vllm serve` with libAiterMK's fleet MPK
# providers (the fleet persistent megakernel: no speculation, EAGLE3 block 6, DFlash block 8; one request at a time):
#   1. export the three providers with the fleet megakernel repository (skipped when PROVIDERS already holds them),
#   2. import each export as a build configuration and build the provider DSOs with this repository's CMake,
#   3. install the vLLM adapter (fleet_vllm) in the container,
#   4. serve each configuration and run the 128 GSM8K questions, every answer checked token for token against the
#      published run,
#   5. print the results next to the published ones.
#
# Usage, from the checkout (repository root) on the host:
#   CONTAINER=... ROCM70=... CK_DIR=... MODEL=... PROVIDERS=... bash aitermk/engine_adapters/vllm/scripts/gpt_oss_build_and_repro.sh
#
# Required environment (container paths unless noted):
#   CONTAINER   a running or stopped vLLM 0.31.0 (or 0.27.1) container that mounts HOST_HOME at CTR_HOME and sees
#               the GPUs
#   ROCM70      ROCm 7.0.0 root (bin/hipcc): the configs were validated with it; another compiler builds other code.
#               An Ubuntu 22.04 build for vLLM's image (e.g. /opt/rocm-7.0.0 of rocm/dev-ubuntu-22.04:7.0)
#   CK_DIR      Composable Kernel at the fleet pin: ROCm/composable_kernel d8ee107a47d8485dbcffc79eb08e4f7c39ea6335
#   MODEL       the gpt-oss-120b checkpoint directory (only its config and tokenizer are read)
# Optional environment:
#   PROVIDERS   host directory with the exports <config>/{src,state} (default $WORK/providers)
#   FLEET_REPO  host path of a fleet megakernel checkout with scripts/nps2/aitermk/exp_run.sh, to create missing exports
#   WORK        host directory for configs, build and results, under HOST_HOME (default ./aitermk_gpt_oss_benchmark)
#   HOST_HOME, CTR_HOME   the host directory the container mounts, and where       (default $HOME -> /root/$USER)
#   DEVICES     the MI355X's two DPX partitions as HIP devices                     (default 0,1)
#   CONFIGS     subset of "ns e3 df8" (the default: the MXFP4 LM head, the released 32K builds behind the blog's
#               headline numbers) and "ns_bf16 e3_bf16 df8_bf16" (the checkpoint's BF16 LM head, the precision vLLM
#               runs, in builds sized for these questions: the blog's matched-precision check); N questions per
#               config (default 128); PORT (default 8061)
#
# Keep GPU monitoring off while it runs: AMD SMI metric queries (amd-smi metric, monitoring agents) stall the kernel.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPTS/../../../.." && pwd)"
PROV_SRC="$ROOT/aitermk/megakernels/fleet/src/gpt_oss_gfx950_mpk"
for v in CONTAINER ROCM70 CK_DIR MODEL; do
    [ -n "${!v:-}" ] || { echo "set $v (see the header of $0)" >&2; exit 1; }
done
WORK="$(mkdir -p "${WORK:-$PWD/aitermk_gpt_oss_benchmark}" && cd "${WORK:-$PWD/aitermk_gpt_oss_benchmark}" && pwd)"
HOST_HOME="${HOST_HOME:-$HOME}"
CTR_HOME="${CTR_HOME:-/root/$(id -un)}"
PROVIDERS="$(mkdir -p "${PROVIDERS:-$WORK/providers}" && cd "${PROVIDERS:-$WORK/providers}" && pwd)"
DEVICES="${DEVICES:-0,1}"
CONFIGS="${CONFIGS:-ns e3 df8}"
N="${N:-128}"
PORT="${PORT:-8061}"
ctr() { echo "${1/#$HOST_HOME/$CTR_HOME}"; }  # a host path (absolute, under HOST_HOME) as the container sees it
in_ctr() { docker exec -i "$CONTAINER" bash -lc "$1"; }
name_of() { case $1 in ns) echo ns ;; e3) echo eagle3_b6 ;; df8) echo dflash_b8 ;;
    ns_bf16) echo ns_bf16 ;; e3_bf16) echo eagle3_b6_bf16 ;; df8_bf16) echo dflash_b8_bf16 ;; *) return 1 ;; esac; }
step() { printf '\n== %s\n' "$*"; }
PIDFILE="$WORK/results/bench.pid"
stop_bench() {  # docker exec does not forward signals: stop the benchmark (and so its server) in the container
    [ -f "$PIDFILE" ] && in_ctr "kill -TERM $(cat "$PIDFILE") 2> /dev/null" || true
    rm -f "$PIDFILE"
}
trap 'stop_bench; exit 130' INT TERM
mkdir -p "$WORK/results"
exec > >(tee -a "$WORK/repro.log") 2>&1
echo "######## gpt_oss_build_and_repro $(date -u +%FT%TZ) WORK=$WORK CONFIGS=$CONFIGS N=$N"

step "0. checks"
case "$PROV_SRC" in "$HOST_HOME"/*) ;; *) echo "the checkout must be under HOST_HOME=$HOST_HOME (mounted at $CTR_HOME)" >&2; exit 1 ;; esac
case "$WORK" in "$HOST_HOME"/*) ;; *) echo "WORK must be under HOST_HOME=$HOST_HOME" >&2; exit 1 ;; esac
case "$PROVIDERS" in "$HOST_HOME"/*) ;; *) echo "PROVIDERS must be under HOST_HOME=$HOST_HOME" >&2; exit 1 ;; esac
docker inspect -f '{{.State.Running}}' "$CONTAINER" 2> /dev/null | grep -q true || docker start "$CONTAINER" > /dev/null
in_ctr "python3 -c 'import vllm, sys; print(\"vllm\", vllm.__version__); sys.exit(0 if vllm.__version__.startswith((\"0.31\", \"0.27\")) else 1)'" \
    || { echo "the container needs vLLM 0.31.x or 0.27.x" >&2; exit 1; }
in_ctr "test -x $ROCM70/bin/hipcc && test -d $CK_DIR/include && test -f $MODEL/config.json" \
    || { echo "missing in the container: ROCM70=$ROCM70, CK_DIR=$CK_DIR or MODEL=$MODEL" >&2; exit 1; }
in_ctr "HIP_VISIBLE_DEVICES=$DEVICES python3 -c 'import torch; n = torch.cuda.device_count(); \
cu = [torch.cuda.get_device_properties(i).multi_processor_count for i in range(n)]; print(\"devices\", n, \"CUs\", cu); \
import sys; sys.exit(0 if n == 2 and all(c == 128 for c in cu) else 1)'" \
    || { echo "DEVICES=$DEVICES must be the two DPX partitions (128 CUs each) of one MI355X in DPX+NPS2" >&2; exit 1; }

step "1. provider exports (state ~100 GB each)"
for c in $CONFIGS; do
    if [ -f "$PROVIDERS/$c/state/manifest.json" ] && [ -f "$PROVIDERS/$c/src/kernel.hip" ]; then
        echo "$c: $PROVIDERS/$c"
        continue
    fi
    [ -n "${FLEET_REPO:-}" ] || { echo "no export in $PROVIDERS/$c: set PROVIDERS, or FLEET_REPO to create it" >&2; exit 1; }
    case $c in *_bf16) b=${c%_bf16}; lm=bf16; ar=466 ;; *) b=$c; lm=mxfp4; ar=32769 ;; esac
    PROVIDERS="$PROVIDERS" LM=$lm ARENA=$ar bash "$FLEET_REPO/scripts/nps2/aitermk/exp_run.sh" "$b"
    [ -f "$PROVIDERS/$c/state/manifest.json" ] || { echo "export of $c failed" >&2; exit 1; }
done

step "2. provider DSOs (the provider's CMake, ROCm 7.0)"
CFG_ROOT="$WORK/configs"
for c in $CONFIGS; do
    nm=$(name_of "$c")
    in_ctr "python3 $(ctr "$PROV_SRC/tools/import_export.py") $(ctr "$PROVIDERS/$c") $(ctr "$CFG_ROOT/$nm") \
        --check $(ctr "$PROV_SRC/configs/$nm")"
done
NAMES=$(for c in $CONFIGS; do name_of "$c"; done | paste -sd';')
TARGETS=$(for c in $CONFIGS; do echo -n "aitermk_provider_gpt_oss_gfx950_fleet_mpk_$(name_of "$c") "; done)
in_ctr "cd $(ctr "$ROOT") && cmake -S aitermk/megakernels/fleet/src/gpt_oss_gfx950_mpk -B $(ctr "$WORK/build") -GNinja \
    -DFLEET_MPK_ROCM_ROOT=$ROCM70 -DFLEET_CK_DIR=$CK_DIR -DFLEET_MPK_CONFIG_ROOT=$(ctr "$CFG_ROOT") \
    '-DFLEET_MPK_CONFIGS=$NAMES' > /dev/null && cmake --build $(ctr "$WORK/build") --target $TARGETS"
SO_DIR="$WORK/build"
ls "$SO_DIR"/libaitermk_provider_gpt_oss_gfx950_fleet_mpk_*.so

step "3. vLLM adapter (fleet_vllm)"
in_ctr "pip install -q --no-deps -e $(ctr "$ROOT/aitermk/engine_adapters/vllm") && pip list 2> /dev/null | grep -i aitermk-fleet-vllm"

step "4. vllm serve + GSM8K-128 (each config ~5 min)"
for c in $CONFIGS; do
    nm=$(name_of "$c")
    in_ctr "HIP_VISIBLE_DEVICES=$DEVICES python3 $(ctr "$SCRIPTS/gsm128_serve_bench.py") \
        --provider $(ctr "$SO_DIR/libaitermk_provider_gpt_oss_gfx950_fleet_mpk_$nm.so") --state $(ctr "$PROVIDERS/$c/state") \
        --refs $(ctr "$SCRIPTS/data/gsm128_ref_$c.json") --model $MODEL --n $N --port $PORT \
        --out $(ctr "$WORK/results/$c.json") --dump $(ctr "$WORK/results/dump_$c") \
        --pidfile $(ctr "$PIDFILE")" > "$WORK/results/$c.log" 2>&1 \
        || { echo "$c failed: see $WORK/results/$c.log" >&2; tail -n 20 "$WORK/results/$c.log" >&2; exit 1; }
    rm -f "$PIDFILE"
    grep -a 'GSM128\] SUMMARY' "$WORK/results/$c.log" | cut -c1-260
done

step "5. results (output tokens/s for one request, pooled over the questions)"
python3 - "$WORK/results" $CONFIGS << 'PYEOF'
import json, os, sys
d, cfgs = sys.argv[1], sys.argv[2:]
label = {"ns": "No speculation", "e3": "EAGLE3", "df8": "DFlash, 8-token block"}
label.update({c + "_bf16": v + " (BF16 head)" for c, v in list(label.items())})
label.update({c: v + " (MXFP4 head)" for c, v in list(label.items()) if not c.endswith("_bf16")})
# served through vllm serve 0.31.0: BF16 head = mean of two serves of builds sized for these questions; MXFP4 head =
# the released 32K builds
published = {"ns_bf16": 879, "e3_bf16": 1410, "df8_bf16": 1215, "ns": 966, "e3": 1472, "df8": 1236}
print("%-34s %16s %18s   %s" % ("config", "libAiterMK serve", "published", "tokens identical"))
for c in cfgs:
    s = json.load(open(os.path.join(d, c + ".json")))["summary"]
    print("%-34s %16.0f %18d   %d/%d" % (label[c], s["decode_tok_s"], published[c], s["tokens_same"],
                                        s["questions"]))
print("\nPublished: MI355X in DPX+NPS2, vLLM 0.31.0, Oct 18 2026 (BLOG_REPRODUCTION.md; BF16 head: "
      "PRECISION_MATCHED_COMPARISON.md).")
print("The same build varies by about 1-2% between server")
print("processes on DPX+NPS2, so compare within that.")
PYEOF
echo "######## done $(date -u +%FT%TZ): $WORK/results"
