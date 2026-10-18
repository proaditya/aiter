#!/bin/bash
# GSM8K-128 baseline for stock vLLM, the comparison column of the providers' speed table: no speculation, EAGLE3 with
# 3 and 5 drafts and DFlash with 7 drafts, one GPU each, run in parallel on four GPUs of one MI355X node in SPX/NPS1
# (vLLM runs one model per GPU). Offline LLM engine, greedy, 256 new tokens, EOS ignored, the megakernel's prompt
# token ids (../data/gsm128_ref_ns.json), decode tok/s from GPU-timeline step times (vllm_step_timing.py).
#
#   MODEL=/models/gpt-oss-120b DFLASH=/models/z-lab_gpt-oss-120b-DFlash \
#   EAGLE3=/models/nvidia_gpt-oss-120b-Eagle3-v3 bash gsm128_vllm_baseline.sh
#
# The megakernel's GEMM precision (MXFP4 weights x FP8 inputs in q/k/v/o and the MoE, BF16 router and LM head, BF16
# KV cache) in vLLM: MODEL=<amd/gpt-oss-120b-w-mxfp4-a-fp8-kv-fp8-fp8attn-no_lmhead_router> EXTRA_ENV=KV_DTYPE=bfloat16
# (that checkpoint asks for an FP8 KV cache; the megakernel's is BF16).
#
# env: IMG (default vllm/vllm-openai-rocm:v0.31.0), GPUS (default "0 1 2 3"), N (questions, default 128),
# OUT (default ./vllm_gsm128), TAG (result name prefix, default v31), EXTRA_ENV (more K=V for vllm_step_timing.py).
# Each run gets a fresh container; every path is mounted at the same path inside it. Run on idle GPUs only.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=${IMG:-vllm/vllm-openai-rocm:v0.31.0}
MODEL=${MODEL:?set MODEL (gpt-oss-120b checkpoint directory)}
DFLASH=${DFLASH:?set DFLASH (z-lab/gpt-oss-120b-DFlash directory)}
EAGLE3=${EAGLE3:?set EAGLE3 (nvidia/gpt-oss-120b-Eagle3-v3 directory)}
read -r -a G <<< "${GPUS:-0 1 2 3}"
[ ${#G[@]} -ge 4 ] || { echo "need four GPUs in GPUS"; exit 1; }
N=${N:-128}; TAG=${TAG:-v31}; OUT=${OUT:-$PWD/vllm_gsm128}; EXTRA_ENV=${EXTRA_ENV:-}
REF=$HERE/../data/gsm128_ref_ns.json
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
C=gsm128_vllm_${TAG}_$$
MNT=""
for p in "$HERE/.." "$OUT" "$MODEL" "$DFLASH" "$EAGLE3"; do MNT="$MNT -v $(cd "$p" && pwd):$(cd "$p" && pwd)"; done
docker run -d --name $C --entrypoint /bin/bash --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64g $MNT $IMG -c "sleep infinity" \
  > /dev/null || exit 1
trap 'docker rm -f $C > /dev/null 2>&1' EXIT
docker exec $C python3 -c 'import vllm; print("vllm", vllm.__version__)' < /dev/null
# the SemiAnalysis MI355X gpt-oss settings (AITER on; for the W4A8 checkpoint this selects AITER's MXFP4 x FP8 MoE)
ENVB="HSA_NO_SCRATCH_RECLAIM=1 AMDGCN_USE_BUFFER_OPS=0 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_ROPE=1"
ENVB="$ENVB VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 HF_HUB_OFFLINE=1 NEW=256 BD_CC=1 MODEL=$MODEL REF=$REF N=$N $EXTRA_ENV"
run() {  # tag gpu K method draft
  local t=${TAG}_$1
  docker exec $C bash -c "mkdir -p /tmp/$t && cd /tmp/$t && env $ENVB HIP_VISIBLE_DEVICES=$2 VLLM_CACHE_ROOT=/tmp/$t/vc \
    TORCHINDUCTOR_CACHE_DIR=/tmp/$t/ti TRITON_CACHE_DIR=/tmp/$t/tc K=$3 METHOD=$4 DRAFT=$5 OUT=$OUT/$t.json \
    timeout 3600 python3 $HERE/vllm_step_timing.py" < /dev/null > "$OUT/$t.log" 2>&1
  echo "$(date -u +%T) $t rc=$? $(grep -c '^\[BD\] q' "$OUT/$t.log") prompts"
}
run k0 ${G[0]} 0 dflash "$DFLASH" &
run df7 ${G[1]} 7 dflash "$DFLASH" &
run e3k3 ${G[2]} 3 eagle3 "$EAGLE3" &
run e3k5 ${G[3]} 5 eagle3 "$EAGLE3" &
wait
python3 "$HERE/gsm128_vllm_summary.py" "$OUT/${TAG}_k0.json" "$OUT/${TAG}_e3k3.json" "$OUT/${TAG}_e3k5.json" \
  "$OUT/${TAG}_df7.json" --out "$OUT/summary_$TAG.json" | tee "$OUT/summary_$TAG.txt"
