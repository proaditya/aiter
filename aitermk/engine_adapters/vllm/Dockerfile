# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Qwen3.8-27B + DFlash2 in vLLM with the aitermk plugin, ROCm 10.0 (gfx950): vLLM's ROCm 10 nightly image of
# 49cb2bbf7 (vLLM 0.32.0rc1.dev281+g49cb2bbf7, torch 2.12.0+rocm10.0.0), the plugin and the Qwen3.8-27B kernels
# package from this checkout, the pinned FlyDSL (frontend 19418894 over the flydsl 0.4.0.dev963 wheel) and the
# megakernel and drafter code objects, prebuilt (compile-only: no GPU at build time). No weights: the model and the
# draft come in at run time. From the repository root:
#
#   docker build -f aitermk/engine_adapters/vllm/Dockerfile -t aitermk-qwen38-vllm .
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
#       -v /path/to/models:/models -e MODEL=/models/Qwen3.8-27B -e DRAFT=/models/Qwen3.8-27B-DFlash2 \
#       -v /path/to/work:/work aitermk-qwen38-vllm
#
# The default command is scripts/build_and_repro.sh: the benchmark next to the published numbers, results in /work.

# vllm/vllm-openai-rocm:nightly-rocm100-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294
FROM vllm/vllm-openai-rocm@sha256:23fbefde881488d439808e1de38cfbd87635e0045ddf346f90b02e59dc1360a3

WORKDIR /opt/aitermk
COPY aitermk/megakernels/qwen38_27b aitermk/megakernels/qwen38_27b
COPY aitermk/engine_adapters/vllm aitermk/engine_adapters/vllm
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        aitermk/megakernels/qwen38_27b aitermk/engine_adapters/vllm

# steps 3 and 4 of the script, here: FlyDSL for this Python, the megakernel and the drafter code objects
ENV FLYDSL_DIR=/opt/qwen38/flydsl \
    DRAFTER_OBJECTS=/opt/qwen38/drafter-objects \
    AITERMK_QWEN38_CACHE=/opt/qwen38/kernel-cache
RUN SETUP_ONLY=1 WORK=/tmp/setup bash aitermk/engine_adapters/vllm/scripts/build_and_repro.sh \
    && rm -rf /tmp/setup "$FLYDSL_DIR/wheel/x" /root/.cache/pip

# checkpoints downloaded by id, prompt sets, results: on the /work volume
ENV WORK=/work HF_HOME=/work/hf-cache
VOLUME /work
ENTRYPOINT []
CMD ["bash", "aitermk/engine_adapters/vllm/scripts/build_and_repro.sh"]
