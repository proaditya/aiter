# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Qwen3.8-27B + DFlash2 in SGLang with the aitermk plugin, ROCm 10.0 (gfx950): SGLang's ROCm 10 image
# (SGLang 0.5.22.dev20261016+g34c4ba14c8, torch 2.11.0+rocm10.0.0), the plugin and the Qwen3.8-27B kernels package
# from this checkout, the pinned FlyDSL (frontend 19418894 over the flydsl 0.4.0.dev963 wheel) and the megakernel
# and drafter code objects, prebuilt (compile-only: no GPU at build time). No weights: the model and the draft come
# in at run time. From the repository root:
#
#   docker build -f aitermk/engine_adapters/sglang/Dockerfile -t aitermk-qwen38-sglang .
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
#       -v /path/to/models:/models -e MODEL=/models/Qwen3.8-27B -e DRAFT=/models/Qwen3.8-27B-DFlash2 \
#       -v /path/to/work:/work aitermk-qwen38-sglang
#
# The default command is scripts/build_and_repro.sh: the benchmark next to the published numbers, results in /work.

# lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261016
FROM lmsysorg/sglang-rocm@sha256:7faebdce08079bf254706487637565fc842471d7f3be8e06efc98cf60a6c8980

WORKDIR /opt/aitermk
COPY aitermk/megakernels/qwen38_27b aitermk/megakernels/qwen38_27b
COPY aitermk/engine_adapters/sglang aitermk/engine_adapters/sglang
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        aitermk/megakernels/qwen38_27b aitermk/engine_adapters/sglang

# steps 3 and 4 of the script, here: FlyDSL for this Python, the megakernel and the drafter code objects
ENV FLYDSL_DIR=/opt/qwen38/flydsl \
    DRAFTER_OBJECTS=/opt/qwen38/drafter-objects \
    AITERMK_QWEN38_CACHE=/opt/qwen38/kernel-cache
RUN SETUP_ONLY=1 WORK=/tmp/setup bash aitermk/engine_adapters/sglang/scripts/build_and_repro.sh \
    && rm -rf /tmp/setup "$FLYDSL_DIR/wheel/x" /root/.cache/pip

# checkpoints downloaded by id, prompt sets, results: on the /work volume
ENV WORK=/work HF_HOME=/work/hf-cache
VOLUME /work
ENTRYPOINT []
CMD ["bash", "aitermk/engine_adapters/sglang/scripts/build_and_repro.sh"]
