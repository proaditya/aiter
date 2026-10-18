# aitermk-sglang

An SGLang general plugin (`sglang.srt.plugins` entry point) that runs AITerMK megakernels inside stock SGLang. It requires no SGLang source changes.

## Qwen3.8-27B + DFlash2 (gfx950)

A decode step of one plain greedy request runs one megakernel round. Plain: no grammar, logprobs or hidden-state
returns, and nothing that changes the logits or the token choice (repetition / frequency / presence penalties, min
new tokens, logit bias, min-p, a custom logit processor, watermarking): the round accepts drafts against its own
argmax of the raw logits, so such a request stays on SGLang's own path, which applies them. The round:
- DFlash2Mono drafts 7 tokens;
- the MXFP4 megakernel verifies the 8 rows over SGLang's KV and GDN pools and picks each row's greedy token;
- the accepted prefix is committed back into SGLang's pools.

Prefill, batches of several requests, sampling and everything else stay on SGLang's DFLASH path. A request is served by the megakernel only if its drafter saw it from its first prefill chunk on.

```bash
pip install aitermk/megakernels/qwen38_27b aitermk/engine_adapters/sglang
export SGLANG_DFLASH_MEGAKERNEL_AOT=<DFlash2Mono export: *.hsaco + manifest.json>
export AITERMK_QWEN38_CACHE=<code-object cache>       # + AITERMK_QWEN38_FLYDSL=<FlyDSL frontend> for the first build
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm DFLASH --speculative-draft-model-path <DFlash2 draft> --speculative-num-draft-tokens 8 \
  --mamba-ssm-dtype float32 --disable-radix-cache --max-running-requests 1 \
  --attention-backend triton --mm-attention-backend sdpa --skip-server-warmup
```

- `SGLANG_DFLASH_MEGAKERNEL_CHECK=N`: the first N rounds also run SGLang's own verify of the same block, and the log compares both.
- `AITERMK_SGL_EAGER=1`: keeps rounds out of the graph.

## Measured (1× MI355X, B = 1, greedy, per-request mean tok/s)

The DFlash benchmark's prompt sets (each test split shuffled with seed 42, the first 128; MT-Bench's 80 first
turns), reasoning effort xhigh, up to 2,048 tokens, one request at a time (2026-10-14):

| set | SGLang + this plugin (MXFP4 megakernel) |
|---|---:|
| GSM8K (128) | 1,102 |
| HumanEval (128) | 925 |
| MT-Bench (80) | 843 |

ATOM serves the same kernels at the same speed (1,105 / 924 / 846). On one MI350X (2026-10-16, SGLang
59eb71a831) the same script gave 1,009 / 852 / 764, next to 997 / 836 / 773 for vLLM with its plugin
(`../vllm`) on that GPU (ROCm 7.2.4), and on ROCm 10.0 (`lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261016`,
SGLang 34c4ba14c8) 1,017 / 866 / 767. SGLang's own DFLASH path runs this dense
model in BF16 only (306 tok/s on GSM8K in an earlier 8-prompt run), so there is no matched MXFP4 baseline here.

Setups: the published numbers above are SGLang f825d72936 in `rocm/sgl-dev:v0.5.16-rocm720-mi35x-20260807`
(ROCm 7.2.0), and ATOM's in `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (ROCm 7.2.4). The
MI350X checks: SGLang 59eb71a831 in `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261009` (ROCm 7.2.4) and
SGLang 34c4ba14c8 in `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261016` (ROCm 10.0).

## Recommended setup (ROCm 10.0)

`lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261016` (SGLang 0.5.22.dev20261016+g34c4ba14c8, ROCm 10.0.0, torch
2.11.0+rocm10.0.0): `pip install --no-deps --no-build-isolation` the two packages and serve as above.

## Reproduce

One command, on one MI355X with SGLang main installed: f825d72936 or later, including SGLang after its
request-KV refactor (checked on 59eb71a831, 2026-10-09, where `Req.req_pool_idx` moved to `Req.kv`):

```bash
bash aitermk/engine_adapters/sglang/scripts/build_and_repro.sh
```

It installs the plugin, fetches Qwen3.8-27B and the DFlash2 draft, sets up the pinned FlyDSL compiler, builds
the megakernel and exports the drafter code objects (once), runs GSM8K, HumanEval and MT-Bench one request at a
time, and prints the results next to the published ones. Set `MODEL` / `DRAFT` to local checkpoints, and
`FLYDSL_SRC` / `FLYDSL_WHEEL` to use local copies of FlyDSL; everything goes under `./aitermk_blog_benchmark`.

Checked from scratch on a fresh MI355X (2026-10-15): the built code objects are bit-identical to the ones behind
the published numbers, and the run gave 1,106 / 923 / 836 tok/s.

Checked again from a fresh clone on another MI355X (2026-10-17): 1,106 / 953 / 846 tok/s in
`lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260913` (ROCm 7.2.0) at ab6afd4, and in the recommended ROCm 10.0 image
1,121 / 934 / 841 at ab6afd4 and 1,126 / 930 / 838 at 231cf19; acceptance length 5.0 / 4.4-4.5 / 3.7.

In a container, and when running it again:
- As root in a container, on a checkout that another user owns (for example a clone bind-mounted from the host),
  a `pip install --no-build-isolation` of the packages stops with git's "detected dubious ownership": the image's
  setuptools-scm calls git on the checkout. Step 1 marks the checkout safe for its own install; for a manual install
  (as in the recommended setup), run `git config --global --add safe.directory "$(pwd)"` first.
- Step 1 skips the install whenever both packages already import, so after updating the checkout in a container that
  has them, reinstall: `pip install --no-deps --no-build-isolation --force-reinstall aitermk/megakernels/qwen38_27b
  aitermk/engine_adapters/sglang`.
- Steps 2-4 also reuse what `WORK` holds: a checkpoint directory that has a `config.json`, `flydsl/` once it has its
  `READY` file, and `drafter-objects/` once it has its `manifest.json`. The FlyDSL setup works only with the Python
  that built it (the wheel's compiled MLIR bindings), so remove `$WORK/flydsl` before reusing `WORK` in an image with
  another Python, and remove `$WORK/drafter-objects` after a drafter change.
- Each run rewrites `$WORK/results/sglang`; move it aside to keep earlier results.
- Code objects built in different images can differ (the v0.5.19 and the ROCm 10.0 images built different ones from
  the same sources, and both reproduced the speeds), so compare code objects within one image.


## Docker

`Dockerfile` here builds the recommended ROCm 10.0 setup: `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261016`
pinned by digest (SGLang 0.5.22.dev20261016+g34c4ba14c8, torch 2.11.0+rocm10.0.0), the two packages from the
checkout (`pip install --no-deps --no-build-isolation`, no git history in the image), FlyDSL set up for the image's
Python, and the megakernel and drafter code objects, prebuilt (`SETUP_ONLY=1` runs the script's steps 1, 3 and 4,
which need no GPU). No weights. From the repository root:

```bash
docker build -f aitermk/engine_adapters/sglang/Dockerfile -t aitermk-qwen38-sglang .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
    -v /path/to/models:/models -e MODEL=/models/Qwen3.8-27B -e DRAFT=/models/Qwen3.8-27B-DFlash2 \
    -v "$PWD/work:/work" aitermk-qwen38-sglang
```

The container runs `scripts/build_and_repro.sh`: the prompt sets, the results (with each output's text) and any
checkpoint given as a Hugging Face id (`MODEL` / `DRAFT` default to the ids) go to the `/work` volume; FlyDSL and the
code objects stay in the image (`FLYDSL_DIR`, `DRAFTER_OBJECTS`, `AITERMK_QWEN38_CACHE`). Checked on an MI350X
(2026-10-17), from a fresh clone: a 42.7 GB image (the base's 42.2 GB plus 0.5) built in 78 s, then 1,012 / 854 / 780
tok/s, outputs lossless (first difference from the megakernel's own greedy decode at median token 48 / 54 on GSM8K /
MT-Bench, against 43 between two identical runs).
