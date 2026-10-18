# aitermk-vllm-plugin

A vLLM general plugin (`vllm.general_plugins` entry point) that runs AITerMK megakernels inside stock vLLM. It
requires no vLLM source changes. The package is `aitermk_vllm_plugin`.

## Qwen3.8-27B + DFlash2 (gfx950)

A speculative step of one plain greedy request runs one megakernel round. Plain: no structured output or logprobs,
and nothing that changes the logits or the token choice (repetition / frequency / presence penalties, logit bias,
bad words, allowed token ids, min tokens, logits-processor arguments, a thinking budget, watermarking, or an
engine-wide logits processor); such a request stays on vLLM's own path, which applies it. The round:
- the MXFP4 megakernel verifies the 8 rows vLLM scheduled (the bonus token and 7 drafts) over vLLM's paged KV
  cache and GDN conv / recurrent state caches, in place of vLLM's target forward; vLLM's rejection sampler
  samples from its logits;
- the accepted prefix is committed into vLLM's caches exactly where vLLM's own speculative step would leave it
  (recurrent state in slot `num_accepted - 1`, the conv window, the rows' K / V). vLLM's hybrid allocator
  splits the layers into KV-cache groups (for Qwen3.8-27B: 10 of GDN layers, 4 of attention layers), each with
  its own block table, so every layer reads and writes through its own group's state slots and blocks;
- DFlash2Mono drafts the next 7 tokens in place of vLLM's DFlash2 drafter.

Prefill, batches of several requests, sampling and everything else stay on vLLM's own path (Model Runner V2,
its DFlash2 speculator). A request is served by the megakernel only if its drafter saw it from its first
prefill chunk on, and it stays there until it takes a step on vLLM's path.

```bash
pip install --no-deps aitermk/megakernels/qwen38_27b aitermk/engine_adapters/vllm
export AITERMK_VLLM_DFLASH_AOT=<DFlash2Mono export: *.hsaco + manifest.json>
export AITERMK_QWEN38_CACHE=<code-object cache>       # + AITERMK_QWEN38_FLYDSL=<FlyDSL frontend> for the first build
vllm serve Qwen/Qwen3.8-27B --language-model-only \
  --speculative-config '{"method": "dflash", "model": "<DFlash2 draft>", "num_speculative_tokens": 7}' \
  --attention-backend TRITON_ATTN --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching \
  --max-num-seqs 1 --max-model-len 40960
```

The plugin fails closed at model load unless the deployment matches what the megakernel reads: DFlash with 7
speculative tokens, TP1, `--mamba-ssm-cache-dtype float32` (the kernel's state is fp32 `[48][128 v][128 k]`),
no prefix caching (`mamba_cache_mode` none: 8 state slots per request), and a TRITON_ATTN bf16 KV cache. The
drafter export's maximum position must cover `--max-model-len` + 8.

- `AITERMK_VLLM_DFLASH_CHECK=N`: the first N rounds also run vLLM's own (BF16) target forward of the same
  block; the log compares the logits (each row's pick, the reference's logit gap to it) and the DFlash aux
  streams.
- `AITERMK_VLLM_EAGER=1`: keeps rounds out of the graph.

### How it hooks vLLM

`register` wraps Model Runner V2 (`vllm.v1.worker.gpu.model_runner.GPUModelRunner`): `load_model` (attach the
megakernel next to vLLM's model, so vLLM's memory profile sees it), `execute_model` (route the step from the
scheduler output's host values), `prepare_inputs` and the model state's `prepare_attn` (keep the step's input
batch and attention metadata), the target's cudagraph replays and `forward` (the verify), `compute_logits` (its
logits) and the DFlash2 speculator's `propose` (the commit, the drafter context, the next draft; on a prefill
step, the prompt's aux hidden states into the megakernel drafter). vLLM's own bookkeeping (positions, block
tables, the GDN `num_accepted_tokens`, the async output copy) runs unchanged; the async copy of a round's
tokens starts before the drafter tail, as in the SGLang plugin's head / tail split.

## Measured (1x MI350X, B = 1, greedy, per-request mean tok/s)

The DFlash benchmark's prompt sets (each test split shuffled with seed 42, the first 128; MT-Bench's 80 first
turns), reasoning effort xhigh, up to 2,048 tokens, one request at a time (2026-10-16, vLLM 3f1bcfd82 from
vLLM's ROCm 7.2.3 nightly wheels, run in `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261009`'s ROCm 7.2.4
userspace, `scripts/build_and_repro.sh`). This is an MI350X, not the MI355X of the published ATOM / SGLang
numbers (1,105 / 924 / 846 and 1,102 / 925 / 843); SGLang + its plugin on the same MI350X is the matched
reference:

| set | vLLM + this plugin (2 runs) | SGLang + plugin, same GPU (2 runs) | vLLM's own DFlash2 (BF16) |
|---|---:|---:|---:|
| GSM8K (128) | 997 / 1,002 | 1,020 / 1,009 | 318 |
| HumanEval (128) | 836 / 843 | 849 / 852 | 266 |
| MT-Bench (80) | 773 / 755 | 776 / 764 | 239 |
| ms per round | 5.61-5.65 | | 17.9-18.2 |

The same on ROCm 10.0 (`rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0`, vLLM 49cb2bbf7 built
from source against it): 1,011 / 836 / 757 with this plugin, vLLM's own DFlash2 (BF16) 321 on GSM8K. The table
above is vLLM's ROCm 7.2.3 wheels on ROCm 7.2.4 userspace.

Acceptance length is the same on both vLLM paths (GSM8K 5.0-5.1, HumanEval 4.4-4.5, MT-Bench 3.7-3.9): the
megakernel drafter matches vLLM's own DFlash2 drafter. vLLM's MXFP4-at-load path (`--quantization mxfp4`) did
not run on this stack (aiter 0.1.25's `gemm_afp4wfp4` picks a tile needing 210 KB LDS), so there is no matched
MXFP4 baseline.

Lossless: a speculative run diverges from the megakernel's own greedy decode (`AITERMK_VLLM_DRAFT_OFF=1`,
every draft rejected) no earlier than two identical speculative runs diverge from each other (GSM8K first
difference at median token 36-44 vs 43; MT-Bench 48-51 vs 43): the megakernel is not bit-reproducible, and
speculation adds nothing to that noise. Against vLLM's BF16 path the outputs differ earlier (median token 19
on GSM8K), as MXFP4 weights do.

## Recommended setup (ROCm 10.0)

vLLM's prebuilt ROCm 10 image `vllm/vllm-openai-rocm:nightly-rocm100-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294`
(vLLM 0.32.0rc1.dev281+g49cb2bbf7, torch 2.12.0+rocm10.0.0): install the two packages with `pip install --no-deps`
and serve as above. `scripts/build_and_repro.sh` in it, from a fresh build of the code objects, on the MI350X:
1,003 / 846 / 747 (5.58-5.60 ms per round). vLLM main can also be built from source in
`rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0` (ROCm as TheRock wheels: add `rocm-sdk-devel`
10.0.0 from `https://stable.repo.amd.com/rocm/whl-next` and the environment of vLLM's
`docker/Dockerfile.rocm_base_gfx1250`). ROCm 7.2: vLLM's ROCm 7.2.3 nightly wheels of the same commits.

The published MI355X numbers this README compares with: ATOM in
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (ROCm 7.2.4); SGLang f825d72936 with the
plugin in `rocm/sgl-dev:v0.5.16-rocm720-mi35x-20260807` (ROCm 7.2.0).

## Reproduce

One command, on one MI355X with vLLM main (3f1bcfd82 or later: Model Runner V2 with DFlash2 drafts, e.g. the
ROCm nightly wheel of that commit) installed:

```bash
bash aitermk/engine_adapters/vllm/scripts/build_and_repro.sh
ARMS="megakernel stock" bash aitermk/engine_adapters/vllm/scripts/build_and_repro.sh   # + vLLM's own DFlash2 path
```

It installs the plugin, fetches Qwen3.8-27B and the DFlash2 draft, sets up the pinned FlyDSL compiler, builds
the megakernel and exports the drafter code objects (once), runs GSM8K, HumanEval and MT-Bench one request at a
time (reasoning effort xhigh, greedy, up to 2,048 tokens), and prints the results next to the published ATOM
and SGLang numbers. With the stock arm it also compares the outputs (`scripts/compare_outputs.py`). Set
`MODEL` / `DRAFT` to local checkpoints, and `FLYDSL_SRC` / `FLYDSL_WHEEL` to use local copies of FlyDSL;
everything goes under `./aitermk_blog_benchmark`.

In a container, and when running it again (as in `../sglang`):
- As root in a container, on a checkout that another user owns (for example a clone bind-mounted from the host), a
  `pip install --no-build-isolation` of the packages can stop with git's "detected dubious ownership": with build
  isolation off, an image that has setuptools-scm installed calls git on the checkout. Step 1 marks the checkout safe
  for its own install; for a manual install, run `git config --global --add safe.directory "$(pwd)"` first.
- Step 1 skips the install whenever both packages already import, so after updating the checkout in a container that
  has them, reinstall: `pip install --no-deps --no-build-isolation --force-reinstall aitermk/megakernels/qwen38_27b
  aitermk/engine_adapters/vllm`.
- Steps 2-4 also reuse what `WORK` holds: a checkpoint directory that has a `config.json`, `flydsl/` once it has its
  `READY` file, and `drafter-objects/` once it has its `manifest.json`. The FlyDSL setup works only with the Python
  that built it (the wheel's compiled MLIR bindings), so remove `$WORK/flydsl` before reusing `WORK` in an image with
  another Python, and remove `$WORK/drafter-objects` after a drafter change.
- Each run rewrites `$WORK/results/vllm-<arm>`; move it aside to keep earlier results.
- Code objects built in different images can differ, so compare code objects within one image.

## Docker

`Dockerfile` here builds the recommended ROCm 10.0 setup: `vllm/vllm-openai-rocm:nightly-rocm100-49cb2bbf7…` pinned
by digest (vLLM 0.32.0rc1.dev281+g49cb2bbf7, torch 2.12.0+rocm10.0.0), the two packages from the checkout (`pip
install --no-deps --no-build-isolation`, no git history in the image), FlyDSL set up for the image's Python, and the
megakernel and drafter code objects, prebuilt (`SETUP_ONLY=1` runs the script's steps 1, 3 and 4, which need no
GPU). No weights. From the repository root:

```bash
docker build -f aitermk/engine_adapters/vllm/Dockerfile -t aitermk-qwen38-vllm .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
    -v /path/to/models:/models -e MODEL=/models/Qwen3.8-27B -e DRAFT=/models/Qwen3.8-27B-DFlash2 \
    -v "$PWD/work:/work" aitermk-qwen38-vllm
```

The container runs `scripts/build_and_repro.sh`: the prompt sets, the results and any checkpoint given as a Hugging
Face id (`MODEL` / `DRAFT` default to the ids) go to the `/work` volume; FlyDSL and the code objects stay in the image
(`FLYDSL_DIR`, `DRAFTER_OBJECTS`, `AITERMK_QWEN38_CACHE`). Add `-e ARMS="megakernel stock"` for vLLM's own path too.
Checked on an MI350X (2026-10-17), from a fresh clone: a 35.8 GB image (the base's 35.3 GB plus 0.5) built in 75 s,
then 1,014 / 833 / 762 tok/s at 5.57-5.60 ms per round, outputs as lossless as above (first difference from the
megakernel's own greedy decode at median token 44 / 49 on GSM8K / MT-Bench).
