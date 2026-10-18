# Qwen3.8-27B megakernel + DFlash2 drafter (FlyDSL)

Single-sequence speculative decoding for Qwen3.8-27B on gfx950 (MI355X): a persistent FlyDSL megakernel verifies 8 tokens through all 64 layers in one launch (MXFP4 weights), and a persistent FlyDSL DFlash2 drafter proposes them (MXFP8, MXFP4-screened LM head).

## Layout

`python/aitermk_qwen38/provider/` is engine-agnostic.

| path | contents |
|---|---|
| `kernels/` | the verify kernel's FlyDSL sources and generators (`native_*.py`, `gen_replay_records.py`), the code-object build (`build.py`: compile-only, cached by source hash) and the HIP loader (`hip.py`) |
| `weights.py`, `mxfp4.py` | Hugging Face checkpoint to the kernel's MXFP4 layout (~15 s on the GPU) |
| `binding.py`, `dims.py` | the kernel's pointer table and buffers |
| `launch_kernels.py`, `state_kernels.py`, `verdict.py` | the pre-launch bookkeeping kernel; the GDN state commit from replay records; the target's greedy token per row |
| `sgl_engine.py`, `sgl_state_kernels.py` | SGLang's pool layout (token-slot KV, conv `[layer, slot, 10240, 3]`, recurrent state `[layer, slot, 48, 128 v, 128 k]` fp32 in place) |
| `vllm_engine.py`, `vllm_state_kernels.py` | vLLM's cache layout (paged KV through a block table, per-layer conv `[slot, 3 + 7, 10240]` and recurrent state `[slot, 48, 128 v, 128 k]` fp32 in the request's 8 speculative state slots, picked by `num_accepted_tokens`) |
| `drafter/` | DFlash2Mono (`dflash2_mono.py` and its dependencies), its AOT exporter and launcher (`aot_runtime.py`), and the drafter kernels' FlyDSL sources (`kernels_src/`) |

Engine adapters live under `../../engine_adapters/` (SGLang: `engine_adapters/sglang`, vLLM: `engine_adapters/vllm`).

## Building the code objects

Builds are compile-only and need no GPU. The verify kernel:

```bash
AITERMK_QWEN38_FLYDSL=<FlyDSL frontend 19418894> AITERMK_QWEN38_CACHE=<cache dir> \
  python -c "from aitermk_qwen38.provider.kernels.build import build; print(build('m8'))"
```

The drafter (exports `*.hsaco` and `manifest.json` for the context lengths you serve):

```bash
FLY_KERNELS_ROOT=<FlyDSL kernels/> python python/aitermk_qwen38/provider/drafter/aot_runtime.py export --out <dir> --P 40968,131136
```

Serving loads only the code objects; FlyDSL is not needed at run time.
