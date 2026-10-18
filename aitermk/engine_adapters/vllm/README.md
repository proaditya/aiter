# aitermk-fleet-vllm

A vLLM general plugin (`vllm.general_plugins` entry point `fleet_mk`) that serves GPT-OSS-120B on the persistent
megakernel in `../../megakernels/fleet/src/gpt_oss_gfx950_mpk` (MI355X, gfx950, in DPX+NPS2), with no vLLM source
changes. It takes plain greedy single requests up to 32K tokens and runs on vLLM 0.31.x or 0.27.x.

To build the providers, serve them and reproduce the published numbers, see the megakernel's
[README](../../megakernels/fleet/src/gpt_oss_gfx950_mpk/README.md) and run:

```bash
bash aitermk/engine_adapters/vllm/scripts/gpt_oss_build_and_repro.sh
```

By default it builds and serves the three released 32K builds (MXFP4 LM head), the blog's headline numbers;
`CONFIGS="ns_bf16 e3_bf16 df8_bf16"` serves the builds with the checkpoint's BF16 LM head, the precision vLLM runs,
behind the blog's matched-precision comparison.
