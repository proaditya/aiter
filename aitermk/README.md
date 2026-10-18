# aitermk

- `megakernels/<model>/`: one model's FlyDSL kernel sources, their compile-only builds (code objects + manifests; FlyDSL is not needed at serving time), and an engine-agnostic Python runtime that binds them to an engine's KV and state pools.
- `engine_adapters/<engine>/`: plugins that route an engine's eligible steps to a megakernel and leave the rest of the engine unchanged.

Current contents: `megakernels/qwen38_27b`, `engine_adapters/sglang` and `engine_adapters/vllm`.
