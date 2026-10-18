"""vLLM decode adapter for the Fleet GPT-OSS gfx950 megakernel provider."""

from .spec import ModelSpec, load_spec, supported_arches

__all__ = ["ModelSpec", "load_spec", "supported_arches"]
