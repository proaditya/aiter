"""vLLM general plugin entry point for the Fleet decode megakernel."""

from __future__ import annotations

import os

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if os.environ.get("AITER_MK_FLEET_VLLM") != "1":
        return
    if os.environ.get("REDLINE_VLLM_ENABLE") == "1":
        return
    if _REGISTERED:
        return
    _REGISTERED = True

    if os.environ.get("AITERMK_FLEET_MPK_PROVIDER"):
        # Native fleet MPK provider: the provider owns prefill, KV and weights (fleet_vllm.native).
        import logging

        from .native.model import register as register_native

        register_native()
        logging.getLogger(__name__).warning(
            "fleet_vllm: native fleet MPK provider registered (%s, state %s)",
            os.environ["AITERMK_FLEET_MPK_PROVIDER"],
            os.environ.get("AITERMK_FLEET_MPK_STATE"),
        )
        return

    from vllm import ModelRegistry

    from .greedy import install_greedy_argmax_fastpath
    from .model import _make_gptoss_class
    from .persistent_scheduler import install_persistent_scheduler_hooks

    install_greedy_argmax_fastpath()
    install_persistent_scheduler_hooks()

    try:
        from .backend import register_fleet_mk_backend

        register_fleet_mk_backend()
    except ImportError as exc:
        import logging

        logging.getLogger(__name__).info(
            "fleet_vllm: CUSTOM attention backend not registered (%s)",
            exc,
        )

    try:
        ModelRegistry.register_model("GptOssForCausalLM", _make_gptoss_class())
        import logging

        logging.getLogger(__name__).warning(
            "fleet_vllm: GptOssForCausalLM decode mixin registered (%s)",
            os.environ.get("AITERMK_FLEET_PROVIDER") or os.environ.get("FLEET_MK_SO"),
        )
    except Exception as exc:
        import logging

        logging.getLogger(__name__).info(
            "fleet_vllm: GptOssForCausalLM not registered (%s)",
            exc,
        )
