from fleet_vllm.plugin import register
from fleet_vllm.spec import load_spec


def test_gpt_oss_spec_matches_aiter_fleet_kernel():
    spec = load_spec("GptOssForCausalLM")

    assert spec.num_layers == 36
    assert spec.page_size == 16
    assert spec.kv_cache_stride == 512
    assert spec.weights_per_layer == 13
    assert spec.ptrs_per_layer == 38
    assert spec.launch_symbol == "gpt_oss_120b_launch"
    assert spec.init_symbol == "gpt_oss_120b_init"


def test_plugin_is_inert_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("AITER_MK_FLEET_VLLM", raising=False)
    monkeypatch.delenv("REDLINE_VLLM_ENABLE", raising=False)
    register()
