import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fleet_vllm.native.limits import request_capacity, sampling_errors, serving_config_errors

MPK = Path(__file__).resolve().parents[3] / "megakernels" / "fleet" / "src" / "gpt_oss_gfx950_mpk"
CONFIGS = ("ns", "eagle3_b6", "dflash_b8")
GOOD_CONFIG = dict(
    max_num_seqs=1,
    max_model_len=465,
    capacity=465,
    enable_prefix_caching=False,
    enforce_eager=True,
    async_scheduling=False,
    load_format="dummy",
)


def _import_export():
    spec = importlib.util.spec_from_file_location(
        "import_export", MPK / "tools" / "import_export.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _greedy(**overrides):
    params = dict(
        temperature=0.0,
        n=1,
        logprobs=None,
        prompt_logprobs=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words=[],
        structured_outputs=None,
        min_tokens=0,
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def test_capacity_leaves_room_for_the_kernels_sequence_bound():
    # the kernel ends a request at step + 1 >= prompt + max_tokens + 1, and ignores a bound above the arena
    assert request_capacity(466) == 465


def test_a_supported_serving_configuration_passes():
    assert serving_config_errors(**GOOD_CONFIG) == []


@pytest.mark.parametrize(
    ("override", "flag"),
    [
        ({"max_num_seqs": 2}, "--max-num-seqs 1"),
        ({"max_model_len": 4096}, "--max-model-len 465"),
        ({"enable_prefix_caching": True}, "--no-enable-prefix-caching"),
        ({"enforce_eager": False}, "--enforce-eager"),
        ({"async_scheduling": True}, "--no-async-scheduling"),
        ({"load_format": "auto"}, "--load-format dummy"),
        ({"watermark": True}, "--watermark-config"),
        ({"logits_processors": True}, "--logits-processors"),
    ],
)
def test_each_unsupported_serving_option_names_its_fix(override, flag):
    errors = serving_config_errors(**{**GOOD_CONFIG, **override})
    assert len(errors) == 1
    assert flag in errors[0]


def test_greedy_requests_are_served():
    assert sampling_errors(_greedy()) == []
    assert sampling_errors(_greedy(temperature=1e-6)) == []  # vLLM decodes greedily below 1e-5
    assert sampling_errors(SimpleNamespace()) == []  # absent fields mean vLLM's defaults
    assert sampling_errors(_greedy(thinking_token_budget=-1)) == []  # -1 means unlimited


@pytest.mark.parametrize(
    "override",
    [
        {"temperature": 0.7},
        {"n": 2},
        {"logprobs": 5},
        {"prompt_logprobs": 1},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
        {"repetition_penalty": 1.2},
        {"logit_bias": {"42": 1.0}},
        {"allowed_token_ids": [1, 2]},
        {"bad_words": ["x"]},
        {"structured_outputs": {"json": {}}},
        {"min_tokens": 4},
        {"thinking_token_budget": 8},
        {"logprob_token_ids": [1]},
        {"prompt_logprob_token_ids": [1]},
        {"trace_decode_token_ids": [1, 2]},
    ],
)
def test_requests_the_kernel_cannot_decode_are_rejected(override):
    assert len(sampling_errors(_greedy(**override))) == 1


def test_import_keeps_only_the_compile_flags():
    ie = _import_export()
    cmd = ["hipcc", "-x", "hip", "test.cu", "-O3", "-DMPK_X=1", "-I/inc", "-o", "out.so", "-shared", "-fPIC",
           "--save-temps", "-L/lib", "-lz", "--offload-arch=gfx950"]  # fmt: skip
    assert ie.flags_of(cmd) == ["-O3", "-DMPK_X=1", "--offload-arch=gfx950"]


def test_check_ignores_only_the_build_id(tmp_path):
    ie = _import_export()
    for d, build in (("a", "fleet-mpk-20261016T053133Z"), ("b", "fleet-mpk-20261015T221027Z")):
        (tmp_path / d).mkdir()
        for f in ie.CHECKED:
            (tmp_path / d / f).write_text(f'#define FLEET_MPK_BUILD_ID "{build}"\nint x;\n')
    assert ie.differing(tmp_path / "a", tmp_path / "b") == []
    (tmp_path / "b" / "kernel.hip").write_text("int y;\n")
    assert ie.differing(tmp_path / "a", tmp_path / "b") == ["kernel.hip"]


@pytest.mark.parametrize("config", CONFIGS)
def test_shipped_configs_carry_the_dpx_launch_fixes(config):
    flags = (MPK / "configs" / config / "compile_flags.txt").read_text().split()
    for fix in (
        "-DMPK_MULTI_DEV=2",
        "-DMPK_PLACE_SERIAL",
        "-DMPK_PLACE_GUARD=1000",
        "-DMPK_GATE_RESCUE=16384",
    ):
        assert fix in flags
    env = json.loads((MPK / "configs" / config / "env.json").read_text())
    assert env["MPK_TOK_HOST"] == "1"
    assert all(k.startswith("MPK_") for k in env)
