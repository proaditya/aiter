"""What a fleet MPK provider can serve, as plain checks (no torch or vLLM imports).

A provider build runs one request at a time, greedily, inside a fixed token arena. Everything else is rejected
before it reaches the engine: an exception inside the
engine step stops the server.
"""

_GREEDY_EPS = 1e-5  # vLLM samples greedily below this temperature


def request_capacity(arena):
    """Longest prompt + max_tokens a build holds.

    The kernel ends a request at step + 1 >= its sequence bound (prompt + max_tokens + 1), and a bound above the
    compiled arena is ignored, so prompt + max_tokens must stay below the arena.
    """
    return arena - 1


def serving_config_errors(
    *,
    max_num_seqs,
    max_model_len,
    capacity,
    enable_prefix_caching,
    enforce_eager,
    async_scheduling,
    load_format,
    watermark=False,
    logits_processors=False,
):
    """Why a vLLM configuration cannot serve this provider (empty if it can)."""
    errors = []
    if max_num_seqs != 1:
        errors.append(
            f"--max-num-seqs {max_num_seqs}: one request at a time (use --max-num-seqs 1)"
        )
    if max_model_len > capacity:
        errors.append(
            f"--max-model-len {max_model_len}: this build holds a prompt plus max_tokens of at most "
            f"{capacity} tokens (use --max-model-len {capacity})"
        )
    if enable_prefix_caching:
        errors.append(
            "prefix caching is on: the provider needs every prompt token (use --no-enable-prefix-caching)"
        )
    if not enforce_eager:
        errors.append(
            "CUDA graphs are on: the stand-in model is not captured (use --enforce-eager)"
        )
    if async_scheduling:
        errors.append("async scheduling is on (use --no-async-scheduling)")
    if load_format != "dummy":
        errors.append(
            f"--load-format {load_format}: the provider brings its own weights (use --load-format dummy)"
        )
    if watermark:
        errors.append(
            "a watermark is configured: the provider decodes by argmax and cannot apply it "
            "(remove --watermark-config)"
        )
    if logits_processors:
        errors.append(
            "custom logits processors are configured: the provider returns no logits "
            "(remove --logits-processors)"
        )
    return errors


def sampling_errors(params):
    """Why the provider cannot serve a request with these vLLM SamplingParams (empty if it can).

    The kernel decodes by argmax and returns no logits, so only greedy decoding without logit changes, logprobs,
    forced tokens, a thinking budget or several sequences per request is served. Stop strings, stop tokens,
    ignore_eos and repetition detection are applied by vLLM's scheduler.
    """

    def get(name, default=None):
        return getattr(params, name, default)

    errors = []
    if (get("temperature") or 0.0) > _GREEDY_EPS:
        errors.append(f"temperature {get('temperature')}: greedy decoding only (temperature 0)")
    if (get("n") or 1) != 1:
        errors.append(f"n {get('n')}: one sequence per request")
    if get("logprobs") is not None:
        errors.append("logprobs are not available")
    if get("prompt_logprobs") is not None:
        errors.append("prompt_logprobs are not available")
    for name, neutral in (
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
    ):
        value = get(name)
        if value is not None and value != neutral:
            errors.append(f"{name} {value}: not supported")
    for name in (
        "logit_bias",
        "allowed_token_ids",
        "bad_words",
        "structured_outputs",
        "guided_decoding",
    ):
        if get(name):
            errors.append(f"{name}: not supported")
    if (get("min_tokens") or 0) > 0:
        errors.append(f"min_tokens {get('min_tokens')}: not supported")
    budget = get("thinking_token_budget")
    if budget is not None and budget >= 0:  # -1 means unlimited
        errors.append(f"thinking_token_budget {budget}: not supported")
    for name in ("logprob_token_ids", "prompt_logprob_token_ids"):
        if get(name) is not None:
            errors.append(f"{name}: logprobs are not available")
    if get("trace_decode_token_ids"):
        errors.append("trace_decode_token_ids: not supported")
    return errors
