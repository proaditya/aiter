"""fleetq_plugin: make stock vLLM's gpt-oss compute with the fleet megakernel's precision (an accuracy experiment).

vLLM general plugin (entry point vllm.general_plugins: fleetq). Inert unless FQ_W or FQ_A is 1.
  FQ_W=1  after load_weights, round-trip every attention q/k/v and o projection weight and the LM head weight
          through fleet's MXFP4 quantizer (demo.py quantize_bf16_to_mxfp4 + its dequantization: E2M1 values,
          one E8M0 scale per 32 inputs, scale = ceil(log2(amax / 6)), midpoint rounding). The checkpoint keeps
          these in BF16 and stock vLLM uses them as BF16; the megakernel quantizes them to MXFP4.
  FQ_A=1  round-trip the inputs of those layers and of the MoE experts through fleet's FP8 activation quantizer
          (demo.py fp8_act_roundtrip: E4M3, one E8M0 scale per block, scale = ceil exponent of amax / 448).
          Blocks of 32 values, and FQ_EXP_BLOCK (default 128, the W13 prequant format) for the experts' input.
          Not emulated: the FP8 input of the experts' down projection (inside vLLM's fused MoE kernel).
  FQ_W_MODS  which weights FQ_W quantizes: qkv, o, lm, joined by , or + (default all three).
  FQ_ATT_FMT / FQ_LM_FMT  format of the q/k/v/o weights / of the LM head weight, default mxfp4 (fleet's quantizer):
             mse    MXFP4, but each block takes whichever of fleet's scale exponent and the next lower one (values past
                    6 x scale saturate) gives the smaller squared error
             fp6    MXFP6 E2M3 (max 7.5), scale exponent ceil(log2(amax / 7.5)); fp6mse likewise with the error choice
             fp8    MXFP8: E4M3 with one E8M0 scale per 32 inputs (fleet's activation quantizer applied to the weight)
  FQ_QKV_PARTS  quantize only these row blocks of the fused qkv_proj weight: q, k, v joined by + (default all).
  FQ_Q_FMT / FQ_KV_FMT / FQ_O_FMT  per-part formats for the q rows, the k and v rows, and o_proj (default FQ_ATT_FMT).
  FQ_A_LM    0 = leave the LM head's input in BF16 under FQ_A (default 1).
Use with --enforce-eager. To load it, put this directory on PYTHONPATH (fleetq_plugin-0.1.dist-info carries the entry
point) and set VLLM_PLUGINS=fleetq; the server log then prints one [FQ] line with what was converted."""
import os

_THR = (0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00)
_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _mxfp4_roundtrip_(w):
    import torch

    out_dim, in_dim = w.shape
    assert in_dim % 32 == 0, f"in_dim {in_dim} is not a multiple of 32"
    thr = torch.tensor(_THR, device=w.device, dtype=torch.float32)
    mag = torch.tensor(_MAG, device=w.device, dtype=torch.float32)
    for r0 in range(0, out_dim, 4096):
        blk = w[r0:r0 + 4096].float().reshape(-1, in_dim // 32, 32)
        amax = blk.abs().amax(dim=-1, keepdim=True)
        e = (torch.ceil(torch.log2(amax.clamp(min=2 ** -126) / 6.0)) + 127.0).clamp(0, 254)
        e = torch.where(amax == 0, torch.zeros_like(e), e)
        sv = torch.pow(2.0, e - 127.0)
        n = blk / sv.clamp(min=2 ** -126)
        q = mag[torch.bucketize(n.abs(), thr, right=True)] * n.sign()
        w[r0:r0 + 4096] = (q * sv).reshape(-1, in_dim).to(w.dtype)


def _fp8_roundtrip(x, block=32):
    import torch

    K = x.shape[-1]
    if K % block != 0:
        return x
    v = x.float().reshape(-1, K // block, block)
    amax = v.abs().amax(dim=-1, keepdim=True)
    u = (amax * (1.0 / 448.0)).contiguous().view(torch.int32)
    raw = (((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).to(torch.int32)).clamp(0, 255)
    scale = torch.where((amax == 0) | (raw == 0), torch.ones_like(amax), (raw << 23).view(torch.float32))
    q = (v / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(x.shape).to(x.dtype)


def _mxfp8_roundtrip_(w):
    for r0 in range(0, w.shape[0], 4096):
        w[r0:r0 + 4096] = _fp8_roundtrip(w[r0:r0 + 4096], 32)


_E2M3 = tuple(m / 8 for m in range(8)) + tuple((1 + m / 8) * 2 ** (e - 1) for e in (1, 2, 3) for m in range(8))


def _mx_roundtrip_(w, mag, mse):
    """Block-32 round trip on an arbitrary non-negative grid `mag` (last entry = max), round to nearest, saturating.
    mse: per block, also try the scale exponent one lower and keep whichever has the smaller squared error."""
    import torch

    out_dim, in_dim = w.shape
    assert in_dim % 32 == 0, f"in_dim {in_dim} is not a multiple of 32"
    m = torch.tensor(mag, device=w.device, dtype=torch.float32)
    thr = (m[1:] + m[:-1]) / 2
    gmax = float(mag[-1])

    def q_at(blk, e):
        sv = torch.pow(2.0, e - 127.0)
        n = blk / sv
        return m[torch.bucketize(n.abs(), thr, right=True)] * n.sign() * sv

    for r0 in range(0, out_dim, 4096):
        blk = w[r0:r0 + 4096].float().reshape(-1, in_dim // 32, 32)
        amax = blk.abs().amax(dim=-1, keepdim=True)
        e = (torch.ceil(torch.log2(amax.clamp(min=2 ** -126) / gmax)) + 127.0).clamp(1, 254)
        q = q_at(blk, e)
        if mse:
            q1 = q_at(blk, (e - 1).clamp(min=1))
            better = ((q1 - blk) ** 2).sum(-1, keepdim=True) < ((q - blk) ** 2).sum(-1, keepdim=True)
            q = torch.where(better, q1, q)
        q = torch.where(amax == 0, torch.zeros_like(q), q)
        w[r0:r0 + 4096] = q.reshape(-1, in_dim).to(w.dtype)


def _quantize_(w, fmt):
    if fmt == "mxfp4":
        _mxfp4_roundtrip_(w)
    elif fmt == "mse":
        _mx_roundtrip_(w, _MAG, True)
    elif fmt == "fp6":
        _mx_roundtrip_(w, _E2M3, False)
    elif fmt == "fp6mse":
        _mx_roundtrip_(w, _E2M3, True)
    elif fmt == "fp8":
        _mxfp8_roundtrip_(w)
    else:
        raise ValueError(f"unknown format {fmt}")


def register():
    W = os.environ.get("FQ_W") == "1"
    A = os.environ.get("FQ_A") == "1"
    if not (W or A):
        return
    eb = int(os.environ.get("FQ_EXP_BLOCK", "128"))
    mods = set((os.environ.get("FQ_W_MODS") or "qkv,o,lm").replace("+", ",").split(","))
    lm_fmt = os.environ.get("FQ_LM_FMT", "mxfp4")
    att_fmt = os.environ.get("FQ_ATT_FMT", "mxfp4")
    a_lm = os.environ.get("FQ_A_LM", "1") == "1"
    qkv_parts = set(os.environ.get("FQ_QKV_PARTS", "q+k+v").replace(",", "+").split("+"))
    part_fmt = {"q": os.environ.get("FQ_Q_FMT", att_fmt), "k": os.environ.get("FQ_KV_FMT", att_fmt),
                "v": os.environ.get("FQ_KV_FMT", att_fmt), "o": os.environ.get("FQ_O_FMT", att_fmt)}
    from vllm.model_executor.models import gpt_oss as g

    if getattr(g.GptOssForCausalLM, "_fq_patched", False):
        return
    g.GptOssForCausalLM._fq_patched = True
    orig_load = g.GptOssForCausalLM.load_weights

    def wrap_input(m, block):
        orig = m.forward

        def f(x, *args, **kw):
            return orig(_fp8_roundtrip(x, block), *args, **kw)

        m.forward = f

    def wrap_experts(m, block):
        orig = m.forward

        def f(*args, **kw):
            if "hidden_states" in kw:
                kw["hidden_states"] = _fp8_roundtrip(kw["hidden_states"], block)
            else:
                args = (_fp8_roundtrip(args[0], block),) + tuple(args[1:])
            return orig(*args, **kw)

        m.forward = f

    def load_weights(self, weights):
        import torch

        res = orig_load(self, weights)
        nw = na = ne = 0
        done = []
        for name, m in self.named_modules():
            is_qkv, is_o, is_lm = name.endswith(".qkv_proj"), name.endswith(".o_proj"), name == "lm_head"
            attn = is_qkv or is_o
            sel = (is_qkv and "qkv" in mods) or (is_o and "o" in mods) or (is_lm and "lm" in mods)
            if W and sel:
                w = getattr(m, "weight", None)
                if w is not None and w.dim() == 2 and w.dtype in (torch.bfloat16, torch.float16):
                    with torch.no_grad():
                        if is_qkv:
                            off = 0
                            for part, rows in zip(("q", "k", "v"), m.output_sizes):
                                if part in qkv_parts:
                                    _quantize_(w.data[off:off + rows], part_fmt[part])
                                off += rows
                        else:
                            _quantize_(w.data, lm_fmt if is_lm else part_fmt["o"])
                    nw += 1
                    if is_lm:
                        done.append(f"lm_head as {lm_fmt}")
                    elif not any(d.startswith("q ") for d in done):
                        done.append(f"q {part_fmt['q']} kv {part_fmt['k']} o {part_fmt['o']}")
            if A and attn:
                wrap_input(m, 32)
                na += 1
            if A and name.endswith(".experts"):
                wrap_experts(m, eb)
                ne += 1
        print(f"[FQ] weight round trip: {nw} tensors (mods {','.join(sorted(mods)) if W else '-'}; "
              f"qkv rows {'+'.join(p for p in ('q', 'k', 'v') if p in qkv_parts)}; "
              f"{', '.join(done) or 'lm_head bf16'}); FP8 inputs: {na} attention projections, {ne} expert "
              f"layers (block {eb}), LM head input {'yes' if A and a_lm else 'no'}", flush=True)
        return res

    g.GptOssForCausalLM.load_weights = load_weights
    if A and a_lm:
        orig_cl = g.GptOssForCausalLM.compute_logits

        def compute_logits(self, hidden_states):
            return orig_cl(self, _fp8_roundtrip(hidden_states, 32))

        g.GptOssForCausalLM.compute_logits = compute_logits
