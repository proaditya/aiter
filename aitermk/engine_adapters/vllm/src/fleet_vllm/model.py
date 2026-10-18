"""GPT-OSS subclass: stock prefill, Fleet megakernel decode."""

from .mixin import FleetMKModelMixin


def _make_gptoss_class():
    from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

    class FleetMKGptOssForCausalLM(FleetMKModelMixin, GptOssForCausalLM):
        """MoE GPT-OSS 120B: 128 experts top-4, attention sinks, per-layer SW."""

        FLEET_MK_ARCH = "GptOssForCausalLM"

        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._fleet_mk_setup_state(vllm_config)

        def _fleet_mk_layer_attn(self, li):
            return self.model.layers[li].attn.attn

        def _fleet_mk_pack_weights(self, packers, spec, dev):
            from .moe_layout import MoeLayout
            from .packing_moe import pack_lm_head_moe, pack_moe_layer

            cos, sin = self._fleet_mk_rope_from_vllm(spec, dev)
            layout = MoeLayout(spec)
            print(layout.describe(), flush=True)

            weight_ptrs_host, refs = [], []
            scale_refs, moe_scale_ptrs = [], []
            for li in range(spec.num_layers):
                vlayer = self.model.layers[li]
                vattn = vlayer.attn
                w_q, w_k, w_v, q_b, k_b, v_b = self._fleet_mk_split_qkv(vattn, spec)
                packed, scales = pack_moe_layer(
                    w_q=w_q,
                    w_k=w_k,
                    w_v=w_v,
                    q_bias=q_b,
                    k_bias=k_b,
                    v_bias=v_b,
                    w_o=vattn.o_proj.weight.data,
                    o_bias=None if vattn.o_proj.bias is None else vattn.o_proj.bias.data,
                    norm_pre=vlayer.input_layernorm.weight.data,
                    norm_post=vlayer.post_attention_layernorm.weight.data,
                    router_w=vlayer.mlp.router.weight.data,
                    router_b=vlayer.mlp.router.bias.data,
                    sinks=vattn.sinks.data,
                    spec=spec,
                    packers=packers,
                    layout=layout,
                    **self._fleet_mk_experts_from_vllm(li, spec),
                    **self._fleet_mk_foreign_experts(li, layout),
                )
                refs.extend(packed)
                weight_ptrs_host.extend(t.data_ptr() for t in packed)
                scale_refs.extend(t for t in scales if t is not None)
                moe_scale_ptrs.append(
                    None if scales[0] is None else (scales[0].data_ptr(), scales[1].data_ptr())
                )

            final_norm_w = packers["pad_weight_1d"](
                self.model.norm.weight.data, spec.padded_hidden_size, pad_value=0.0
            ).contiguous()
            lm_head_packed, lm_head_bias = pack_lm_head_moe(self.lm_head.weight.data, spec, packers)

            return dict(
                weight_ptrs_host=weight_ptrs_host,
                weight_refs=refs,
                moe_scale_refs=scale_refs,
                moe_scale_ptrs=moe_scale_ptrs if moe_scale_ptrs[0] else None,
                final_norm_w=final_norm_w,
                lm_head_packed=lm_head_packed,
                lm_head_bias=lm_head_bias,
                cos=cos,
                sin=sin,
            )

        def _fleet_mk_foreign_experts(self, li, layout):
            if not layout.alias:
                return {}
            exp = self.model.layers[li].mlp.experts.routed_experts
            return dict(foreign_w13=exp.w13_weight.data, foreign_w2=exp.w2_weight.data)

        def _fleet_mk_experts_from_vllm(self, li, spec):
            exp = self.model.layers[li].mlp.experts.routed_experts
            inter, hidden = spec.intermediate_size, spec.hidden_size
            nb_h, nb_i = hidden // 32, inter // 32

            def blocks(t, rows, nblk):
                E, _, nbytes = t.shape
                return t.view(E, -1, nbytes // 16, 16)[:, :rows, :nblk]

            return dict(
                gate_up_blocks=blocks(exp.w13_weight.data, 2 * inter, nb_h),
                gate_up_scales=exp.w13_weight_scale.data[:, : 2 * inter, :nb_h],
                gate_up_bias=exp.w13_bias.data[:, : 2 * inter],
                down_blocks=blocks(exp.w2_weight.data, hidden, nb_i),
                down_scales=exp.w2_weight_scale.data[:, :hidden, :nb_i],
                down_bias=exp.w2_bias.data[:, :hidden],
            )

        def _fleet_mk_split_qkv(self, vattn, spec):
            q_end = spec.num_q_heads * spec.head_dim
            k_end = q_end + spec.num_kv_heads * spec.head_dim
            v_end = k_end + spec.num_kv_heads * spec.head_dim
            w = vattn.qkv_proj.weight.data
            assert w.shape[0] == v_end == spec.qkv_output_size, (
                tuple(w.shape),
                v_end,
                spec.qkv_output_size,
            )
            b = None if vattn.qkv_proj.bias is None else vattn.qkv_proj.bias.data
            return (
                w[:q_end],
                w[q_end:k_end],
                w[k_end:v_end],
                None if b is None else b[:q_end],
                None if b is None else b[q_end:k_end],
                None if b is None else b[k_end:v_end],
            )

        def _fleet_mk_rope_from_vllm(self, spec, dev):
            import torch

            rope = self.model.layers[0].attn.rotary_emb
            cache = rope.cos_sin_cache
            rows = self._max_model_len + 1
            assert cache.shape[0] >= rows, (cache.shape, rows)
            cos_raw, sin_raw = cache[:rows].chunk(2, dim=-1)
            pad = spec.head_dim - cos_raw.shape[-1]
            cos = (
                torch.nn.functional.pad(cos_raw, (0, pad))
                .to(device=dev, dtype=torch.bfloat16)
                .contiguous()
            )
            sin = (
                torch.nn.functional.pad(sin_raw, (0, pad))
                .to(device=dev, dtype=torch.bfloat16)
                .contiguous()
            )
            return cos, sin

    return FleetMKGptOssForCausalLM
