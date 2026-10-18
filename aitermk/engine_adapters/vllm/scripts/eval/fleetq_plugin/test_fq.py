"""test_fq.py DEMO_PY: the plugin's quantizers against fleet's own, CPU only.

DEMO_PY is the fleet checkout's demo/gpt_oss/demo.py; its quantize_bf16_to_mxfp4, dequant_mxfp4_to_bf16 and
fp8_act_roundtrip are extracted and compared with the plugin's on random tensors (expect "identical True")."""
import ast
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleetq_plugin as fq  # noqa: E402

src = open(sys.argv[1]).read()
tree = ast.parse(src)
want = {"fp8_act_roundtrip", "quantize_bf16_to_mxfp4", "dequant_mxfp4_to_bf16"}
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want)
         or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_FP4_MAGNITUDES" for t in n.targets))]
ns = {"torch": torch, "print": lambda *a, **k: None}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "demo_extract", "exec"), ns)
print("extracted:", sorted(k for k in ns if k in want or k == "_FP4_MAGNITUDES"))
torch.manual_seed(0)
for shape in ((640, 2880), (2880, 4096), (512, 2880)):
    w = (torch.randn(shape) * 0.02).to(torch.bfloat16)
    w[3, :32] = 0
    b, s = ns["quantize_bf16_to_mxfp4"](w)
    ref = ns["dequant_mxfp4_to_bf16"](b, s)[0].to(torch.bfloat16).reshape(shape)
    mine = w.clone()
    fq._mxfp4_roundtrip_(mine)
    print(f"mxfp4 {shape}: identical {torch.equal(ref, mine)}, max diff {(ref.float() - mine.float()).abs().max():.3g}, "
          f"rel err vs bf16 {((mine.float() - w.float()).norm() / w.float().norm()):.3f}")
for shape in ((7, 2880), (3, 4096)):
    x = (torch.randn(shape) * 3).to(torch.bfloat16)
    ref = ns["fp8_act_roundtrip"](x)
    mine = fq._fp8_roundtrip(x, 32)
    print(f"fp8 {shape}: identical {torch.equal(ref, mine)}, rel err vs bf16 {((mine.float() - x.float()).norm() / x.float().norm()):.4f}")

