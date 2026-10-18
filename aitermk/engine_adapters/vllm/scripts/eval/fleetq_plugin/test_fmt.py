"""test_fmt.py: the plugin's weight formats on a synthetic weight, CPU only.

Checks that the generic block-32 round trip on the E2M1 grid equals fleet's MXFP4 quantizer bit for bit (expect
"True"), then prints each format's relative error (mxfp4 about 0.12, fp6 and fp8 about 0.03)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleetq_plugin as fq  # noqa: E402

torch.manual_seed(0)
w0 = (torch.randn(4096, 2880) * 0.02).to(torch.bfloat16)
w0[::97, ::31] *= 8  # some outliers, as real weight rows have

a = w0.clone()
fq._mxfp4_roundtrip_(a)
b = w0.clone()
fq._mx_roundtrip_(b, fq._MAG, False)
print("generic grid == fleet mxfp4 bit for bit:", torch.equal(a, b))
ref = w0.float()
for fmt in ("mxfp4", "mse", "fp6", "fp6mse", "fp8"):
    t = w0.clone()
    fq._quantize_(t, fmt)
    rel = ((t.float() - ref).norm() / ref.norm()).item()
    print(f"{fmt:7} relative error {rel:.4f}")
