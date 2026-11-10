"""Export an FP16 SD1.5 UNet to ONNX directly from PyTorch (model in half precision).

diffusers casts the timestep embedding to the sample dtype itself, so the graph has
consistent types. Inputs and outputs stay float32 (cast at the boundary). Writes
runs/onnx/unet_fp16_torch.onnx and checks it against the FP32 ONNX export.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch

from podlab.model import Backend, Denoiser

OUT = ROOT / "runs" / "onnx"


class W(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample.half(), timestep, encoder_hidden_states=encoder_hidden_states.half()).sample.float()


def main():
    den = Denoiser(Backend("cpu", "fp16"))
    rng = np.random.default_rng(0)
    sample = torch.from_numpy(rng.standard_normal((2, 4, 64, 64)).astype(np.float32))
    t = torch.full((2,), 961, dtype=torch.long)
    ehs = torch.from_numpy((rng.standard_normal((2, 77, 768)) * 0.5).astype(np.float32))
    path = OUT / "unet_fp16_torch.onnx"
    t0 = time.time()
    torch.onnx.export(W(den.unet).eval(), (sample, t, ehs), str(path), input_names=["sample", "timestep", "encoder_hidden_states"],
                      output_names=["eps"], opset_version=17, dynamo=False)
    print("exported", path.name, round(path.stat().st_size / 1e9, 2), "GB", f"{time.time() - t0:.0f}s", flush=True)
    feeds = {"sample": sample.numpy(), "timestep": t.numpy(), "encoder_hidden_states": ehs.numpy()}
    a = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    b = ort.InferenceSession(str(OUT / "unet_fp32.onnx"), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    print("fp16 vs fp32 onnx rel L2:", float(np.linalg.norm(a - b) / np.linalg.norm(b)), "finite:", bool(np.isfinite(a).all()), flush=True)


if __name__ == "__main__":
    main()
