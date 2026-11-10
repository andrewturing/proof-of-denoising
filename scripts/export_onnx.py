"""Export the SD1.5 UNet and text encoder to ONNX for the in-browser WebGPU backend.

Fixed shapes for the guided step: batch 2 (unconditional + conditional), 4x64x64 latents,
77x768 text embeddings. Writes to runs/onnx/ (gitignored):
  unet_fp32_single.onnx (+ .data)   FP32 UNet, weights in one external-data file (browser)
  text_encoder_fp32.onnx            FP32 text encoder
and checks parity against PyTorch CPU FP32 with onnxruntime. For an FP16 UNet use
scripts/export_unet_fp16.py (note: WebGPU FP16 kernels drift ~100x more than native FP16).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import torch

from podlab.model import Backend, Denoiser
from podlab.sampler import SamplerSpec, initial_noise

OUT = ROOT / "runs" / "onnx"
OUT.mkdir(parents=True, exist_ok=True)


class UNetWrap(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample


class TextWrap(torch.nn.Module):
    def __init__(self, te):
        super().__init__()
        self.te = te

    def forward(self, input_ids):
        return self.te(input_ids).last_hidden_state


def main():
    den = Denoiser(Backend("cpu", "fp32"))
    spec = SamplerSpec()
    z = initial_noise(0, spec)
    embs = np.stack([den.encode(""), den.encode("a red bicycle leaning against a brick wall")])
    sample = torch.from_numpy(np.repeat(z[None], 2, axis=0))
    t = torch.full((2,), 961, dtype=torch.long)
    ehs = torch.from_numpy(embs)

    fp32 = OUT / "unet_fp32.onnx"
    if not fp32.exists():
        t0 = time.time()
        torch.onnx.export(UNetWrap(den.unet).eval(), (sample, t, ehs), str(fp32), input_names=["sample", "timestep", "encoder_hidden_states"],
                          output_names=["eps"], opset_version=17, dynamo=False, external_data=True)
        print("exported fp32 UNet", f"{time.time() - t0:.0f}s", flush=True)
    single = OUT / "unet_fp32_single.onnx"
    if not single.exists():
        m = onnx.load(str(fp32), load_external_data=True)
        onnx.save_model(m, str(single), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=single.name + ".data", size_threshold=1024)
        print("saved single-file fp32 UNet", flush=True)
    te = OUT / "text_encoder_fp32.onnx"
    if not te.exists():
        ids = den.tokenizer("a cat", padding="max_length", max_length=77, return_tensors="pt").input_ids
        torch.onnx.export(TextWrap(den.text_encoder).eval(), (ids,), str(te), input_names=["input_ids"],
                          output_names=["last_hidden_state"], opset_version=17, dynamo=False)
        print("exported text encoder", flush=True)

    import onnxruntime as ort
    with torch.no_grad():
        ref = UNetWrap(den.unet)(sample, t, ehs).numpy()
    for name in ("unet_fp32.onnx", "unet_fp32_single.onnx"):
        sess = ort.InferenceSession(str(OUT / name), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"sample": sample.numpy(), "timestep": t.numpy(), "encoder_hidden_states": embs})[0]
        rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        print(f"parity {name}: rel L2 vs torch cpu fp32 = {rel:.2e}", flush=True)


if __name__ == "__main__":
    main()
