"""TensorRT backend (NVIDIA only, best effort): the SD1.5 UNet compiled with Torch-TensorRT.

TensorRT fuses and re-orders kernels and picks its own tactics, so it is a distinct numerics
stack from PyTorch-CUDA on the same GPU. PoD must tolerate it inside the job's precision
class (or the class must exclude it). Backend name: "trt-fp16" (or "trt-fp32").

Requires: pip install torch-tensorrt (matching the installed torch build).
"""
from __future__ import annotations

import numpy as np
import torch

from .model import Backend, Denoiser

DT = {"fp16": torch.float16, "fp32": torch.float32}


class _Wrap(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, ehs):
        return self.unet(sample, timestep, encoder_hidden_states=ehs, return_dict=False)[0]


class TRTDenoiser(Denoiser):
    """Same interface as Denoiser; the UNet forward runs as a TensorRT engine (batch 2)."""

    def __init__(self, precision: str = "fp16"):
        super().__init__(Backend("cuda", precision))
        import torch_tensorrt  # noqa: F401  (import error -> caller skips this backend)

        dt = DT[precision]
        self.name = f"trt-{precision}"
        ex = (torch.randn(2, 4, 64, 64, device="cuda", dtype=dt),
              torch.full((2,), 961.0, device="cuda", dtype=torch.float32),
              torch.randn(2, 77, 768, device="cuda", dtype=dt))
        self._trt = torch_tensorrt.compile(_Wrap(self.unet).eval(), ir="dynamo", inputs=list(ex),
                                           enabled_precisions={dt}, truncate_double=True,
                                           min_block_size=1)
        self._dt = dt

    @torch.no_grad()
    def unet_eps(self, z: np.ndarray, t: int, embs: np.ndarray) -> np.ndarray:
        b = embs.shape[0]
        e = embs if b == 2 else np.repeat(embs[:1], 2, axis=0)
        zt = torch.from_numpy(np.repeat(z[None], 2, axis=0)).to("cuda", self._dt)
        tt = torch.full((2,), float(t), device="cuda", dtype=torch.float32)
        ht = torch.from_numpy(e).to("cuda", self._dt)
        return self._trt(zt, tt, ht).float().cpu().numpy()[:b]


def make(name: str) -> Denoiser:
    """'trt-fp16' / 'trt-fp32' -> TRTDenoiser; anything else -> regular Denoiser."""
    if name.startswith("trt-"):
        return TRTDenoiser(name.split("-", 1)[1])
    return Denoiser(Backend.parse(name))
