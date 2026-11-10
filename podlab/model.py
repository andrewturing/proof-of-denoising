"""SD1.5-class denoiser on a chosen on-device backend (Apple GPU via MPS, CUDA, or CPU).

Weights load from the local Hugging Face cache only (HF_HUB_OFFLINE), and only from
safetensors files; fetch them once with scripts/fetch_models.py. Inference makes no network
calls. Set POD_BASE_REPO to use a different SD1.5 mirror.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer

from .sampler import LATENT_SCALE

BASE_CANDIDATES = ("stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5")


def _resolve_base() -> str:
    """POD_BASE_REPO if set; otherwise the first SD1.5 mirror present in the local cache."""
    if os.environ.get("POD_BASE_REPO"):
        return os.environ["POD_BASE_REPO"]
    for repo in BASE_CANDIDATES:
        try:
            hf_hub_download(repo, "unet/config.json", local_files_only=True)
            return repo
        except Exception:
            continue
    return BASE_CANDIDATES[0]


BASE_REPO = _resolve_base()
UNET_REPOS = {
    "sd15": BASE_REPO,
    "bk-sdm-small": "nota-ai/bk-sdm-small",
    "bk-sdm-tiny": "nota-ai/bk-sdm-tiny",
}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}


def set_tf32(enabled: bool) -> None:
    """CUDA only: TF32 is ON by default for cuDNN convolutions on Ampere+, which silently moves an
    "fp32" worker out of the FP32 precision class. PoD makes it explicit: cuda-fp32 = true FP32,
    cuda-tf32 = TF32 matmuls and convolutions."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


@dataclass(frozen=True)
class Backend:
    device: str  # "cpu", "mps" or "cuda"
    dtype: str  # "fp32", "fp16", "bf16" or "tf32" (CUDA)

    @property
    def name(self) -> str:
        return f"{self.device}-{self.dtype}"

    @classmethod
    def parse(cls, name: str) -> "Backend":
        device, dtype = name.split("-")
        return cls(device, dtype)


def local_dir(repo: str) -> Path:
    """Snapshot directory of a cached repo. Works for partial snapshots (e.g. UNet-only
    downloads), which snapshot_download(local_files_only=True) rejects as incomplete."""
    return Path(hf_hub_download(repo, "unet/config.json", local_files_only=True)).parent.parent


def weights_hash(repo: str, sub: str = "unet") -> str:
    """SHA-256 of the safetensors file, used in the job header (cached next to nothing; recomputed)."""
    path = local_dir(repo) / sub / "diffusion_pytorch_model.safetensors"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fake_quantize_(module: torch.nn.Module, bits: int) -> None:
    """Symmetric per-output-channel weight quantization, applied in place (a cheating worker's
    precision downgrade). Only Conv2d and Linear weights are touched."""
    qmax = 2 ** (bits - 1) - 1
    for m in module.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            w = m.weight.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().amax(dim=1).clamp(min=1e-12) / qmax
            q = torch.round(flat / scale[:, None]).clamp(-qmax, qmax)
            m.weight.data = (q * scale[:, None]).reshape(w.shape).to(w.dtype)


class Denoiser:
    """Text encoder + UNet (+ optional VAE) on one backend."""

    def __init__(self, backend: Backend, unet: str = "sd15", load_vae: bool = False, quant_bits: int | None = None):
        self.backend = backend
        self.unet_name = unet
        dev, dt = backend.device, DTYPES[backend.dtype]
        self._tf32 = backend.dtype == "tf32"
        if dev == "cuda" and os.environ.get("POD_DETERMINISTIC") == "1":
            # Same-GPU reruns become bit-identical; drift then appears only across GPUs/precisions/stacks.
            # Needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment (set by gtc_kit/run_nvidia.sh).
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        if dev == "cuda":
            set_tf32(self._tf32)
        base = local_dir(BASE_REPO)
        self.tokenizer = CLIPTokenizer.from_pretrained(base / "tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(base / "text_encoder", torch_dtype=dt, use_safetensors=True).to(dev).eval()
        self.unet = UNet2DConditionModel.from_pretrained(local_dir(UNET_REPOS[unet]) / "unet", torch_dtype=dt, use_safetensors=True).to(dev).eval()
        if quant_bits:
            fake_quantize_(self.unet, quant_bits)
        self.vae = AutoencoderKL.from_pretrained(base / "vae", torch_dtype=dt, use_safetensors=True).to(dev).eval() if load_vae else None
        self._dev, self._dt = dev, dt

    @torch.no_grad()
    def encode(self, prompt: str) -> np.ndarray:
        ids = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self._dev)
        return self.text_encoder(ids).last_hidden_state[0].float().cpu().numpy()

    @torch.no_grad()
    def unet_eps(self, z: np.ndarray, t: int, embs: np.ndarray) -> np.ndarray:
        """Raw UNet noise predictions for a batch of conditionings. Returns float32 (B, 4, h, w)."""
        b = embs.shape[0]
        if self._dev == "cuda":
            set_tf32(self._tf32)  # flags are process-global; re-assert per call
        zt = torch.from_numpy(np.repeat(z[None], b, axis=0)).to(self._dev, self._dt)
        tt = torch.full((b,), t, device=self._dev, dtype=torch.long)
        ht = torch.from_numpy(embs).to(self._dev, self._dt)
        out = self.unet(zt, tt, encoder_hidden_states=ht).sample
        return out.float().cpu().numpy()

    def eps_fn(self, cond: np.ndarray, uncond: np.ndarray, guidance: float):
        """Guided noise prediction; the CFG combination runs in float32 on the CPU."""
        embs = np.stack([uncond, cond])

        def f(z: np.ndarray, t: int) -> np.ndarray:
            e_u, e_c = self.unet_eps(z, t, embs)
            return e_u + guidance * (e_c - e_u)

        return f

    @torch.no_grad()
    def decode(self, z0: np.ndarray) -> np.ndarray:
        assert self.vae is not None, "load_vae=True is required"
        zt = torch.from_numpy(z0[None] / LATENT_SCALE).to(self._dev, self._dt)
        img = self.vae.decode(zt).sample[0].float().cpu().numpy()
        return ((np.clip(img, -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)

    def sync(self) -> None:
        if self._dev == "mps":
            torch.mps.synchronize()
        elif self._dev == "cuda":
            torch.cuda.synchronize()
