"""SDXL on the SD1.5 path: the same Markov DDIM sampler (sampler.py), workers (workers.py) and verifier
(verify.py), with a 2.6B-parameter UNet at 1024 x 1024 and two text encoders.

SDXL's scheduler uses the same noise schedule as SD1.5 (scaled-linear betas 0.00085-0.012, 1000 train steps,
steps_offset 1, epsilon prediction), so sampler.py applies unchanged with SamplerSpec(height=1024, width=1024).

A conditioning is one float32 array of shape (78, 2048): rows 0-76 are the per-token embeddings (CLIP-L and
OpenCLIP-bigG penultimate hidden states, concatenated), and row 77 carries the pooled bigG embedding in its first
1280 entries. Packing it this way keeps every worker in workers.py (which stacks and slices conditionings as plain
arrays) valid for SDXL. The size conditioning ("time ids") is fixed by the job: original and target size equal to
the render size, no crop.

Substitutes for the smaller-model deviation are distilled SDXL UNets that reuse SDXL's text encoders and VAE:
SSD-1B (1.3B parameters) and Segmind-Vega (0.74B).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .model import DTYPES, Backend, fake_quantize_, set_tf32

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
UNET_REPOS = {"sdxl": BASE, "ssd1b": "segmind/SSD-1B", "vega": "segmind/Segmind-Vega"}
LATENT_SCALE = 0.13025
TOKENS, WIDTH, POOLED = 77, 2048, 1280


class SDXLDenoiser:
    """Two text encoders + UNet (+ optional VAE) on one backend."""

    def __init__(self, backend: Backend, unet: str = "sdxl", load_vae: bool = False, quant_bits: int | None = None,
                 height: int = 1024, width: int = 1024):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        self.backend, self.unet_name = backend, unet
        dev, dt = backend.device, DTYPES[backend.dtype]
        self._tf32 = backend.dtype == "tf32"
        if dev == "cuda" and os.environ.get("POD_DETERMINISTIC") == "1":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        if dev == "cuda":
            set_tf32(self._tf32)
        kw = dict(torch_dtype=dt, use_safetensors=True)
        self.tok1 = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
        self.tok2 = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer_2")
        self.te1 = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", **kw).to(dev).eval()
        self.te2 = CLIPTextModelWithProjection.from_pretrained(BASE, subfolder="text_encoder_2", **kw).to(dev).eval()
        self.unet = UNet2DConditionModel.from_pretrained(UNET_REPOS[unet], subfolder="unet", **kw).to(dev).eval()
        if quant_bits:
            fake_quantize_(self.unet, quant_bits)
        self.vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32, use_safetensors=True).to(dev).eval() if load_vae else None
        self._dev, self._dt = dev, dt
        self._time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=dt, device=dev)

    @property
    def name(self) -> str:
        return self.backend.name

    @torch.no_grad()
    def encode(self, prompt: str) -> np.ndarray:
        """(78, 2048) float32: token embeddings, then the pooled embedding in the last row."""
        hs = []
        pooled = None
        for tok, te in ((self.tok1, self.te1), (self.tok2, self.te2)):
            ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(self._dev)
            out = te(ids, output_hidden_states=True)
            hs.append(out.hidden_states[-2][0])
            if te is self.te2:
                pooled = out[0][0]  # projected pooled embedding of the second encoder
        emb = torch.cat(hs, dim=-1).float().cpu().numpy()
        row = np.zeros((1, WIDTH), np.float32)
        row[0, :POOLED] = pooled.float().cpu().numpy()
        return np.concatenate([emb, row], axis=0)

    def free_text_encoder(self) -> None:
        self.te1 = self.te2 = None
        if self._dev == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def unet_eps(self, z: np.ndarray, t: int, embs: np.ndarray) -> np.ndarray:
        """Raw UNet noise predictions for a batch of conditionings. Returns float32 (B, 4, h, w)."""
        b = embs.shape[0]
        if self._dev == "cuda":
            set_tf32(self._tf32)  # flags are process-global; re-assert per call
        zt = torch.from_numpy(np.repeat(z[None], b, axis=0)).to(self._dev, self._dt)
        tt = torch.full((b,), t, device=self._dev, dtype=torch.long)
        ht = torch.from_numpy(np.ascontiguousarray(embs[:, :TOKENS])).to(self._dev, self._dt)
        pooled = torch.from_numpy(np.ascontiguousarray(embs[:, TOKENS, :POOLED])).to(self._dev, self._dt)
        added = {"text_embeds": pooled, "time_ids": self._time_ids.expand(b, -1)}
        out = self.unet(zt, tt, encoder_hidden_states=ht, added_cond_kwargs=added).sample
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
        zt = torch.from_numpy(z0[None] / LATENT_SCALE).to(self._dev, torch.float32)
        img = self.vae.decode(zt).sample[0].float().cpu().numpy()
        return ((np.clip(img, -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)

    def sync(self) -> None:
        if self._dev == "mps":
            torch.mps.synchronize()
        elif self._dev == "cuda":
            torch.cuda.synchronize()
