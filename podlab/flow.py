"""Proof-of-Denoising for flow-matching transformers (SANA: linear DiT, rectified flow; Wan 2.1: video DiT).

Same protocol as the SD1.5 path (sampler.py + model.py), second model family:
  * pinned first-order Euler sampler over shifted flow sigmas, sigma_0 = 1 (noise) -> sigma_N = 0;
    z_{i+1} = z_i + (sigma_{i+1} - sigma_i) * v(z_i, sigma_i), arithmetic in float64/float32 numpy.
    SANA's default DPM-Solver++(2M) carries the previous model output between steps, so the latent
    alone is not a Markov state; PoD pins Euler (or would commit the solver history with the latent).
  * Philox initial noise keyed by (seed, 0), as in sampler.py.
  * The network evaluation (Gemma-2 text encoder + SANA transformer) runs on the worker's backend.

Cheats mirror workers.py: smaller-model substitution (job pins SANA-1.6B, worker runs SANA-0.6B with
the same text encoder and autoencoder), W8/W4 fake-quantized weights, CFG skip, velocity caching
(all / late), step skipping, safety-prompt stripping.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .model import DTYPES, Backend, fake_quantize_, set_tf32
from .sampler import _normal

SANA_REPOS = {"sana1600": "Efficient-Large-Model/Sana_1600M_512px_diffusers",
              "sana600": "Efficient-Large-Model/Sana_600M_512px_diffusers"}
DTYPES = {**DTYPES, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class FlowSpec:
    steps: int = 25
    guidance: float = 4.5
    shift: float = 3.0
    height: int = 512
    width: int = 512
    channels: int = 32
    down: int = 32  # DC-AE f32c32
    frames: int = 0  # latent frames of a video model; 0 for an image model

    @property
    def shape(self) -> tuple[int, ...]:
        hw = (self.height // self.down, self.width // self.down)
        return (self.channels, self.frames, *hw) if self.frames else (self.channels, *hw)

    def sigmas(self) -> np.ndarray:
        t = 1.0 - np.arange(self.steps + 1, dtype=np.float64) / self.steps
        return self.shift * t / (1.0 + (self.shift - 1.0) * t)

    def to_dict(self) -> dict:
        return asdict(self)


def initial_noise(seed: int, spec: FlowSpec) -> np.ndarray:
    return _normal([seed, 0], spec.shape)


def euler_update(z: np.ndarray, v: np.ndarray, i: int, spec: FlowSpec, span: int = 1) -> np.ndarray:
    s = spec.sigmas()
    return (z.astype(np.float64) + (s[i + span] - s[i]) * v.astype(np.float64)).astype(np.float32)


def step(v_fn, z: np.ndarray, i: int, spec: FlowSpec) -> np.ndarray:
    return euler_update(z, v_fn(z, float(spec.sigmas()[i])), i, spec)


def sample(v_fn, seed: int, spec: FlowSpec) -> np.ndarray:
    states = [initial_noise(seed, spec)]
    for i in range(spec.steps):
        states.append(step(v_fn, states[-1], i, spec))
    return np.stack(states)


class SanaDenoiser:
    """Gemma-2 text encoder + SANA transformer (+ optional DC-AE decoder) on one backend.
    Conditionings are dicts {"emb": (L, 2304) float32, "mask": (L,) int64}."""

    def __init__(self, backend: Backend, model: str = "sana1600", quant_bits: int | None = None, load_vae: bool = False):
        from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
        self.backend, self.model = backend, model
        dev, dt = backend.device, DTYPES[backend.dtype]
        self._tf32 = backend.dtype == "tf32"
        if dev == "cuda" and os.environ.get("POD_DETERMINISTIC") == "1":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        if dev == "cuda":
            set_tf32(self._tf32)
        repo = SANA_REPOS[model]
        self.pipe = SanaPipeline.from_pretrained(repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
        self.chi = inspect.signature(SanaPipeline.__call__).parameters["complex_human_instruction"].default
        self.transformer = SanaTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dt).to(dev).eval()
        if quant_bits:
            fake_quantize_(self.transformer, quant_bits)
        self.vae = AutoencoderDC.from_pretrained(repo, subfolder="vae", torch_dtype=dt).to(dev).eval() if load_vae else None
        self._dev, self._dt = dev, dt

    @property
    def name(self) -> str:
        return self.backend.name

    @torch.no_grad()
    def encode(self, prompt: str, negative: str = "") -> tuple[dict, dict]:
        """(cond, uncond) for a prompt and the job's negative prompt, encoded on this backend."""
        pe, pm, ne, nm = self.pipe.encode_prompt(prompt, do_classifier_free_guidance=True, negative_prompt=negative,
                                                 device=self._dev, max_sequence_length=300,
                                                 complex_human_instruction=self.chi)
        to = lambda e, m: {"emb": e[0].float().cpu().numpy(), "mask": m[0].long().cpu().numpy()}
        return to(pe, pm), to(ne, nm)

    def free_text_encoder(self) -> None:
        """Encodings are cached by callers; drop the 2B-parameter text encoder to make room."""
        self.pipe.text_encoder = None
        if self._dev == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def velocity(self, z: np.ndarray, sigma: float, conds: list[dict]) -> np.ndarray:
        """Raw transformer velocity predictions for a batch of conditionings. float32 (B, C, h, w)."""
        b = len(conds)
        if self._dev == "cuda":
            set_tf32(self._tf32)
        zt = torch.from_numpy(np.repeat(z[None], b, axis=0)).to(self._dev, self._dt)
        L = max(c["emb"].shape[0] for c in conds)
        emb = torch.from_numpy(np.stack([c["emb"] for c in conds])).to(self._dev, self._dt)
        mask = torch.from_numpy(np.stack([c["mask"] for c in conds])).to(self._dev)
        assert emb.shape[1] == L
        tt = torch.full((b,), sigma * 1000.0, device=self._dev, dtype=self._dt)
        out = self.transformer(zt, encoder_hidden_states=emb, encoder_attention_mask=mask, timestep=tt, return_dict=False)[0]
        return out.float().cpu().numpy()

    def v_fn(self, cond: dict, uncond: dict, guidance: float):
        def f(z: np.ndarray, sigma: float) -> np.ndarray:
            v_u, v_c = self.velocity(z, sigma, [uncond, cond])
            return v_u + guidance * (v_c - v_u)
        return f

    @torch.no_grad()
    def decode(self, z0: np.ndarray) -> np.ndarray:
        assert self.vae is not None, "load_vae=True is required"
        zt = torch.from_numpy(z0[None] / self.vae.config.scaling_factor).to(self._dev, self._dt)
        img = self.vae.decode(zt, return_dict=False)[0][0].float().cpu().numpy()
        return ((np.clip(img, -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)

    def sync(self) -> None:
        if self._dev == "cuda":
            torch.cuda.synchronize()
        elif self._dev == "mps":
            torch.mps.synchronize()


# ---- Wan 2.1 (text-to-video, 1.3B): the same sampler and workers, a 5-D latent (C, T, h, w) ----

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_TEXT_LEN = 512


class WanText:
    """UMT5-XXL text encoder for Wan, in FP32 on the CPU. Its 5.7B parameters do not fit beside the transformer on
    a 24 GB GPU, so the job pins this encoder and callers cache its embeddings. An embedding is (512, 4096)
    float32, zero after the prompt's last token, as in the reference pipeline."""

    def __init__(self):
        from transformers import AutoTokenizer, UMT5EncoderModel
        self.tok = AutoTokenizer.from_pretrained(WAN_REPO, subfolder="tokenizer")
        self.enc = UMT5EncoderModel.from_pretrained(WAN_REPO, subfolder="text_encoder", torch_dtype=torch.float32).eval()

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        text = " ".join(text.split())  # the job's prompts are plain text: whitespace-normalised, nothing else
        t = self.tok([text], padding="max_length", max_length=WAN_TEXT_LEN, truncation=True, add_special_tokens=True,
                     return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.gt(0).sum())
        h = self.enc(t.input_ids, t.attention_mask).last_hidden_state[0].float().numpy()
        out = np.zeros((WAN_TEXT_LEN, h.shape[1]), np.float32)
        out[:n] = h[:n]
        return out


class WanDenoiser:
    """Wan 2.1 video transformer (+ optional VAE) on one backend. Conditionings are (512, 4096) float32 arrays
    from WanText; this class holds no text encoder."""

    def __init__(self, backend: Backend, model: str = "wan13", quant_bits: int | None = None, load_vae: bool = False):
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        self.backend, self.model = backend, model
        dev, dt = backend.device, DTYPES[backend.dtype]
        self._tf32 = backend.dtype == "tf32"
        if dev == "cuda" and os.environ.get("POD_DETERMINISTIC") == "1":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        if dev == "cuda":
            set_tf32(self._tf32)
        self.transformer = WanTransformer3DModel.from_pretrained(WAN_REPO, subfolder="transformer", torch_dtype=dt).to(dev).eval()
        if quant_bits:
            fake_quantize_(self.transformer, quant_bits)
        self.vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.float32).to(dev).eval() if load_vae else None
        self._dev, self._dt = dev, dt

    @property
    def name(self) -> str:
        return self.backend.name

    def free_text_encoder(self) -> None:
        pass

    @torch.no_grad()
    def velocity(self, z: np.ndarray, sigma: float, conds: list) -> np.ndarray:
        """Raw transformer velocity predictions for a batch of conditionings. float32 (B, C, T, h, w)."""
        b = len(conds)
        if self._dev == "cuda":
            set_tf32(self._tf32)
        zt = torch.from_numpy(np.repeat(z[None], b, axis=0)).to(self._dev, self._dt)
        emb = torch.from_numpy(np.stack(conds)).to(self._dev, self._dt)
        tt = torch.full((b,), sigma * 1000.0, device=self._dev, dtype=torch.float32)
        out = self.transformer(hidden_states=zt, timestep=tt, encoder_hidden_states=emb, return_dict=False)[0]
        return out.float().cpu().numpy()

    def v_fn(self, cond, uncond, guidance: float):
        def f(z: np.ndarray, sigma: float) -> np.ndarray:
            v_u, v_c = self.velocity(z, sigma, [uncond, cond])
            return v_u + guidance * (v_c - v_u)
        return f

    @torch.no_grad()
    def decode(self, z0: np.ndarray) -> np.ndarray:
        """Final latent -> video frames, uint8 (F, H, W, 3)."""
        assert self.vae is not None, "load_vae=True is required"
        c = self.vae.config
        mean = torch.tensor(c.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(c.latents_std).view(1, -1, 1, 1, 1)
        lat = (torch.from_numpy(z0[None]).float() * std + mean).to(self._dev)
        vid = self.vae.decode(lat, return_dict=False)[0][0].float().cpu().numpy()  # (3, F, H, W)
        return ((np.clip(vid, -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 3, 0)

    def sync(self) -> None:
        if self._dev == "cuda":
            torch.cuda.synchronize()
        elif self._dev == "mps":
            torch.mps.synchronize()


# ---- workers: every worker returns the full committed trajectory (steps + 1 latents) ----

def honest(den, cond, uncond, seed, spec):
    return sample(den.v_fn(cond, uncond, spec.guidance), seed, spec)


def cfg_skip(den, cond, uncond, seed, spec):
    return sample(lambda z, s: den.velocity(z, s, [cond])[0], seed, spec)


def v_cache(den, cond, uncond, seed, spec, period: int = 2, start: int = 0):
    f = den.v_fn(cond, uncond, spec.guidance)
    sig = spec.sigmas()
    states, last = [initial_noise(seed, spec)], None
    for i in range(spec.steps):
        if last is None or i < start or (i - start) % period == 0:
            last = f(states[-1], float(sig[i]))
        states.append(euler_update(states[-1], last, i, spec))
    return np.stack(states)


def step_skip(den, cond, uncond, seed, spec):
    f = den.v_fn(cond, uncond, spec.guidance)
    sig = spec.sigmas()
    states, i = [initial_noise(seed, spec)], 0
    while i < spec.steps:
        z = states[-1]
        v = f(z, float(sig[i]))
        if i + 1 < spec.steps:
            z2 = euler_update(z, v, i, spec, span=2)
            states += [((z + z2) / 2).astype(np.float32), z2]
            i += 2
        else:
            states.append(euler_update(z, v, i, spec))
            i += 1
    return np.stack(states)


def neg_strip(den, cond, uncond_empty, seed, spec):
    return sample(den.v_fn(cond, uncond_empty, spec.guidance), seed, spec)


def check_steps(den, cond, uncond, states, idx, spec):
    """Same statistic as verify.check_steps: r_i = ||Step_V(z_i) - z_{i+1}|| / ||Step_V(z_i) - z_i||."""
    f = den.v_fn(cond, uncond, spec.guidance)
    out = []
    for i in idx:
        z_hat = step(f, states[i], i, spec)
        d = z_hat.astype(np.float64) - states[i + 1]
        s = z_hat.astype(np.float64) - states[i]
        diff, sn = float(np.linalg.norm(d)), float(np.linalg.norm(s))
        out.append({"i": int(i), "diff": diff, "step_norm": sn, "ratio": diff / max(sn, 1e-12), "maxabs": float(np.abs(d).max())})
    return out
