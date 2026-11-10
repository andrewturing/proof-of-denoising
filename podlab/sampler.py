"""Explicit, Markov DDIM sampler for SD1.5-class latent diffusion.

Proof-of-Denoising needs every step to be a pure function of the committed state, so the
sampler is written out here instead of using a diffusers scheduler object (which keeps
hidden counters). The DDIM update uses float64 numpy on the CPU and rounds each stored
latent to float32. Guidance combination uses float32; network forward passes run on the
worker's backend. This fixed arithmetic confines backend-dependent drift to the model
evaluations (including text encoding).

Noise comes from numpy's counter-based Philox generator keyed by (seed, step), so any
verifier on any machine reproduces it bit-exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

# SD1.5 scheduler config: scaled_linear betas, 1000 train steps, steps_offset=1,
# set_alpha_to_one=False, clip_sample=False, epsilon prediction.
TRAIN_STEPS = 1000
BETA_START, BETA_END = 0.00085, 0.012
STEPS_OFFSET = 1
LATENT_SCALE = 0.18215

EpsFn = Callable[[np.ndarray, int], np.ndarray]


def alphas_cumprod() -> np.ndarray:
    betas = np.linspace(BETA_START**0.5, BETA_END**0.5, TRAIN_STEPS, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


_AC = alphas_cumprod()


@dataclass(frozen=True)
class SamplerSpec:
    steps: int = 25
    guidance: float = 7.5
    eta: float = 0.0  # 0: deterministic DDIM (ODE); 1: DDPM-like stochastic sampler (SDE)
    height: int = 512
    width: int = 512

    @property
    def shape(self) -> tuple[int, int, int]:
        return (4, self.height // 8, self.width // 8)

    def timesteps(self) -> np.ndarray:
        ratio = TRAIN_STEPS // self.steps
        return (np.arange(self.steps) * ratio)[::-1].astype(np.int64) + STEPS_OFFSET

    def to_dict(self) -> dict:
        return asdict(self)


def _normal(key: list[int], shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.Generator(np.random.Philox(key=key))
    return rng.standard_normal(shape, dtype=np.float64).astype(np.float32)


def initial_noise(seed: int, spec: SamplerSpec) -> np.ndarray:
    return _normal([seed, 0], spec.shape)


def step_noise(seed: int, i: int, spec: SamplerSpec) -> np.ndarray:
    return _normal([seed, i + 1], spec.shape)


def ddim_coeffs(i: int, spec: SamplerSpec) -> tuple[int, float, float, float]:
    """Return (t, alpha_t, alpha_prev, sigma) for step index i (0 = noisiest)."""
    t = int(spec.timesteps()[i])
    prev = t - TRAIN_STEPS // spec.steps
    a_t = float(_AC[t])
    a_prev = float(_AC[prev]) if prev >= 0 else float(_AC[0])
    sigma = spec.eta * np.sqrt((1 - a_prev) / (1 - a_t)) * np.sqrt(1 - a_t / a_prev)
    return t, a_t, a_prev, float(sigma)


def ddim_update(z: np.ndarray, eps: np.ndarray, i: int, spec: SamplerSpec, seed: int) -> np.ndarray:
    """One DDIM update given the (guided) noise prediction. Pure float32/float64 numpy."""
    _, a_t, a_prev, sigma = ddim_coeffs(i, spec)
    z64, e64 = z.astype(np.float64), eps.astype(np.float64)
    x0 = (z64 - np.sqrt(1 - a_t) * e64) / np.sqrt(a_t)
    out = np.sqrt(a_prev) * x0 + np.sqrt(max(1 - a_prev - sigma**2, 0.0)) * e64
    if sigma > 0:
        out = out + sigma * step_noise(seed, i, spec).astype(np.float64)
    return out.astype(np.float32)


def step(eps_fn: EpsFn, z: np.ndarray, i: int, spec: SamplerSpec, seed: int) -> np.ndarray:
    """z_{i+1} = Step(z_i). eps_fn(z, t) returns the guided noise prediction."""
    t = int(spec.timesteps()[i])
    return ddim_update(z, eps_fn(z, t), i, spec, seed)


def sample(eps_fn: EpsFn, seed: int, spec: SamplerSpec) -> np.ndarray:
    """Run the full trajectory. Returns states with shape (steps + 1, 4, H/8, W/8);
    states[0] is the initial noise and states[-1] is the final latent."""
    states = [initial_noise(seed, spec)]
    for i in range(spec.steps):
        states.append(step(eps_fn, states[-1], i, spec, seed))
    return np.stack(states)
