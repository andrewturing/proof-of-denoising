"""Honest and cheating workers. Every worker returns the full list of committed states
(steps + 1 latents), as the protocol requires. Cheats imitate rational shortcuts that save
compute, or that strip a safety component, while still producing a full-length trajectory.
"""
from __future__ import annotations

import numpy as np

from .model import Denoiser
from .sampler import SamplerSpec, ddim_coeffs, ddim_update, initial_noise, sample


def honest(den: Denoiser, cond, uncond, seed: int, spec: SamplerSpec) -> np.ndarray:
    return sample(den.eps_fn(cond, uncond, spec.guidance), seed, spec)


def cfg_skip(den: Denoiser, cond, uncond, seed: int, spec: SamplerSpec) -> np.ndarray:
    """Skip the unconditional pass (half the network evaluations) and report the
    conditional prediction as if it were guided."""
    embs = cond[None]
    return sample(lambda z, t: den.unet_eps(z, t, embs)[0], seed, spec)


def eps_cache(den: Denoiser, cond, uncond, seed: int, spec: SamplerSpec, period: int = 2, start: int = 0) -> np.ndarray:
    """Feature-caching shortcut: from step `start` on, evaluate the network only every
    `period` steps and reuse the last guided prediction in between."""
    f = den.eps_fn(cond, uncond, spec.guidance)
    states = [initial_noise(seed, spec)]
    last = None
    for i in range(spec.steps):
        t = int(spec.timesteps()[i])
        if last is None or i < start or (i - start) % period == 0:
            last = f(states[-1], t)
        states.append(ddim_update(states[-1], last, i, spec, seed))
    return np.stack(states)


def step_skip(den: Denoiser, cond, uncond, seed: int, spec: SamplerSpec) -> np.ndarray:
    """Evaluate the network on even steps only, jump two steps with one DDIM update, and
    fill the odd states by linear interpolation."""
    assert spec.eta == 0, "step skipping is defined for the deterministic sampler"
    f = den.eps_fn(cond, uncond, spec.guidance)
    states = [initial_noise(seed, spec)]
    i = 0
    while i < spec.steps:
        z = states[-1]
        eps = f(z, int(spec.timesteps()[i]))
        if i + 1 < spec.steps:
            _, a_t, _, _ = ddim_coeffs(i, spec)
            _, _, a_prev2, _ = ddim_coeffs(i + 1, spec)
            x0 = (z.astype(np.float64) - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)
            z2 = (np.sqrt(a_prev2) * x0 + np.sqrt(1 - a_prev2) * eps).astype(np.float32)
            states.append(((z + z2) / 2).astype(np.float32))
            states.append(z2)
            i += 2
        else:
            states.append(ddim_update(z, eps, i, spec, seed))
            i += 1
    return np.stack(states)


def neg_strip(den: Denoiser, cond, uncond_empty, seed: int, spec: SamplerSpec) -> np.ndarray:
    """Safety-guidance stripping: the job requires a safety negative prompt in the
    unconditional branch, but the worker uses the empty prompt instead. Callers pass the
    empty-prompt embedding as `uncond_empty`. Same cost as honest; it removes a safety layer."""
    return sample(den.eps_fn(cond, uncond_empty, spec.guidance), seed, spec)
