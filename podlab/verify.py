"""Verifier side: recompute challenged steps on the verifier's own backend and measure how
far the committed next state is from the recomputation.

The test statistic for step i is the relative discrepancy
    r_i = ||Step_V(z_i) - z_{i+1}|| / ||Step_V(z_i) - z_i||
i.e. the mismatch between the committed and the recomputed next state, measured against
the size of the step itself. Thresholds are calibrated per step index from honest
cross-backend pairs (see calibrate.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model import Denoiser
from .sampler import SamplerSpec, step


@dataclass
class StepCheck:
    i: int
    diff: float  # ||Step_V(z_i) - z_{i+1}||
    step_norm: float  # ||Step_V(z_i) - z_i||
    ratio: float  # diff / step_norm
    maxabs: float  # max |Step_V(z_i) - z_{i+1}|

    def to_dict(self) -> dict:
        return asdict(self)


def check_steps(den: Denoiser, cond: np.ndarray, uncond: np.ndarray, states: np.ndarray,
                idx: list[int], spec: SamplerSpec, seed: int) -> list[StepCheck]:
    f = den.eps_fn(cond, uncond, spec.guidance)
    out = []
    for i in idx:
        z_hat = step(f, states[i], i, spec, seed)
        d = z_hat.astype(np.float64) - states[i + 1]
        s = z_hat.astype(np.float64) - states[i]
        diff, sn = float(np.linalg.norm(d)), float(np.linalg.norm(s))
        out.append(StepCheck(i, diff, sn, diff / max(sn, 1e-12), float(np.abs(d).max())))
    return out


def detection_probability(cheated_steps: int, steps: int, k: int) -> float:
    """P(at least one of k distinct uniformly drawn challenges hits a cheated step)."""
    from math import comb
    if k > steps:
        k = steps
    honest_steps = steps - cheated_steps
    if k > honest_steps:
        return 1.0
    return 1.0 - comb(honest_steps, k) / comb(steps, k)
