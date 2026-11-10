"""End-to-end Proof-of-Denoising protocol: commit, challenge, open, verify.

    worker:   states, tree = run_and_commit(den_w, job)          # publish tree.root
    verifier: idx = draw_challenges(job.spec.steps, k)            # AFTER the root is published
    worker:   openings = open_steps(states, tree, idx)
    verifier: ok, checks = verify_openings(
                  den_v, job, tree.root, openings, taus, challenges=idx,
                  header_path=tree.path(0), initial_state=states[0],
                  initial_path=tree.path(1))

`taus` maps step index -> threshold for the job's precision class (calibrate them on honest
cross-device runs; see scripts/analyze_study.py). Challenges must come from fresh
randomness drawn after the commitment; deriving them from the root lets a cheater grind.
"""
from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field

import numpy as np

from .commit import MerkleTree, commit, header_leaf, state_leaf, verify_indexed_path
from .sampler import SamplerSpec, initial_noise, sample, step
from .verify import StepCheck


@dataclass(frozen=True)
class Job:
    prompt: str
    negative: str
    seed: int
    spec: SamplerSpec = field(default_factory=SamplerSpec)
    model_id: str = "sd15"  # e.g. a hash of the UNet safetensors
    precision_class: str = "fp32"  # pinned by the requester; sets which thresholds apply

    def header(self) -> dict:
        d = asdict(self)
        d["spec"] = self.spec.to_dict()
        return d


@dataclass
class Opening:
    i: int
    z_i: np.ndarray
    z_next: np.ndarray
    path_i: list
    path_next: list


def run_and_commit(den, job: Job) -> tuple[np.ndarray, MerkleTree]:
    """Honest worker: sample the full trajectory and commit to the header and every state."""
    f = den.eps_fn(den.encode(job.prompt), den.encode(job.negative), job.spec.guidance)
    states = sample(f, job.seed, job.spec)
    return states, commit(job.header(), states)


def draw_challenges(steps: int, k: int, rng: "np.random.Generator | None" = None) -> list[int]:
    """k distinct step indices. Explicit seeded generators are for experiments only.

    The caller must pin the worker's root before calling this function and retain
    the returned indices independently of the worker's response.
    """
    if (type(steps) is not int or type(k) is not int or steps <= 0 or not 1 <= k <= steps):
        raise ValueError("challenge budget must satisfy 1 <= k <= steps")
    if rng is None:
        return sorted(secrets.SystemRandom().sample(range(steps), k))
    return sorted(rng.choice(steps, size=k, replace=False).tolist())


def open_steps(states: np.ndarray, tree: MerkleTree, idx: list[int]) -> list[Opening]:
    """Reveal (z_i, z_{i+1}) for each challenged step with Merkle paths (leaf j+1 = state j)."""
    return [Opening(i, states[i], states[i + 1], tree.path(i + 1), tree.path(i + 2)) for i in idx]


def verify_openings(den, job: Job, root: bytes, openings: list[Opening], taus: dict[int, float],
                    *, challenges: list[int], header_path: list,
                    initial_state: np.ndarray, initial_path: list) -> tuple[bool, list[StepCheck]]:
    """Verify exactly the verifier-selected nonempty challenge set against its pinned root.

    Header and initial-noise proofs are mandatory even when step zero is not sampled.
    The caller supplies its own challenge list, job, root and calibrated thresholds,
    never values selected by the worker. A passing result means sampled transitions
    agree within tolerance; it does not attest the delivered image, safety, or the
    actual hardware/weights executed by the worker.
    """
    if (not isinstance(challenges, (list, tuple)) or not challenges
            or any(type(i) is not int or not 0 <= i < job.spec.steps for i in challenges)
            or len(set(challenges)) != len(challenges)
            or not isinstance(openings, (list, tuple))
            or any(not isinstance(o, Opening) or type(o.i) is not int for o in openings)
            or sorted(o.i for o in openings) != sorted(challenges)):
        return False, []

    def valid_state(z):
        # Hashing uses float32. Do not verify a different float64 value that merely
        # rounds to the committed bytes; also reject broadcasting and NaN/Inf inputs.
        return (isinstance(z, np.ndarray) and z.dtype == np.dtype("float32")
                and z.shape == job.spec.shape and bool(np.isfinite(z).all()))

    count = job.spec.steps + 2  # header plus N+1 states
    if (not verify_indexed_path(header_leaf(job.header()), header_path, root, 0, count)
            or not valid_state(initial_state)
            or not verify_indexed_path(state_leaf(0, initial_state), initial_path, root, 1, count)
            or not np.array_equal(initial_state, initial_noise(job.seed, job.spec))):
        return False, []
    # Validate the entire response before invoking the expensive denoiser.
    for o in openings:
        tau = taus.get(o.i)
        if (not isinstance(tau, (int, float, np.integer, np.floating))
                or isinstance(tau, (bool, np.bool_)) or not np.isfinite(tau) or tau < 0
                or not valid_state(o.z_i) or not valid_state(o.z_next)
                or not verify_indexed_path(state_leaf(o.i, o.z_i), o.path_i, root, o.i + 1, count)
                or not verify_indexed_path(state_leaf(o.i + 1, o.z_next), o.path_next, root, o.i + 2, count)):
            return False, []
    f = den.eps_fn(den.encode(job.prompt), den.encode(job.negative), job.spec.guidance)
    checks = []
    for o in openings:
        z_hat = step(f, o.z_i, o.i, job.spec, job.seed)
        if not valid_state(z_hat):
            return False, checks
        d = z_hat.astype(np.float64) - o.z_next
        s = z_hat.astype(np.float64) - o.z_i
        diff, sn = float(np.linalg.norm(d)), float(np.linalg.norm(s))
        checks.append(StepCheck(o.i, diff, sn, diff / max(sn, 1e-12), float(np.abs(d).max())))
    ok = all(np.isfinite(c.ratio) and c.ratio <= taus[c.i] for c in checks)
    return ok, checks
