"""Baseline at equal budget: job-level replay audits versus per-step spot checks, from stored trajectories.

A replay audit re-executes a whole job on the verifier's stack and compares final latents,
    d = ||z_N(worker) - z_N(verifier)|| / ||z_N(verifier)||,
against a tolerance, because honest stacks do not reproduce each other bit for bit. We give the baseline the same
threshold rule as the per-step checks: tau_out = 1.5 x max honest d over the other prompts (leave-one-prompt-out).

  honest pairs   the same FP32 job rendered independently on the NVIDIA L4 (runs/nvidia48) and on the Apple GPU
                 (runs/study for prompts 0-7, runs/calib_prompts for the rest)
  deviations     the L4's deviating trajectories for that job (and its honest TF32 / FP16 runs claimed as FP32),
                 compared with the Apple GPU's honest replay

A full replay (budget 100%) detects a deviation when d > tau_out. A random job audit with budget b replays a
fraction b of jobs, so its per-job detection is b times that. Per-step spot checks at k = 2 spend b = 2/25.

    python scripts/baseline_compare.py
Writes runs/baseline_compare.json (read by export_macros_cvpr.py).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
N = 25
KINDS = ["sub-bk-sdm-small", "sub-bk-sdm-tiny", "w8", "w4", "cfg-skip", "neg-strip", "step-skip", "cache2", "cache2-late"]
DOWNGRADES = {"tf32-as-fp32": "cuda-tf32", "fp16-as-fp32": "cuda-fp16"}


def final(path: Path):
    return np.load(path)["states"][-1].astype(np.float64) if path.exists() else None


def apple(p: int):
    for run in ("study", "calib_prompts"):
        z = final(RUNS / run / "traj" / f"mps-fp32__honest__{p:02d}.npz")
        if z is not None:
            return z
    return None


def dist(z, ref) -> float:
    return float(np.linalg.norm(z - ref) / np.linalg.norm(ref))


def main():
    L4 = RUNS / "nvidia48" / "traj"
    prompts = [p for p in range(48) if apple(p) is not None and (L4 / f"cuda-fp32__honest__{p:02d}.npz").exists()]
    ref = {p: apple(p) for p in prompts}
    honest = {p: dist(final(L4 / f"cuda-fp32__honest__{p:02d}.npz"), ref[p]) for p in prompts}
    tau = {p: 1.5 * max(d for q, d in honest.items() if q != p) for p in prompts}  # leave-one-prompt-out
    bit_exact = sum(d == 0.0 for d in honest.values())
    out = {"prompts": len(prompts), "budget_k2": 2 / N,
           "honest": {"median": float(np.median(list(honest.values()))), "max": max(honest.values()),
                      "min": min(honest.values()), "bit_exact_pairs": bit_exact,
                      "flagged": sum(honest[p] > tau[p] for p in prompts)},
           "deviations": {}}
    rows = {k: ("cuda-fp32", k) for k in KINDS}
    rows.update({name: (backend, "honest") for name, backend in DOWNGRADES.items()})
    for name, (backend, kind) in rows.items():
        ds = {p: dist(z, ref[p]) for p in prompts if (z := final(L4 / f"{backend}__{kind}__{p:02d}.npz")) is not None}
        if not ds:
            continue
        det = float(np.mean([ds[p] > tau[p] for p in ds]))
        out["deviations"][name] = {"n": len(ds), "median_d": float(np.median(list(ds.values()))), "min_d": min(ds.values()),
                                   "replay_detect": det, "job_audit_detect_at_k2_budget": det * 2 / N}
    (RUNS / "baseline_compare.json").write_text(json.dumps(out, indent=1))
    h = out["honest"]
    print(f"{len(prompts)} prompts; honest L4 vs Apple GPU final-latent distance: median {h['median']:.2e}, max {h['max']:.2e}, "
          f"bit-exact {bit_exact}, flagged (held out) {h['flagged']}")
    for k, v in out["deviations"].items():
        print(f"  {k:18s} n={v['n']:2d} median d={v['median_d']:.2e} min d={v['min_d']:.2e} replay detects {v['replay_detect']:.2f} "
              f"-> at the k=2 budget {v['job_audit_detect_at_k2_budget']:.3f}")


if __name__ == "__main__":
    main()
