"""Honest FP32 output drift across stacks: the same job's final latent from two stacks, both decoded by ONE
decoder (so only trajectory drift is measured). The yardstick for how far a within-threshold attacker moves
the image (scripts/adaptive_attacks.py).

    python scripts/output_drift.py [--device mps]
Writes runs/output_drift_fp32.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from podlab.model import Backend, Denoiser

SOURCES = {"mps-fp32": "study", "cpu-fp32": "study", "cuda-fp32": "nvidia"}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prompts", type=int, default=8)
    a = ap.parse_args()
    den = Denoiser(Backend(a.device, "fp32"), load_vae=True)
    rows = []
    for p in range(a.prompts):
        zs = {}
        for b, run in SOURCES.items():
            f = ROOT / "runs" / run / "traj" / f"{b}__honest__{p:02d}.npz"
            if f.exists():
                zs[b] = np.load(f)["states"][-1]
        imgs = {b: den.decode(z) for b, z in zs.items()}
        for x, y in itertools.combinations(sorted(zs), 2):
            dz = float(np.linalg.norm(zs[x].astype(np.float64) - zs[y]) / np.linalg.norm(zs[x].astype(np.float64)))
            rows.append({"prompt": p, "pair": f"{x}~{y}", "psnr": psnr(imgs[x], imgs[y]), "latent_rel": dz})
            print(rows[-1], flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["pair"], []).append(r)
    summ = {k: {"n": len(v), "psnr_min": min(r["psnr"] for r in v), "psnr_median": float(np.median([r["psnr"] for r in v])),
                "latent_rel_max": max(r["latent_rel"] for r in v)} for k, v in by.items()}
    (ROOT / "runs" / "output_drift_fp32.json").write_text(json.dumps({"decoder": f"{a.device}-fp32", "pairs": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
