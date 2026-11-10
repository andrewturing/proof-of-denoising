"""Precision-downgrade cheat: a job pins the FP32 class, the worker silently runs a cheaper class
(CUDA TF32: PyTorch's default for cuDNN convolutions on Ampere and later; or FP16).

Uses only logged honest trajectories of the cheaper class (they are exactly what such a worker
commits) checked by FP32 verifiers, against FP32-class thresholds calibrated leave-one-prompt-out
on honest cross-device FP32 pairs from every stack (tau_i = 1.5 x max).

    python scripts/precision_downgrade.py [run+run+...]
Writes runs/precision_downgrade.json.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
NAMES = (sys.argv[1] if len(sys.argv) > 1 else "study+calib_webgpu-fp32+nvidia+cross_nvidia+cross_mac+arbiter_cuda").split("+")
MARGIN = 1.5


def cls(b):
    return b.rsplit("-", 1)[-1]


def main():
    recs = [json.loads(l) for n in NAMES if (RUNS / n / "checks.jsonl").exists()
            for l in (RUNS / n / "checks.jsonl").read_text().splitlines() if l.strip()]
    cal = [r for r in recs if r["kind"] == "honest" and cls(r["worker"]) == "fp32" and cls(r["verifier"]) == "fp32"
           and r["worker"] != r["verifier"]]
    taus = {}
    for p in sorted({r["prompt"] for r in recs}):
        acc = defaultdict(list)
        for c in cal:
            if c["prompt"] != p:
                for x in c["checks"]:
                    acc[x["i"]].append(x["ratio"])
        taus[p] = {i: MARGIN * max(v) for i, v in acc.items()}
    out = {"runs": NAMES, "margin": MARGIN, "calibration_records": len(cal), "downgrades": {}}
    for worker in sorted({r["worker"] for r in recs if r["kind"] == "honest" and cls(r["worker"]) != "fp32"}):
        rs = [r for r in recs if r["kind"] == "honest" and r["worker"] == worker and cls(r["verifier"]) == "fp32"]
        flags = steps = 0
        pk = []
        for r in rs:
            f = [x["ratio"] > taus[r["prompt"]].get(x["i"], np.inf) for x in r["checks"]]
            flags, steps = flags + sum(f), steps + len(f)
            if len(f) >= 2:
                pk.append(1 - math.comb(len(f) - sum(f), 2) / math.comb(len(f), 2))
        out["downgrades"][worker] = {"records": len(rs), "verifiers": sorted({r["verifier"] for r in rs}),
                                     "steps": steps, "flagged": flags, "flag_frac": flags / max(steps, 1),
                                     "P_k2_mean": float(np.mean(pk)) if pk else None, "P_k2_min": float(np.min(pk)) if pk else None}
        print(f"{worker} claimed as FP32: {flags}/{steps} steps flagged; P(detect, k=2) mean "
              f"{out['downgrades'][worker]['P_k2_mean']:.3f} min {out['downgrades'][worker]['P_k2_min']:.3f} ({len(rs)} jobs)")
    (RUNS / "precision_downgrade.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
