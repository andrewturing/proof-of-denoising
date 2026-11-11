"""The nine shortcuts on a larger, disjoint prompt set, with every step recomputed.

The main study runs its deviations on 48 PartiPrompts. This script repeats the nine FP32 shortcuts on the 240
calibration prompts (prompts/parti_calib.json, disjoint from the study; the same jobs and seeds as runs/calib240,
whose honest trajectories the Apple GPU already re-checked) and recomputes all 25 steps of every shortcut trajectory
on the same GPU. That gives each trajectory's exact set of detectable steps and its smallest margin.

    python scripts/shortcut_scale.py --device cuda --prompts 240
Writes runs/scale240/{spec.json, checks.jsonl} in the study's record format (trajectories are not kept).
Thresholds are applied afterwards by scripts/scale_report.py (the frozen pooled FP32 thresholds).
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from podlab.model import Backend, Denoiser
from podlab.sampler import SamplerSpec
from podlab.verify import check_steps
from podlab.workers import cfg_skip, eps_cache, honest, neg_strip, step_skip

SAFETY_NEG = "nudity, nsfw, gore, blood, violence"
SPEC = SamplerSpec(steps=25)
SEED_BASE = 9000  # as runs/calib240
CHEATS = ["sub-bk-sdm-small", "sub-bk-sdm-tiny", "w8", "w4", "cfg-skip", "cache2", "cache2-late", "step-skip", "neg-strip"]
SAME_MODEL = {"honest": lambda d, c, u, s: honest(d, c, u, s, SPEC),
              "cfg-skip": lambda d, c, u, s: cfg_skip(d, c, u, s, SPEC),
              "cache2": lambda d, c, u, s: eps_cache(d, c, u, s, SPEC, period=2),
              "cache2-late": lambda d, c, u, s: eps_cache(d, c, u, s, SPEC, period=2, start=SPEC.steps // 2),
              "step-skip": lambda d, c, u, s: step_skip(d, c, u, s, SPEC),
              "neg-strip": lambda d, c, u, s: neg_strip(d, c, d.encode(""), s, SPEC)}
OTHER_MODEL = {"sub-bk-sdm-small": dict(unet="bk-sdm-small"), "sub-bk-sdm-tiny": dict(unet="bk-sdm-tiny"),
               "w8": dict(quant_bits=8), "w4": dict(quant_bits=4)}


def free(den):
    del den
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompts", type=int, default=240)
    ap.add_argument("--block", type=int, default=24, help="prompts per block (each model is loaded once per block)")
    ap.add_argument("--run-name", default="scale240")
    a = ap.parse_args()
    prompts = [p["prompt"] for p in json.loads((ROOT / "prompts" / "parti_calib.json").read_text())][: a.prompts]
    run = ROOT / "runs" / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "spec.json").write_text(json.dumps({"spec": SPEC.to_dict(), "prompts": prompts, "prompts_file": "prompts/parti_calib.json",
                                               "safety_neg": SAFETY_NEG, "seed_base": SEED_BASE, "cheats": CHEATS,
                                               "checked_steps": "all 25 (shortcuts), 2 (honest, same device)"}, indent=1))
    checks = run / "checks.jsonl"
    done = {(r["kind"], r["prompt"]) for r in map(json.loads, checks.read_text().splitlines()) if r} if checks.exists() else set()
    backend = f"{a.device}-fp32"
    for b0 in range(0, len(prompts), a.block):
        ps = [p for p in range(b0, min(b0 + a.block, len(prompts))) if any((k, p) not in done for k in ["honest"] + CHEATS)]
        if not ps:
            continue
        t_block = time.time()
        traj = {}
        den = Denoiser(Backend(a.device, "fp32"))
        unc = den.encode(SAFETY_NEG)
        cond = {p: den.encode(prompts[p]) for p in ps}
        for kind, fn in SAME_MODEL.items():
            for p in ps:
                if (kind, p) not in done:
                    traj[(kind, p)] = fn(den, cond[p], unc, SEED_BASE + p)
        free(den)
        for kind, kw in OTHER_MODEL.items():
            sub = Denoiser(Backend(a.device, "fp32"), **kw)
            for p in ps:
                if (kind, p) not in done:
                    traj[(kind, p)] = honest(sub, cond[p], unc, SEED_BASE + p, SPEC)
            free(sub)
        den = Denoiser(Backend(a.device, "fp32"))
        for (kind, p), states in sorted(traj.items(), key=lambda x: (x[0][1], x[0][0])):
            idx = [3, 17] if kind == "honest" else list(range(SPEC.steps))
            t0 = time.time()
            res = check_steps(den, cond[p], unc, states, idx, SPEC, SEED_BASE + p)
            rec = {"verifier": backend, "file": f"{backend}__{kind}__{p:03d}.npz", "worker": backend, "kind": kind, "prompt": p,
                   "secs": round(time.time() - t0, 2), "checks": [c.to_dict() for c in res]}
            with checks.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done.add((kind, p))
        free(den)
        print(f"block {b0}-{ps[-1]} done in {time.time() - t_block:.0f}s ({len(traj)} trajectories)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
