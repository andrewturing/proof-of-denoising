"""Generate honest trajectories only (calibration material for a verifier stack), with exactly the
study's job definition: same SamplerSpec, PartiPrompts, safety negative prompt and seeds.

    python scripts/gen_honest.py --backend mps-fp32 --from 8 --to 32 --run-name calib_prompts

Writes runs/<run>/{spec.json, traj/<backend>__honest__<pp>.npz}. Prompts here must never be
test pages (the storybook uses its own panel prompts).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from podlab.model import Backend, Denoiser
from podlab.sampler import SamplerSpec
from podlab.workers import honest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mps-fp32")
    ap.add_argument("--from", dest="lo", type=int, default=8)
    ap.add_argument("--to", dest="hi", type=int, default=32)
    ap.add_argument("--run-name", default="calib_prompts")
    ap.add_argument("--prompts-file", default=None, help="e.g. prompts/parti_calib.json (default: the study's prompts)")
    ap.add_argument("--seed-base", type=int, default=None, help="seed of prompt p is seed_base + p (default: the study's)")
    ap.add_argument("--name", default=None, help="worker name in the file names (default: the backend), e.g. t4-fp32 or a100-fp32")
    a = ap.parse_args()
    study = json.loads((ROOT / "runs" / "study" / "spec.json").read_text())
    if a.prompts_file:  # same job definition, other prompts and seeds; spec.json records them for the verifiers
        study["prompts"] = [p["prompt"] for p in json.loads((ROOT / a.prompts_file).read_text())]
        study["prompts_file"] = a.prompts_file
    if a.seed_base is not None:
        study["seed_base"] = a.seed_base
    spec = SamplerSpec(**study["spec"])
    run = ROOT / "runs" / a.run_name
    (run / "traj").mkdir(parents=True, exist_ok=True)
    (run / "spec.json").write_text(json.dumps(study, indent=1))
    den = Denoiser(Backend.parse(a.backend))
    unc = den.encode(study["safety_neg"])
    for p in range(a.lo, a.hi):
        out = run / "traj" / f"{a.name or a.backend}__honest__{p:02d}.npz"
        if out.exists():
            continue
        t0 = time.time()
        states = honest(den, den.encode(study["prompts"][p]), unc, study["seed_base"] + p, spec)
        np.savez(out, states=states, secs=time.time() - t0)
        print("honest", a.name or a.backend, p, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
