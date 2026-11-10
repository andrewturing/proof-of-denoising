"""Cross-device verification: verify trajectories produced on one machine with a backend on
another (e.g. Apple-GPU trajectories on an NVIDIA GPU, or NVIDIA trajectories on the Mac).

    python scripts/cross_verify.py --traj-run study --verifier cuda-fp32 --k 8 --out-run cross_nvidia

Each trajectory gets k seeded random challenged steps (seed derived from verifier + file
name, so reruns are reproducible). Records are appended to runs/<out-run>/checks.jsonl in
the same format as the study, so scripts/analyze_study.py <out-run> works on them.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from podlab.model import Backend, Denoiser
from podlab.sampler import SamplerSpec
from podlab.verify import check_steps

SAFETY_NEG = "nudity, nsfw, gore, blood, violence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-run", required=True, help="run directory under runs/ holding traj/ and spec.json")
    ap.add_argument("--verifier", required=True, help="e.g. cuda-fp32, mps-fp32, cpu-fp32")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--only-kinds", default="", help="comma list, e.g. honest,w8 (default: all)")
    ap.add_argument("--only-workers", default="", help="comma list, e.g. cuda-fp32 (default: all)")
    ap.add_argument("--only-prompts", type=int, default=0, help="use prompts < N only (0 = all)")
    ap.add_argument("--tag", default="", help="name logged as verifier, e.g. x86cpu-fp32 (default: --verifier)")
    a = ap.parse_args()
    vname = a.tag or a.verifier
    src = ROOT / "runs" / a.traj_run
    spec_doc = json.loads((src / "spec.json").read_text())
    prompts = spec_doc["prompts"]
    spec = SamplerSpec(**spec_doc["spec"])
    seed_base = spec_doc.get("seed_base", 1000 if a.traj_run.startswith("pilot") else 5000)
    out = ROOT / "runs" / a.out_run
    out.mkdir(parents=True, exist_ok=True)
    checks = out / "checks.jsonl"
    done = set()
    if checks.exists():
        done = {(r["verifier"], r["file"]) for r in map(json.loads, checks.read_text().splitlines()) if r}
    kinds = set(filter(None, a.only_kinds.split(",")))
    workers = set(filter(None, a.only_workers.split(",")))
    if a.verifier.startswith("trt-"):
        from podlab.trt import make
        den = make(a.verifier)
    else:
        den = Denoiser(Backend.parse(a.verifier))
    for path in sorted((src / "traj").glob("*.npz")):
        if path.name.startswith("._") or path.name.endswith(".tmp.npz") or (vname, path.name) in done:  # ._ = macOS tar metadata
            continue
        worker, kind, p = path.stem.split("__")
        if (kinds and kind not in kinds) or (workers and worker not in workers):
            continue
        p = int(p)
        if a.only_prompts and p >= a.only_prompts:
            continue
        h = int(hashlib.sha256(f"{vname}|{path.name}".encode()).hexdigest()[:16], 16)
        idx = sorted(np.random.default_rng(h).choice(spec.steps, size=min(a.k, spec.steps), replace=False).tolist())
        states = np.load(path)["states"]
        t0 = time.time()
        res = check_steps(den, den.encode(prompts[p]), den.encode(SAFETY_NEG), states, idx, spec, seed_base + p)
        rec = {"verifier": vname, "file": path.name, "worker": worker, "kind": kind, "prompt": p,
               "secs": round(time.time() - t0, 2), "checks": [c.to_dict() for c in res], "traj_run": a.traj_run}
        with checks.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("xverify", vname, path.name, f"{rec['secs']}s", flush=True)
    (out / "spec.json").write_text(json.dumps(spec_doc, indent=1))


if __name__ == "__main__":
    main()
