"""Proof-of-Denoising pilot on one Apple-silicon laptop (no NVIDIA GPU).

Stages (each one resumes; finished outputs are skipped):
  honest  - honest trajectories on every worker backend
  cheats  - cheating trajectories, produced on the mps-fp16 worker
  verify  - every verifier backend recomputes challenged steps of every trajectory
  decode  - decode final latents to images on one fixed backend (for output-level drift)

Outputs go to runs/pilot/ (gitignored). Prompts are generic and written for this pilot.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from podlab.model import Backend, Denoiser
from podlab.sampler import SamplerSpec
from podlab.verify import check_steps
from podlab.workers import cfg_skip, eps_cache, honest, neg_strip, step_skip

PROMPTS = [
    "a red bicycle leaning against a brick wall",
    "a bowl of oranges on a wooden kitchen table",
    "a lighthouse on a rocky coast at sunset",
    "two cats sleeping on a blue sofa",
    "a snowy mountain village at night with warm lights",
    "a close-up photo of a sunflower with a bee",
    "an astronaut riding a horse on the moon, digital art",
    "a busy city street in the rain, reflections on the pavement",
    "a watercolor painting of a sailboat on a calm lake",
    "a cup of coffee and a croissant on a cafe table",
    "a green parrot sitting on a tree branch in a rainforest",
    "a vintage car parked in front of a diner",
]
SAFETY_NEG = "nudity, nsfw, gore, blood, violence"
SPEC = SamplerSpec(steps=25)
CHEAT_WORKER = "mps-fp16"
CPU_STEPS = [0, 4, 8, 12, 16, 20, 24]

RUN = ROOT / "runs" / "pilot"
TRAJ, IMG = RUN / "traj", RUN / "img"
CHECKS = RUN / "checks.jsonl"

_models: dict = {}


def model(backend: str, unet: str = "sd15", quant: int | None = None, vae: bool = False) -> Denoiser:
    key = (backend, unet, quant, vae)
    if key not in _models:
        _models[key] = Denoiser(Backend.parse(backend), unet=unet, quant_bits=quant, load_vae=vae)
    return _models[key]


def seed_of(p: int) -> int:
    return 1000 + p


def save(path: Path, states: np.ndarray, secs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=states, secs=secs)


def stage_honest(workers: list[str], prompts: range) -> None:
    for wb in workers:
        for p in prompts:
            path = TRAJ / f"{wb}__honest__{p:02d}.npz"
            if path.exists():
                continue
            den = model(wb)
            t0 = time.time()
            states = honest(den, den.encode(PROMPTS[p]), den.encode(SAFETY_NEG), seed_of(p), SPEC)
            save(path, states, time.time() - t0)
            print("honest", wb, p, f"{time.time() - t0:.1f}s", flush=True)


def stage_cheats(prompts: range) -> None:
    wb = CHEAT_WORKER

    def base():
        return model(wb)

    cheats = {
        "sub-bk-small": lambda c, u, s: honest(model(wb, "bk-sdm-small"), c, u, s, SPEC),
        "sub-bk-tiny": lambda c, u, s: honest(model(wb, "bk-sdm-tiny"), c, u, s, SPEC),
        "w8": lambda c, u, s: honest(model(wb, quant=8), c, u, s, SPEC),
        "w4": lambda c, u, s: honest(model(wb, quant=4), c, u, s, SPEC),
        "cfg-skip": lambda c, u, s: cfg_skip(base(), c, u, s, SPEC),
        "cache2": lambda c, u, s: eps_cache(base(), c, u, s, SPEC, period=2),
        "cache2-late": lambda c, u, s: eps_cache(base(), c, u, s, SPEC, period=2, start=SPEC.steps // 2),
        "step-skip": lambda c, u, s: step_skip(base(), c, u, s, SPEC),
        "neg-strip": lambda c, u, s: neg_strip(base(), c, base().encode(""), s, SPEC),
    }
    for name, fn in cheats.items():
        for p in prompts:
            path = TRAJ / f"{wb}__{name}__{p:02d}.npz"
            if path.exists():
                continue
            den = base()
            t0 = time.time()
            try:
                states = fn(den.encode(PROMPTS[p]), den.encode(SAFETY_NEG), seed_of(p))
            except Exception as e:  # a missing optional model must not stop the stage
                print("skip", name, type(e).__name__, str(e)[:120], flush=True)
                break
            save(path, states, time.time() - t0)
            print("cheat", name, p, f"{time.time() - t0:.1f}s", flush=True)


def done_checks() -> set:
    if not CHECKS.exists():
        return set()
    return {(r["verifier"], r["file"]) for r in map(json.loads, CHECKS.read_text().splitlines())}


def stage_verify(verifiers: list[str]) -> None:
    done = done_checks()
    for vb in verifiers:
        idx = list(range(SPEC.steps)) if vb.startswith("mps") else CPU_STEPS
        for path in sorted(TRAJ.glob("*.npz")):
            if (vb, path.name) in done:
                continue
            worker, kind, p = path.stem.split("__")
            p = int(p)
            den = model(vb)
            states = np.load(path)["states"]
            t0 = time.time()
            checks = check_steps(den, den.encode(PROMPTS[p]), den.encode(SAFETY_NEG), states, idx, SPEC, seed_of(p))
            rec = {"verifier": vb, "file": path.name, "worker": worker, "kind": kind, "prompt": p,
                   "secs": round(time.time() - t0, 2), "checks": [c.to_dict() for c in checks]}
            with CHECKS.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print("verify", vb, path.name, f"{rec['secs']}s", flush=True)


def stage_decode(decoder: str) -> None:
    for path in sorted(TRAJ.glob("*.npz")):
        out = IMG / f"{path.stem}.png"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        z0 = np.load(path)["states"][-1]
        Image.fromarray(model(decoder, vae=True).decode(z0)).save(out)
        print("decode", path.stem, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="honest,cheats,verify,decode")
    ap.add_argument("--workers", default="mps-fp16,mps-fp32,cpu-fp32")
    ap.add_argument("--verifiers", default="mps-fp32,mps-fp16,cpu-fp32")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--decoder", default="mps-fp32")
    a = ap.parse_args()
    prompts = range(a.prompts)
    for st in a.stages.split(","):
        print("== stage", st, flush=True)
        if st == "honest":
            stage_honest(a.workers.split(","), prompts)
        elif st == "cheats":
            stage_cheats(prompts)
        elif st == "verify":
            stage_verify(a.verifiers.split(","))
        elif st == "decode":
            stage_decode(a.decoder)
    (RUN / "spec.json").write_text(json.dumps({"spec": SPEC.to_dict(), "prompts": PROMPTS, "safety_neg": SAFETY_NEG, "seed_base": 1000,
                                               "cheat_worker": CHEAT_WORKER, "cpu_steps": CPU_STEPS}, indent=2))
