"""Proof-of-Denoising study on one laptop: PartiPrompts, FP16- and FP32-class cheats.

Work is done in blocks of BLOCK prompts so that partial overnight runs still give complete
prompts. Within a block every model is loaded once, used for all prompts, then freed
(memory stays bounded on a 32 GB machine).

Roles (run as separate processes so the GPU and the CPU work in parallel):
  gpu  - on --device (mps or cuda): honest workers <dev>-fp16 and <dev>-fp32; FP16-class
         cheats on <dev>-fp16; FP32-class cheats on <dev>-fp32; verification on <dev>-fp32
         (all steps for honest trajectories, K_RANDOM random steps for cheats)
         ("--role mps" is kept as an alias for "--role gpu --device mps")
  cpu  - honest worker cpu-fp32; verification on cpu-fp32 with K_CPU random steps of every
         trajectory whose block the gpu role has finished generating
Use --run-name to keep runs from different machines apart (default: study).

Challenge indices are drawn per (verifier, trajectory) from a seeded RNG, so runs are
reproducible and the indices are unpredictable to the worker. Outputs: runs/study/.
"""
import argparse
import gc
import hashlib
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

PROMPTS = [p["prompt"] for p in json.loads((ROOT / "prompts" / "parti_sample.json").read_text())]
SAFETY_NEG = "nudity, nsfw, gore, blood, violence"
SPEC = SamplerSpec(steps=25)
BLOCK = 8
import os as _os
K_RANDOM = int(_os.environ.get("POD_K_RANDOM", "8"))  # random challenged steps per cheat trajectory (25 = all)
K_CPU = 4
RUN = ROOT / "runs" / "study"
TRAJ = RUN / "traj"
CHECKS = RUN / "checks.jsonl"


def set_run(name: str):
    global RUN, TRAJ, CHECKS
    RUN = ROOT / "runs" / name
    TRAJ, CHECKS = RUN / "traj", RUN / "checks.jsonl"
CHEATS = ["sub-bk-sdm-small", "sub-bk-sdm-tiny", "w8", "w4", "cfg-skip", "cache2", "cache2-late", "step-skip", "neg-strip"]


def seed_of(p: int) -> int:
    return 5000 + p


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def save(path: Path, states: np.ndarray, secs: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, states=states, secs=secs)
    tmp.rename(path)


def gen(den: Denoiser, fn, prompts, tag: str):
    for p in prompts:
        path = TRAJ / f"{tag}__{p:02d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        states = fn(den, den.encode(PROMPTS[p]), den.encode(SAFETY_NEG), seed_of(p))
        save(path, states, time.time() - t0)
        print("gen", tag, p, f"{time.time() - t0:.1f}s", flush=True)


def generate_block(prompts, device: str):  # device: "cpu", "mps" or "cuda"
    """All trajectories for a block on one device class."""
    S = SPEC
    if device == "cpu":
        den = Denoiser(Backend("cpu", "fp32"))
        gen(den, lambda d, c, u, s: honest(d, c, u, s, S), prompts, "cpu-fp32__honest")
        free(den)
        return
    for dt in ("fp16", "fp32"):
        wb = f"{device}-{dt}"
        den = Denoiser(Backend(device, dt))
        gen(den, lambda d, c, u, s: honest(d, c, u, s, S), prompts, f"{wb}__honest")
        gen(den, lambda d, c, u, s: cfg_skip(d, c, u, s, S), prompts, f"{wb}__cfg-skip")
        gen(den, lambda d, c, u, s: eps_cache(d, c, u, s, S, period=2), prompts, f"{wb}__cache2")
        gen(den, lambda d, c, u, s: eps_cache(d, c, u, s, S, period=2, start=S.steps // 2), prompts, f"{wb}__cache2-late")
        gen(den, lambda d, c, u, s: step_skip(d, c, u, s, S), prompts, f"{wb}__step-skip")
        gen(den, lambda d, c, u, s: neg_strip(d, c, d.encode(""), s, S), prompts, f"{wb}__neg-strip")
        free(den)
        for unet in ("bk-sdm-small", "bk-sdm-tiny"):
            sub = Denoiser(Backend(device, dt), unet=unet)
            gen(sub, lambda d, c, u, s: honest(d, c, u, s, S), prompts, f"{wb}__sub-{unet}")
            free(sub)
        for bits in (8, 4):
            q = Denoiser(Backend(device, dt), quant_bits=bits)
            gen(q, lambda d, c, u, s: honest(d, c, u, s, S), prompts, f"{wb}__w{bits}")
            free(q)
    if device == "cuda":  # TF32 (default for cuDNN convs on Ampere+) as its own honest class
        den = Denoiser(Backend("cuda", "tf32"))
        gen(den, lambda d, c, u, s: honest(d, c, u, s, S), prompts, "cuda-tf32__honest")
        free(den)


def done_checks() -> set:
    if not CHECKS.exists():
        return set()
    return {(r["verifier"], r["file"]) for r in map(json.loads, CHECKS.read_text().splitlines()) if r}


def challenge(verifier: str, fname: str, k: int) -> list[int]:
    h = int(hashlib.sha256(f"{verifier}|{fname}".encode()).hexdigest()[:16], 16)
    return sorted(np.random.default_rng(h).choice(SPEC.steps, size=k, replace=False).tolist())


def verify_files(den: Denoiser, name: str, files, all_steps_for_honest: bool, k: int):
    done = done_checks()
    for path in files:
        if (name, path.name) in done:
            continue
        worker, kind, p = path.stem.split("__")
        p = int(p)
        idx = list(range(SPEC.steps)) if (all_steps_for_honest and kind == "honest") else challenge(name, path.name, k)
        states = np.load(path)["states"]
        t0 = time.time()
        checks = check_steps(den, den.encode(PROMPTS[p]), den.encode(SAFETY_NEG), states, idx, SPEC, seed_of(p))
        rec = {"verifier": name, "file": path.name, "worker": worker, "kind": kind, "prompt": p,
               "secs": round(time.time() - t0, 2), "checks": [c.to_dict() for c in checks]}
        with CHECKS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("verify", name, path.name, f"{rec['secs']}s", flush=True)


def block_files(prompts):
    ps = set(prompts)
    return sorted(f for f in TRAJ.glob("*.npz") if not f.name.endswith(".tmp.npz") and int(f.stem.split("__")[2]) in ps)


def marker(b: int) -> Path:
    return RUN / f"block{b:02d}.generated"


def run_gpu(blocks, device: str = "mps"):
    for b, prompts in blocks:
        if not marker(b).exists():
            generate_block(prompts, device)
            marker(b).write_text(time.ctime())
        den = Denoiser(Backend(device, "fp32"))
        verify_files(den, f"{device}-fp32", block_files(prompts), all_steps_for_honest=True, k=K_RANDOM)
        free(den)
        print(f"== {device} block done", b, flush=True)
    (RUN / f"{device}.finished").write_text(time.ctime())


def run_mps(blocks):  # backward-compatible name
    run_gpu(blocks, "mps")


def run_cpu(blocks):
    for b, prompts in blocks:
        generate_block(prompts, "cpu")
        while not marker(b).exists():
            time.sleep(60)
        den = Denoiser(Backend("cpu", "fp32"))
        verify_files(den, "cpu-fp32", block_files(prompts), all_steps_for_honest=False, k=K_CPU)
        free(den)
        print("== cpu block done", b, flush=True)
    (RUN / "cpu.finished").write_text(time.ctime())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["gpu", "mps", "cpu"], required=True)
    ap.add_argument("--device", choices=["mps", "cuda"], default="mps", help="GPU for --role gpu")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--run-name", default="study")
    a = ap.parse_args()
    set_run(a.run_name)
    n = min(a.prompts, len(PROMPTS))
    blocks = [(b, list(range(b * BLOCK, min((b + 1) * BLOCK, n)))) for b in range((n + BLOCK - 1) // BLOCK)]
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "spec.json").write_text(json.dumps({"spec": SPEC.to_dict(), "prompts": PROMPTS, "safety_neg": SAFETY_NEG, "seed_base": 5000,
                                               "block": BLOCK, "k_random": K_RANDOM, "k_cpu": K_CPU,
                                               "cheats": CHEATS}, indent=1))
    if a.role == "cpu":
        run_cpu(blocks)
    else:
        run_gpu(blocks, "mps" if a.role == "mps" else a.device)
