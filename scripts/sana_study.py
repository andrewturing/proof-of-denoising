"""Second model family for the CVPR study: SANA (linear DiT, rectified flow), via podlab/flow.py.

    # GPU: honest FP32/TF32/BF16 + 8 cheats in FP32 and BF16, verified on cuda-fp32
    python scripts/sana_study.py --role gpu --prompts 24 --run-name sana
    # CPU: a second FP32 stack checks every step of the honest CUDA-FP32 jobs (logged as x86cpu-fp32)
    python scripts/sana_study.py --role cpu --prompts 24 --run-name sana --tag x86cpu-fp32

Job: SANA-1.6B 512px, 25 Euler steps over shifted flow sigmas (shift 3), CFG 4.5 with the safety
negative prompt, Philox noise keyed by (seed_base + prompt, 0). Prompts: the study's PartiPrompts.
Writes runs/<run>/{spec.json, traj/<worker>__<kind>__<pp>.npz, checks.jsonl, img/}; the record
format matches scripts/study.py, so scripts/analyze_study.py <run> works unchanged.
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

from podlab import flow as F
from podlab.model import Backend

SAFETY_NEG = "nudity, nsfw, gore, blood, violence"
SEED_BASE = 7000
K_RANDOM = 12
SPEC = F.FlowSpec()
CHEATS = {  # kind -> (model, quant_bits, worker fn)
    "sub-sana600": ("sana600", None, F.honest),
    "w8": ("sana1600", 8, F.honest),
    "w4": ("sana1600", 4, F.honest),
    "cfg-skip": ("sana1600", None, F.cfg_skip),
    "cache2": ("sana1600", None, lambda d, c, u, s, sp: F.v_cache(d, c, u, s, sp, 2, 0)),
    "cache2-late": ("sana1600", None, lambda d, c, u, s, sp: F.v_cache(d, c, u, s, sp, 2, sp.steps // 2)),
    "step-skip": ("sana1600", None, F.step_skip),
    "neg-strip": ("sana1600", None, F.neg_strip),
}


def free(den):
    del den
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_all(den, prompts):
    out = []
    for p in prompts:
        c, u = den.encode(p, SAFETY_NEG)
        _, e = den.encode(p, "")  # empty negative: what a safety-stripping worker uses
        out.append((c, u, e))
    den.free_text_encoder()
    return out


def generate(run, prompts, backend, model, quant, kinds_fns):
    todo = [(k, fn, p) for k, fn in kinds_fns for p in range(len(prompts))
            if not (run / "traj" / f"{backend}__{k}__{p:02d}.npz").exists()]
    if not todo:
        return
    den = F.SanaDenoiser(Backend.parse(backend), model=model, quant_bits=quant)
    enc = encode_all(den, prompts)
    for kind, fn, p in todo:
        c, u, e = enc[p]
        t0 = time.time()
        states = fn(den, c, e if kind == "neg-strip" else u, SEED_BASE + p, SPEC)
        np.savez(run / "traj" / f"{backend}__{kind}__{p:02d}.npz", states=states, secs=time.time() - t0)
        print("gen", backend, kind, p, f"{time.time() - t0:.1f}s", flush=True)
    free(den)


def verify(run, prompts, verifier, tag, files, all_steps_for_honest, k, checks_name="checks.jsonl"):
    f_out = run / checks_name
    done = {(r["verifier"], r["file"]) for r in map(json.loads, f_out.read_text().splitlines()) if r} if f_out.exists() else set()
    files = [f for f in files if (tag, f.name) not in done]
    if not files:
        return
    den = F.SanaDenoiser(Backend.parse(verifier), model="sana1600")
    enc = encode_all(den, prompts)
    for path in files:
        worker, kind, p = path.stem.split("__")
        p = int(p)
        states = np.load(path)["states"]
        if kind == "honest" and all_steps_for_honest:
            idx = list(range(SPEC.steps))
        else:
            h = int(hashlib.sha256(f"{tag}|{path.name}".encode()).hexdigest()[:16], 16)
            idx = sorted(np.random.default_rng(h).choice(SPEC.steps, size=min(k, SPEC.steps), replace=False).tolist())
        c, u, _ = enc[p]
        t0 = time.time()
        checks = F.check_steps(den, c, u, states, idx, SPEC)
        rec = {"verifier": tag, "file": path.name, "worker": worker, "kind": kind, "prompt": p,
               "secs": round(time.time() - t0, 2), "checks": checks, "model": "sana1600"}
        with f_out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("verify", tag, path.name, f"{rec['secs']}s", "max r", f"{max(x['ratio'] for x in checks):.2e}", flush=True)
    free(den)


def decode_examples(run, n_prompts=4):
    from PIL import Image
    files = [f for f in sorted((run / "traj").glob("cuda-fp32__*.npz")) if int(f.stem.split("__")[2]) < n_prompts]
    (run / "img").mkdir(exist_ok=True)
    todo = [f for f in files if not (run / "img" / (f.stem + ".png")).exists()]
    if not todo:
        return
    den = F.SanaDenoiser(Backend.parse("cuda-fp32"), model="sana1600", load_vae=True)
    den.free_text_encoder()
    for f in todo:
        Image.fromarray(den.decode(np.load(f)["states"][-1])).save(run / "img" / (f.stem + ".png"))
    free(den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--run-name", default="sana")
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--tag", default="x86cpu-fp32")
    ap.add_argument("--precisions", default="fp32,tf32,bf16")
    ap.add_argument("--cpu-k", type=int, default=8, help="random steps per honest job for the CPU stack (25 = all)")
    ap.add_argument("--checks", default="checks.jsonl", help="log file under runs/<run>/ (a separate file allows a second, denser pass)")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run_name
    (run / "traj").mkdir(parents=True, exist_ok=True)
    prompts = json.loads((ROOT / "runs" / "study" / "spec.json").read_text())["prompts"][: a.prompts]
    (run / "spec.json").write_text(json.dumps({"model": "sana1600", "spec": SPEC.to_dict(), "prompts": prompts,
                                               "safety_neg": SAFETY_NEG, "seed_base": SEED_BASE, "k_random": K_RANDOM,
                                               "cheats": list(CHEATS)}, indent=1))
    if a.role == "gpu":
        precs = a.precisions.split(",")
        for pr in precs:
            generate(run, prompts, f"cuda-{pr}", "sana1600", None, [("honest", F.honest)])
        for pr in [p for p in precs if p in ("fp32", "bf16")]:
            by_model = {}
            for kind, (model, quant, fn) in CHEATS.items():
                by_model.setdefault((model, quant), []).append((kind, fn))
            for (model, quant), kf in by_model.items():
                generate(run, prompts, f"cuda-{pr}", model, quant, kf)
        files = sorted(f for f in (run / "traj").glob("cuda-*.npz"))
        verify(run, prompts, "cuda-fp32", "cuda-fp32", files, all_steps_for_honest=True, k=K_RANDOM)
        decode_examples(run)
    else:
        files = sorted((run / "traj").glob("cuda-fp32__honest__*.npz"))
        verify(run, prompts, "cpu-fp32", a.tag, files, all_steps_for_honest=a.cpu_k >= SPEC.steps, k=a.cpu_k, checks_name=a.checks)
    print("done", a.role, flush=True)


if __name__ == "__main__":
    main()
