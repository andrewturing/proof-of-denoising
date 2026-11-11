"""The deviation study on further model families, with the same jobs, workers and checks as the SD1.5 and SANA
studies:

  sdxl   SDXL base 1.0 (2.6B UNet, 1024 x 1024, two text encoders), DDIM path (podlab/sdxl.py).
         Substitutes: SSD-1B and Segmind-Vega, distilled SDXL UNets.
  wan    Wan 2.1 text-to-video 1.3B (video transformer, flow matching), Euler path (podlab/flow.py).

    # GPU: honest runs in every precision class, every deviation in FP32, all checked on cuda-fp32
    python scripts/family_study.py --family sdxl --role gpu --prompts 16
    # a second FP32 stack re-checks k random steps of the honest FP32 jobs (the host CPU, or another machine)
    python scripts/family_study.py --family sdxl --role verify --verifier cpu-fp32 --tag x86cpu-fp32 --k 4
    # Wan: the text encoder runs once on the CPU (see podlab/flow.py); then as above
    python scripts/family_study.py --family wan --role text --prompts 8
    python scripts/family_study.py --family wan --role gpu --prompts 8
    # seconds per guided step on a backend (to size a run)
    python scripts/family_study.py --family wan --role time --verifier cuda-fp32

Jobs: 25 steps, classifier-free guidance with the safety negative prompt, Philox noise keyed by (seed_base + prompt).
Prompts are the study's PartiPrompts. Writes runs/<run>/{spec.json, traj/<worker>__<kind>__<pp>.npz, checks.jsonl,
img/}; the record format matches scripts/study.py, so scripts/analyze_study.py <run> works unchanged.
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
from podlab import workers as W
from podlab.model import Backend
from podlab.sampler import SamplerSpec
from podlab.verify import check_steps as ddim_check

SAFETY_NEG = "nudity, nsfw, gore, blood, violence"
K_RANDOM = 12

DDIM_WORKERS = {"honest": W.honest, "cfg-skip": W.cfg_skip, "step-skip": W.step_skip, "neg-strip": W.neg_strip,
                "cache2": lambda d, c, u, s, sp: W.eps_cache(d, c, u, s, sp, period=2),
                "cache2-late": lambda d, c, u, s, sp: W.eps_cache(d, c, u, s, sp, period=2, start=sp.steps // 2)}
FLOW_WORKERS = {"honest": F.honest, "cfg-skip": F.cfg_skip, "step-skip": F.step_skip, "neg-strip": F.neg_strip,
                "cache2": lambda d, c, u, s, sp: F.v_cache(d, c, u, s, sp, 2, 0),
                "cache2-late": lambda d, c, u, s, sp: F.v_cache(d, c, u, s, sp, 2, sp.steps // 2)}
SHORTCUTS = ["cfg-skip", "cache2", "cache2-late", "step-skip", "neg-strip"]


def sdxl_denoiser(backend, model, quant, vae=False):
    from podlab.sdxl import SDXLDenoiser
    return SDXLDenoiser(Backend.parse(backend), unet=model, quant_bits=quant, load_vae=vae)


def wan_denoiser(backend, model, quant, vae=False):
    return F.WanDenoiser(Backend.parse(backend), model=model, quant_bits=quant, load_vae=vae)


FAMILIES = {
    "sdxl": dict(kind="ddim", run="sdxl", main="sdxl", seed_base=11000, make=sdxl_denoiser, text="backend",
                 spec=SamplerSpec(steps=25, guidance=5.0, height=1024, width=1024), low=["tf32", "fp16"],
                 cheats={"sub-ssd1b": ("ssd1b", None, "honest"), "sub-vega": ("vega", None, "honest"),
                         "w8": ("sdxl", 8, "honest"), "w4": ("sdxl", 4, "honest"),
                         **{k: ("sdxl", None, k) for k in SHORTCUTS}}),
    "wan": dict(kind="flow", run="wan", main="wan13", seed_base=12000, make=wan_denoiser, text="cached",
                spec=F.FlowSpec(steps=25, guidance=5.0, shift=3.0, height=480, width=832, channels=16, down=8, frames=9),
                low=["tf32", "bf16"],
                cheats={"w8": ("wan13", 8, "honest"), "w4": ("wan13", 4, "honest"),
                        **{k: ("wan13", None, k) for k in SHORTCUTS}}),
}


def free(den):
    del den
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def key(text: str) -> str:
    return "t" + hashlib.sha256(text.encode()).hexdigest()[:16]


def encode_text(run, prompts):
    """Wan: encode every text once on the CPU and cache it (runs/<run>/text_emb.npz)."""
    f = run / "text_emb.npz"
    have = dict(np.load(f)) if f.exists() else {}
    todo = [t for t in dict.fromkeys(list(prompts) + [SAFETY_NEG, ""]) if key(t) not in have]
    if todo:
        enc = F.WanText()
        for t in todo:
            t0 = time.time()
            have[key(t)] = enc.encode(t)
            print("text", repr(t[:40]), f"{time.time() - t0:.1f}s", flush=True)
        np.savez(f, **have)
        free(enc)
    return have


def conds(fam, run, den, prompts):
    """Per prompt: (conditioning, safety-negative conditioning, empty-prompt conditioning)."""
    if fam["text"] == "cached":
        emb = encode_text(run, prompts)
        return [(emb[key(p)], emb[key(SAFETY_NEG)], emb[key("")]) for p in prompts]
    u, e = den.encode(SAFETY_NEG), den.encode("")
    out = [(den.encode(p), u, e) for p in prompts]
    den.free_text_encoder()
    return out


def generate(fam, run, prompts, backend, model, quant, kinds):
    todo = [(k, p) for k in kinds for p in range(len(prompts))
            if not (run / "traj" / f"{backend}__{k}__{p:02d}.npz").exists()]
    if not todo:
        return
    den = fam["make"](backend, model, quant)
    cs = conds(fam, run, den, prompts)
    fns = DDIM_WORKERS if fam["kind"] == "ddim" else FLOW_WORKERS
    for kind, p in todo:
        c, u, e = cs[p]
        fn = fns[fam["cheats"][kind][2]] if kind in fam["cheats"] else fns["honest"]
        t0 = time.time()
        states = fn(den, c, e if kind == "neg-strip" else u, fam["seed_base"] + p, fam["spec"])
        tmp = run / "traj" / f"{backend}__{kind}__{p:02d}.tmp.npz"
        np.savez(tmp, states=states, secs=time.time() - t0)
        tmp.rename(run / "traj" / f"{backend}__{kind}__{p:02d}.npz")
        print("gen", backend, kind, p, f"{time.time() - t0:.1f}s", flush=True)
    free(den)


def verify(fam, run, prompts, verifier, tag, files, all_honest, k, checks_name="checks.jsonl"):
    f_out = run / checks_name
    done = {(r["verifier"], r["file"]) for r in map(json.loads, f_out.read_text().splitlines()) if r} if f_out.exists() else set()
    files = [f for f in files if (tag, f.name) not in done]
    if not files:
        return
    den = fam["make"](verifier, fam["main"], None)
    cs = conds(fam, run, den, prompts)
    spec, n = fam["spec"], fam["spec"].steps
    for path in files:
        worker, kind, p = path.stem.split("__")
        p = int(p)
        states = np.load(path)["states"]
        if kind == "honest" and all_honest:
            idx = list(range(n))
        else:
            h = int(hashlib.sha256(f"{tag}|{path.name}".encode()).hexdigest()[:16], 16)
            idx = sorted(np.random.default_rng(h).choice(n, size=min(k, n), replace=False).tolist())
        c, u, _ = cs[p]
        t0 = time.time()
        if fam["kind"] == "ddim":
            checks = [x.to_dict() for x in ddim_check(den, c, u, states, idx, spec, fam["seed_base"] + p)]
        else:
            checks = F.check_steps(den, c, u, states, idx, spec)
        rec = {"verifier": tag, "file": path.name, "worker": worker, "kind": kind, "prompt": p,
               "secs": round(time.time() - t0, 2), "checks": checks, "model": fam["main"]}
        with f_out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("verify", tag, path.name, f"{rec['secs']}s", "max r", f"{max(x['ratio'] for x in checks):.2e}", flush=True)
    free(den)


def decode_examples(fam, run, n_prompts=2):
    """PNG for an image model; a strip of five frames for a video model."""
    from PIL import Image
    files = [f for f in sorted((run / "traj").glob("cuda-fp32__*.npz")) if int(f.stem.split("__")[2]) < n_prompts]
    (run / "img").mkdir(exist_ok=True)
    todo = [f for f in files if not (run / "img" / (f.stem + ".png")).exists()]
    if not todo:
        return
    den = fam["make"]("cuda-fp32", fam["main"], None, True)
    den.free_text_encoder()
    for f in todo:
        out = den.decode(np.load(f)["states"][-1])
        if out.ndim == 4:  # video (F, H, W, 3): five evenly spaced frames, side by side
            out = np.concatenate([out[i] for i in np.linspace(0, len(out) - 1, 5).round().astype(int)], axis=1)
        Image.fromarray(out).save(run / "img" / (f.stem + ".png"))
    free(den)


def time_step(fam, run, prompts, backend):
    den = fam["make"](backend, fam["main"], None)
    c, u, _ = conds(fam, run, den, prompts[:1])[0]
    spec = fam["spec"]
    if fam["kind"] == "ddim":
        from podlab.sampler import initial_noise, step
        z, f = initial_noise(1, spec), den.eps_fn(c, u, spec.guidance)
        run_step = lambda i: step(f, z, i, spec, 1)
    else:
        z, f = F.initial_noise(1, spec), den.v_fn(c, u, spec.guidance)
        run_step = lambda i: F.step(f, z, i, spec)
    run_step(0)
    den.sync()
    t0 = time.time()
    for i in (1, 2):
        run_step(i)
    den.sync()
    s = (time.time() - t0) / 2
    mem = f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if torch.cuda.is_available() else ""
    print(f"{backend}: {s:.2f} s per guided step, {s * spec.steps:.0f} s per job, latent {spec.shape}{mem}", flush=True)
    free(den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--role", choices=["gpu", "verify", "text", "time"], required=True)
    ap.add_argument("--run-name")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--kinds", help="comma-separated deviations to run (default: all)")
    ap.add_argument("--verifier", default="cpu-fp32", help="backend for --role verify / time, e.g. cpu-fp32, mps-fp32")
    ap.add_argument("--tag", help="verifier name in the log (default: the backend)")
    ap.add_argument("--files", default="cuda-fp32__honest__*.npz", help="trajectories for --role verify")
    ap.add_argument("--k", type=int, default=4, help="random steps per trajectory for --role verify (25 = all)")
    ap.add_argument("--checks", default="checks.jsonl")
    ap.add_argument("--frames", type=int, help="wan: latent frames (video frames = 4 x (frames - 1) + 1)")
    ap.add_argument("--size", help="HxW override, e.g. 320x576")
    a = ap.parse_args()
    fam = dict(FAMILIES[a.family])
    if a.frames or a.size:
        d = fam["spec"].to_dict()
        if a.frames:
            d["frames"] = a.frames
        if a.size:
            d["height"], d["width"] = map(int, a.size.split("x"))
        fam["spec"] = type(fam["spec"])(**d)
    run = ROOT / "runs" / (a.run_name or fam["run"])
    (run / "traj").mkdir(parents=True, exist_ok=True)
    prompts = json.loads((ROOT / "runs" / "study" / "spec.json").read_text())["prompts"][: a.prompts]
    kinds = [k for k in fam["cheats"] if not a.kinds or k in a.kinds.split(",")]
    (run / "spec.json").write_text(json.dumps({"model": fam["main"], "family": a.family, "spec": fam["spec"].to_dict(),
                                               "prompts": prompts, "safety_neg": SAFETY_NEG, "seed_base": fam["seed_base"],
                                               "k_random": K_RANDOM, "cheats": kinds}, indent=1))
    if a.role == "text":
        encode_text(run, prompts)
    elif a.role == "time":
        time_step(fam, run, prompts, a.verifier)
    elif a.role == "gpu":
        for pr in ["fp32"] + fam["low"]:
            generate(fam, run, prompts, f"cuda-{pr}", fam["main"], None, ["honest"])
        by_model = {}
        for kind in kinds:
            model, quant, _ = fam["cheats"][kind]
            by_model.setdefault((model, quant), []).append(kind)
        for (model, quant), ks in by_model.items():
            generate(fam, run, prompts, "cuda-fp32", model, quant, ks)
        files = sorted((run / "traj").glob("cuda-*.npz"))
        verify(fam, run, prompts, "cuda-fp32", "cuda-fp32", files, all_honest=True, k=K_RANDOM)
        decode_examples(fam, run)
    else:
        files = sorted((run / "traj").glob(a.files))
        verify(fam, run, prompts, a.verifier, a.tag or a.verifier, files, all_honest=a.k >= fam["spec"].steps, k=a.k,
               checks_name=a.checks)
    print("done", a.family, a.role, flush=True)


if __name__ == "__main__":
    main()
