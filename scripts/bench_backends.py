"""Time one guided denoising step (two UNet evaluations at 512x512) on each backend."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from podlab.model import Backend, Denoiser
from podlab.sampler import SamplerSpec, initial_noise, step

ap = argparse.ArgumentParser()
ap.add_argument("--backends", default="mps-fp16,mps-fp32,cpu-fp32")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", type=Path, default=None, help="also write the JSON here")
args = ap.parse_args()

spec = SamplerSpec()
z = initial_noise(0, spec)
out = {}
for name in args.backends.split(","):
    t0 = time.time()
    if name.startswith("trt-"):
        try:
            from podlab.trt import make
            den = make(name)
        except Exception as e:  # TensorRT is optional; report and continue
            print(name, "SKIPPED:", type(e).__name__, str(e)[:200], flush=True)
            out[name] = {"skipped": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
    else:
        den = Denoiser(Backend.parse(name))
    load_s = time.time() - t0
    cond, uncond = den.encode("a red bicycle leaning against a brick wall"), den.encode("")
    f = den.eps_fn(cond, uncond, spec.guidance)
    step(f, z, 0, spec, 0)  # warm-up
    den.sync()
    t0 = time.time()
    for _ in range(args.reps):
        step(f, z, 0, spec, 0)
    den.sync()
    out[name] = {"load_s": round(load_s, 1), "step_s": round((time.time() - t0) / args.reps, 3)}
    print(name, out[name], flush=True)
    del den
print(json.dumps(out))
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
