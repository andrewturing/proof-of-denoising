"""How visible is each deviation in the image, and who catches it?

Decodes the final latents of the NVIDIA L4 study (runs/nvidia48: honest FP32, honest TF32 / FP16, nine deviations)
with the SD1.5 VAE and compares each image with the honest FP32 image of the same job: PSNR in dB (higher = closer;
above ~40 dB a difference is hard to see). Joins, per deviation:
  saved      the fraction of the job's compute it avoids. By definition for the step-level shortcuts (network
             evaluations skipped); from the L4 step-time benchmark for TF32 / FP16; not applicable for simulated
             quantization and for the safety strip. For a substitute UNet: its step time relative to SD1.5,
             measured here on this machine's GPU.
  replay     how often replaying the whole job detects it (runs/baseline_compare.json)
  pod        how often two step checks detect it (pooled summary; precision_downgrade.json)

    python scripts/perceptual_table.py [--device mps]
Writes runs/perceptual.json (read by export_macros_cvpr.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"
L4 = RUNS / "nvidia48" / "traj"
N = 25
ROWS = [  # name, trajectory prefix, how the saving is obtained
    ("tf32-as-fp32", "cuda-tf32__honest", "bench:cuda-tf32"),
    ("fp16-as-fp32", "cuda-fp16__honest", "bench:cuda-fp16"),
    ("cache2-late", "cuda-fp32__cache2-late", 6 / N),
    ("cache2", "cuda-fp32__cache2", 12 / N),
    ("step-skip", "cuda-fp32__step-skip", 12 / N),
    ("cfg-skip", "cuda-fp32__cfg-skip", 0.5),
    ("sub-bk-sdm-small", "cuda-fp32__sub-bk-sdm-small", "time:bk-sdm-small"),
    ("sub-bk-sdm-tiny", "cuda-fp32__sub-bk-sdm-tiny", "time:bk-sdm-tiny"),
    ("w8", "cuda-fp32__w8", None),
    ("w4", "cuda-fp32__w4", None),
    ("neg-strip", "cuda-fp32__neg-strip", 0.0),
]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def step_seconds(den, spec) -> float:
    from podlab.sampler import initial_noise, step
    c = den.encode("a red bicycle leaning against a brick wall")
    f, z = den.eps_fn(c, den.encode(""), spec.guidance), initial_noise(1, spec)
    step(f, z, 0, spec, 1)
    den.sync()
    t0 = time.time()
    for i in (1, 2, 3):
        step(f, z, i, spec, 1)
    den.sync()
    return (time.time() - t0) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    from podlab.model import Backend, Denoiser
    from podlab.sampler import SamplerSpec
    spec = SamplerSpec(**json.loads((RUNS / "nvidia48" / "spec.json").read_text())["spec"])
    den = Denoiser(Backend(a.device, "fp32"), load_vae=True)
    prompts = sorted(int(f.stem.rsplit("__", 1)[1]) for f in L4.glob("cuda-fp32__honest__*.npz"))
    decode = lambda prefix, p: den.decode(np.load(L4 / f"{prefix}__{p:02d}.npz")["states"][-1])
    honest = {p: decode("cuda-fp32__honest", p) for p in prompts}
    base_s = step_seconds(den, spec)
    out = {"prompts": len(prompts), "decoder": f"{a.device}-fp32", "rows": {}}
    vals = {}
    for name, prefix, _ in ROWS:
        vals[name] = [psnr(decode(prefix, p), honest[p]) for p in prompts if (L4 / f"{prefix}__{p:02d}.npz").exists()]
        print(f"{name:18s} PSNR median {np.median(vals[name]):5.1f} dB, min {min(vals[name]):5.1f}, max {max(vals[name]):5.1f} (n={len(vals[name])})", flush=True)
    del den
    bench = json.loads((RUNS / "bench_nvidia.json").read_text())
    bc = json.loads((RUNS / "baseline_compare.json").read_text())["deviations"]
    pool = max((d.name for d in RUNS.iterdir() if d.is_dir() and d.name.startswith("study+")), key=lambda n: n.count("+"))
    bw = json.loads((RUNS / pool / "summary.json").read_text())["classes"]["fp32"]["cheats_by_worker"]
    dg = json.loads((RUNS / "precision_downgrade.json").read_text())["downgrades"]
    for name, prefix, how in ROWS:
        if isinstance(how, str) and how.startswith("bench:"):
            saved, src = 1 - bench[how[6:]]["step_s"] / bench["cuda-fp32"]["step_s"], "L4 step time"
        elif isinstance(how, str) and how.startswith("time:"):
            sub = Denoiser(Backend(a.device, "fp32"), unet=how[5:])
            saved, src = 1 - step_seconds(sub, spec) / base_s, f"{a.device} step time"
            del sub
        else:
            saved, src = how, "by definition" if how is not None else "simulated"
        pod = dg[prefix.split("__")[0]]["P_k2_min"] if name.endswith("-as-fp32") else bw[f"{name}|cuda-fp32|cuda-fp32|max"]["P_k2"]
        out["rows"][name] = {"n": len(vals[name]), "psnr_median": float(np.median(vals[name])), "psnr_min": float(min(vals[name])),
                             "psnr_max": float(max(vals[name])), "saved": saved, "saved_source": src,
                             "replay_detect": bc[name]["replay_detect"], "pod_k2": pod}
    (RUNS / "perceptual.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()} for k, v in out["rows"].items()}, indent=1))


if __name__ == "__main__":
    main()
