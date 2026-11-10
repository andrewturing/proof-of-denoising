"""Adaptive attacks on sampled-transition checks: full-cost workers that try to control the OUTPUT while
passing the checks. Parameters are fixed in advance (no tuning on the evaluated prompts).

  steer33 / steer95  every committed step is the honest step plus a perturbation of size
                     c * tau_i * ||Step(z_i) - z_i|| (c = 0.33 / 0.95) pointing at the same step of another
                     prompt's honest trajectory. With tau = 1.5 x max honest drift, c = 0.33 keeps r_i <= tau_i
                     against any verifier whose drift stays within the calibrated maximum; c = 0.95 relies on
                     the verifier drifting little. The question is how far the image can be moved.
  splice12 / splice20  honest up to z_s; the committed z_{s+1} is the target trajectory's z_{s+1}; honest
                     from there (with the job's own prompt). Exactly one inconsistent transition (m = 1).

Targets: the honest trajectory of prompt (p+1) mod n on the same device. Thresholds: the frozen pooled FP32
thresholds in runs/storybook_fresh/taus_fp32.json (policy fixed before these attacks were written).

    python scripts/adaptive_attacks.py --device cuda --honest-run nvidia48 --prompts 48
Writes runs/adaptive/{spec.json, traj/<backend>__<attack>__<pp>.npz, img/, outputs.jsonl}. Cross-device checks
of these trajectories run with scripts/cross_verify.py --traj-run adaptive --k 25.
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
from podlab.sampler import SamplerSpec, initial_noise, step
from podlab.workers import honest

STEER = {"steer33": 0.33, "steer95": 0.95}
SPLICE = {"splice12": 12, "splice20": 20}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def steer(f, seed, spec, taus, target, c):
    states = [initial_noise(seed, spec)]
    for i in range(spec.steps):
        z = states[-1]
        zh = step(f, z, i, spec, seed).astype(np.float64)
        d = target[i + 1].astype(np.float64) - zh
        s = float(np.linalg.norm(zh - z.astype(np.float64)))
        states.append((zh + c * taus[i] * s * d / max(float(np.linalg.norm(d)), 1e-12)).astype(np.float32))
    return np.stack(states)


def splice(f, seed, spec, base, target, s):
    states = list(base[: s + 1]) + [target[s + 1]]
    for i in range(s + 1, spec.steps):
        states.append(step(f, states[-1], i, spec, seed))
    return np.stack(states).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--honest-run", default="nvidia48", help="run whose traj/<backend>__honest__pp.npz are reused")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--taus", default="runs/storybook_fresh/taus_fp32.json")
    ap.add_argument("--images", type=int, default=6, help="save PNGs for the first N prompts")
    ap.add_argument("--out-run", default="adaptive")
    a = ap.parse_args()
    study = json.loads((ROOT / "runs" / "study" / "spec.json").read_text())
    spec = SamplerSpec(**study["spec"])
    taus_doc = json.loads((ROOT / a.taus).read_text())
    taus = {int(i): v for i, v in taus_doc["tau"].items()}
    backend = f"{a.device}-fp32"
    out = ROOT / "runs" / a.out_run
    (out / "traj").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    (out / "spec.json").write_text(json.dumps({**study, "attacks": {**STEER, **SPLICE}, "taus": a.taus,
                                               "taus_policy": taus_doc.get("policy"), "target": "(p+1) mod n"}, indent=1))
    den = Denoiser(Backend(a.device, "fp32"), load_vae=True)
    unc = den.encode(study["safety_neg"])
    n = a.prompts
    hon = {}

    def honest_states(p):
        if p not in hon:
            f = ROOT / "runs" / a.honest_run / "traj" / f"{backend}__honest__{p:02d}.npz"
            if f.exists():
                hon[p] = np.load(f)["states"]
            else:
                hon[p] = honest(den, den.encode(study["prompts"][p]), unc, study["seed_base"] + p, spec)
        return hon[p]

    log = out / "outputs.jsonl"
    done = {json.loads(l)["prompt"] for l in log.read_text().splitlines()} if log.exists() else set()
    for p in range(n):
        if p in done:
            continue
        seed, tgt_p = study["seed_base"] + p, (p + 1) % n
        base, target = honest_states(p), honest_states(tgt_p)
        f = den.eps_fn(den.encode(study["prompts"][p]), unc, spec.guidance)
        img_h, img_t = den.decode(base[-1]), den.decode(target[-1])
        rec = {"prompt": p, "target": tgt_p, "psnr_target_vs_honest": psnr(img_t, img_h), "attacks": {}}
        if p < a.images:
            from PIL import Image
            Image.fromarray(img_h).save(out / "img" / f"{p:02d}__honest.png")
            Image.fromarray(img_t).save(out / "img" / f"{p:02d}__target.png")
        for name in list(STEER) + list(SPLICE):
            t0 = time.time()
            states = (steer(f, seed, spec, taus, target, STEER[name]) if name in STEER
                      else splice(f, seed, spec, base, target, SPLICE[name]))
            np.savez(out / "traj" / f"{backend}__{name}__{p:02d}.npz", states=states, secs=time.time() - t0)
            img = den.decode(states[-1])
            zN, zh, zt = (x.astype(np.float64) for x in (states[-1], base[-1], target[-1]))
            rec["attacks"][name] = {"psnr_vs_honest": psnr(img, img_h), "psnr_vs_target": psnr(img, img_t),
                                    "latent_progress": float(np.linalg.norm(zN - zh) / np.linalg.norm(zt - zh)),
                                    "secs": round(time.time() - t0, 2)}
            if p < a.images:
                from PIL import Image
                Image.fromarray(img).save(out / "img" / f"{p:02d}__{name}.png")
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("attack", p, {k: round(v["psnr_vs_honest"], 1) for k, v in rec["attacks"].items()}, flush=True)


if __name__ == "__main__":
    main()
