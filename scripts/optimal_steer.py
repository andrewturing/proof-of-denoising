"""Optimised steering: the strongest first-order attack that stays inside every step's tolerance.

scripts/adaptive_attacks.py steers each step toward another prompt's trajectory, a direction chosen without
looking at the model. Here the worker uses the model's gradients. Every committed step is the honest step plus a
perturbation of size c * tau_i * ||Step(z_i) - z_i||, so each check still passes (r_i = c * tau_i on the worker's
own stack); only the direction changes. Perturbations of relative size 1e-4 act linearly on the final latent, so
for a unit vector u the direction at step i that moves <z_N, u> the most is the adjoint
lam_{i+1} = d<z_N, u> / dz_{i+1}, obtained by back-propagating through the remaining sampler steps (one
vector-Jacobian product of the guided network per step).

  opt33 / opt95   u points from the honest final latent to the target's (another prompt's honest image): the most
                  that in-tolerance steering can move the image toward a chosen target.
  max33 / max95   u is found by power iteration (u <- direction of the displacement the attack achieved): the
                  largest change of the final latent that in-tolerance steering can cause, in any direction.

c = 0.33 keeps r_i <= tau_i against any verifier whose drift stays within the calibrated maximum (tau is 1.5 x
that maximum); c = 0.95 relies on the verifier drifting little. Thresholds: the frozen pooled FP32 thresholds
(runs/storybook_fresh/taus_fp32.json), as for the other adaptive attacks. Target: prompt (p+1) mod n.

    python scripts/optimal_steer.py --device mps --prompts 16
Writes runs/adaptive_opt/{spec.json, traj/<backend>__<attack>__<pp>.npz, img/, outputs.jsonl}. The record format
matches scripts/adaptive_attacks.py; cross-device checks run with scripts/cross_verify.py --traj-run adaptive_opt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from podlab.model import Backend, Denoiser
from podlab.sampler import SamplerSpec, ddim_coeffs, initial_noise, step
from podlab.workers import honest

LEVELS = {"33": 0.33, "95": 0.95}
POWER_ITERS = 4  # iteration 1 is the targeted attack; three more refine the direction of largest change


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return x / max(float(np.linalg.norm(x)), 1e-300)


def adjoint(den, embs, guidance, states, spec, u):
    """lam[i] = d<z_N, u> / dz_i for i = 1..N around the trajectory `states` (discrete adjoint of the DDIM sampler).

    Step_i(z) = A_i z + B_i eps(z, t_i), so lam_i = A_i lam_{i+1} + B_i J_eps(z_i)^T lam_{i+1}."""
    n = spec.steps
    lam = [None] * (n + 1)
    lam[n] = unit(u)
    ht = torch.from_numpy(embs).to(den._dev, den._dt)  # (uncond, cond)
    for i in range(n - 1, 0, -1):
        t, a_t, a_prev, _ = ddim_coeffs(i, spec)
        a, b = np.sqrt(a_prev / a_t), np.sqrt(1 - a_prev) - np.sqrt(a_prev) * np.sqrt(1 - a_t) / np.sqrt(a_t)
        scale = float(np.linalg.norm(lam[i + 1]))
        v = torch.from_numpy((lam[i + 1] / scale).astype(np.float32)).to(den._dev, den._dt)
        z = torch.from_numpy(states[i][None]).to(den._dev, den._dt).requires_grad_(True)
        e = den.unet(z.expand(2, -1, -1, -1), torch.full((2,), t, device=den._dev, dtype=torch.long),
                     encoder_hidden_states=ht).sample
        (g,) = torch.autograd.grad(((e[0] + guidance * (e[1] - e[0])) * v).sum(), z)
        lam[i] = a * lam[i + 1] + b * scale * g[0].float().cpu().numpy().astype(np.float64)
    return lam


def steer(f, seed, spec, taus, lam, c):
    """Honest step plus c * tau_i * ||step|| along lam_{i+1}. Returns (states, budget), budget_i = the size used."""
    states, budget = [initial_noise(seed, spec)], []
    for i in range(spec.steps):
        z = states[-1]
        zh = step(f, z, i, spec, seed).astype(np.float64)
        eps = c * taus[i] * float(np.linalg.norm(zh - z.astype(np.float64)))
        budget.append(eps)
        states.append((zh + eps * unit(lam[i + 1])).astype(np.float32))
    return np.stack(states), budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--honest-run", default="study", help="run whose traj/<backend>__honest__pp.npz are reused when present")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--targets-mod", type=int, default=48, help="the target of prompt p is prompt (p+1) mod this")
    ap.add_argument("--taus", default="runs/storybook_fresh/taus_fp32.json")
    ap.add_argument("--images", type=int, default=6, help="save PNGs for the first N prompts")
    ap.add_argument("--out-run", default="adaptive_opt")
    a = ap.parse_args()
    study = json.loads((ROOT / "runs" / "study" / "spec.json").read_text())
    spec = SamplerSpec(**study["spec"])
    taus_doc = json.loads((ROOT / a.taus).read_text())
    taus = {int(i): v for i, v in taus_doc["tau"].items()}
    backend = f"{a.device}-fp32"
    out = ROOT / "runs" / a.out_run
    (out / "traj").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    (out / "spec.json").write_text(json.dumps({**study, "attacks": {f"{k}{l}": c for k in ("opt", "max") for l, c in LEVELS.items()},
                                               "taus": a.taus, "taus_policy": taus_doc.get("policy"),
                                               "target": f"(p+1) mod {a.targets_mod}", "power_iters": POWER_ITERS}, indent=1))
    den = Denoiser(Backend(a.device, "fp32"), load_vae=True)
    den.unet.requires_grad_(False)
    unc = den.encode(study["safety_neg"])
    hon = {}

    def honest_states(p):
        if p not in hon:
            f = ROOT / "runs" / a.honest_run / "traj" / f"{backend}__honest__{p:02d}.npz"
            hon[p] = np.load(f)["states"] if f.exists() else honest(den, den.encode(study["prompts"][p]), unc, study["seed_base"] + p, spec)
        return hon[p]

    log = out / "outputs.jsonl"
    done = {json.loads(l)["prompt"] for l in log.read_text().splitlines()} if log.exists() else set()
    for p in range(a.prompts):
        if p in done:
            continue
        t0 = time.time()
        seed, tgt_p = study["seed_base"] + p, (p + 1) % a.targets_mod
        base, target = honest_states(p), honest_states(tgt_p)
        cond = den.encode(study["prompts"][p])
        embs, f = np.stack([unc, cond]), den.eps_fn(cond, unc, spec.guidance)
        zh, zt = base[-1].astype(np.float64), target[-1].astype(np.float64)
        gap = float(np.linalg.norm(zt - zh))
        img_h, img_t = den.decode(base[-1]), den.decode(target[-1])
        rec = {"prompt": p, "target": tgt_p, "psnr_target_vs_honest": psnr(img_t, img_h), "attacks": {}, "power": []}

        def record(name, states, budget, lam, u):
            np.savez(out / "traj" / f"{backend}__{name}__{p:02d}.npz", states=states)
            img, d = den.decode(states[-1]), states[-1].astype(np.float64) - zh
            rec["attacks"][name] = {
                "psnr_vs_honest": psnr(img, img_h), "psnr_vs_target": psnr(img, img_t),
                "latent_progress": float(np.linalg.norm(d) / gap),             # size of the change, in units of the gap
                "toward_target": float(np.dot(d.ravel(), unit(zt - zh).ravel()) / gap),  # its component toward the target
                "first_order": float(sum(e * np.linalg.norm(lam[i + 1]) for i, e in enumerate(budget)) / gap),  # along u
                "along_u": float(np.dot(d.ravel(), unit(u).ravel()) / gap)}
            if p < a.images:
                from PIL import Image
                Image.fromarray(img).save(out / "img" / f"{p:02d}__{name}.png")
            return d

        u = zt - zh
        for it in range(POWER_ITERS):
            lam = adjoint(den, embs, spec.guidance, base, spec, u)
            states, budget = steer(f, seed, spec, taus, lam, LEVELS["95"])
            last = it == POWER_ITERS - 1
            if it == 0 or last:
                kind = "opt" if it == 0 else "max"
                d = record(f"{kind}95", states, budget, lam, u)
                s33, b33 = steer(f, seed, spec, taus, lam, LEVELS["33"])
                record(f"{kind}33", s33, b33, lam, u)
            else:
                d = states[-1].astype(np.float64) - zh
            rec["power"].append(float(np.linalg.norm(d) / gap))
            u = d  # next: the direction this attack actually moved the final latent
        rec["secs"] = round(time.time() - t0, 1)
        if p < a.images:
            from PIL import Image
            Image.fromarray(img_h).save(out / "img" / f"{p:02d}__honest.png")
            Image.fromarray(img_t).save(out / "img" / f"{p:02d}__target.png")
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("attack", p, f"{rec['secs']}s", {k: (round(100 * v["latent_progress"], 2), round(v["psnr_vs_honest"], 1)) for k, v in rec["attacks"].items()},
              "power", [round(100 * x, 2) for x in rec["power"]], flush=True)


if __name__ == "__main__":
    main()
