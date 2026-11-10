"""Analyze the overnight study (runs/study/checks.jsonl) with precision-class-aware thresholds.

A job pins a precision class:
  FP32 class: honest workers {cpu-fp32, mps-fp32, webgpu-fp32}, FP32 verifiers only.
  FP16 class: honest worker mps-fp16, any verifier.
Cheats generated on mps-fp16 are FP16-class jobs; cheats on mps-fp32 are FP32-class jobs.

Thresholds are calibrated on cross-device honest pairs only (same-backend re-runs are
bit-identical) and evaluated leave-one-prompt-out, so honest false-accusation estimates never
reuse calibration data:
  max       tau_i = MARGIN x max honest r at step i
  conformal s = r_i / m_i with m_i the median honest r at step i; tau_i = q * m_i where q is
            the ceil((n+1)(1 - alpha/k))-th smallest pooled s (Proposition 1, k = 2)

Detection: cheats are checked on a seeded random subset of steps. For each trajectory with f
flagged steps among c checked, P(detect with k challenges) = 1 - C(c-f, k)/C(c, k), an
unbiased estimate of the detection probability for k uniformly random challenges.
Writes runs/study/summary.json and summary.md.
"""
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
NAMES = (sys.argv[1] if len(sys.argv) > 1 else "study").split("+")  # "a+b": pool the check logs of several runs
RUN = RUNS / "+".join(NAMES)  # output directory (created for combined runs)
MARGIN, ALPHA, KCAL, MIN_N = 1.5, 0.01, 2, 5
GPU_PREFIXES = ("mps-", "cuda-")  # devices that generate the cheats


def cls_of(backend: str) -> str:
    """Precision class from the backend name suffix: fp32, tf32 (CUDA TF32), fp16, bf16."""
    return backend.rsplit("-", 1)[-1]


def honest_ok(rec, cls) -> bool:
    if rec["kind"] != "honest" or cls_of(rec["worker"]) != cls:
        return False
    # FP32 jobs need FP32 verifiers; TF32 jobs accept FP32/TF32 verifiers; FP16 jobs any verifier
    return {"fp32": cls_of(rec["verifier"]) == "fp32",
            "tf32": cls_of(rec["verifier"]) in ("fp32", "tf32")}.get(cls, True)


def det(c: int, f: int, k: int) -> float:
    if k > c:
        return float("nan")
    return 1.0 - math.comb(c - f, k) / math.comb(c, k)


def main():
    warnings.simplefilter("ignore", RuntimeWarning)  # nan entries mark k larger than the steps checked
    raw = [json.loads(l) for n in NAMES for l in (RUNS / n / "checks.jsonl").read_text().splitlines() if l.strip()]
    # Pooled runs can contain the same (verifier, trajectory) twice, e.g. a 48-prompt rerun of the same jobs
    # (CUDA deterministic: bit-identical) or a dense re-check of steps sampled earlier. Merge them, one check per step.
    merged = {}
    for r in raw:
        key = (r["verifier"], Path(r["file"]).name)
        if key not in merged:
            merged[key] = {**r, "checks": list(r["checks"])}
        else:
            have = {c["i"] for c in merged[key]["checks"]}
            merged[key]["checks"] += [c for c in r["checks"] if c["i"] not in have]
    recs = list(merged.values())
    RUN.mkdir(parents=True, exist_ok=True)
    prompts = sorted({r["prompt"] for r in recs})
    out = {"n_records": len(recs), "n_prompts": len(prompts), "classes": {}}
    md = [f"# Study summary ({len(prompts)} prompts, {len(recs)} check records)", ""]
    for cls in ("fp32", "tf32", "fp16", "bf16"):
        hon = [r for r in recs if honest_ok(r, cls)]
        if not hon:
            continue
        xdev = [r for r in hon if r["worker"] != r["verifier"]]  # same-backend re-runs are bit-identical: no drift info
        if not xdev:  # no cross-device honest pair: thresholds are undefined, so no detection numbers
            out["classes"][cls] = {"honest_records": len(hon), "uncalibrated": True}
            md += [f"## {cls.upper()} class", "", "No cross-device honest pairs in these logs: thresholds undefined. "
                   "Combine with a cross-vendor run (e.g. nvidia+cross_nvidia).", ""]
            continue

        def calib(p_out):
            acc, accv = defaultdict(list), defaultdict(lambda: defaultdict(list))
            for r in xdev:
                if r["prompt"] == p_out:
                    continue
                for c in r["checks"]:
                    acc[c["i"]].append(c["ratio"])
                    accv[r["verifier"]][c["i"]].append(c["ratio"])
            tau_max = {i: MARGIN * max(xs) for i, xs in acc.items()}
            # per verifier stack (the protocol's rule): pooled value where that stack has < MIN_N samples
            tau_v = {v: {i: (MARGIN * max(st[i]) if len(st.get(i, [])) >= MIN_N else tau_max[i]) for i in tau_max}
                     for v, st in accv.items()}
            med = {i: max(float(np.median(xs)), 1e-12) for i, xs in acc.items()}
            sc = np.sort([x / med[i] for i, xs in acc.items() for x in xs])
            n = len(sc)
            rank = math.ceil((n + 1) * (1 - ALPHA / KCAL))
            q = sc[rank - 1] if rank <= n else float("inf")
            return {"max": tau_max, "maxV": tau_v, "conformal": {i: q * med[i] for i in med}, "n": n}

        taus = {p: calib(p) for p in prompts}

        def tau_for(rule, p, verifier):
            t = taus[p]
            return t["maxV"].get(verifier, t["max"]) if rule == "maxV" else t[rule]

        n_fold = min(taus[p]["n"] for p in prompts)
        conformal_ok = math.ceil((n_fold + 1) * (1 - ALPHA / KCAL)) <= n_fold
        rules = ["max", "maxV"] + (["conformal"] if conformal_ok else [])
        n_x = sum(len(r["checks"]) for r in xdev)
        res = {"honest_records": len(hon), "honest_steps": sum(len(r["checks"]) for r in hon), "honest_steps_xdev": n_x,
               "calib_steps_min_fold": n_fold, "conformal_defined": conformal_ok, "rules": rules}
        for rule in rules:
            fl = [(r, c) for r in xdev for c in r["checks"] if c["ratio"] > tau_for(rule, r["prompt"], r["verifier"]).get(c["i"], np.inf)]
            res[f"honest_false_flags_{rule}"] = len(fl)
            res[f"honest_false_flag_rate_{rule}"] = len(fl) / n_x
            res[f"honest_false_flag_pairs_{rule}"] = dict(sorted(defaultdict(int, {}).items()))
            for r, c in fl:
                k = f"{r['worker']}->{r['verifier']}"
                res[f"honest_false_flag_pairs_{rule}"][k] = res[f"honest_false_flag_pairs_{rule}"].get(k, 0) + 1
        pairs = defaultdict(list)
        for r in hon:
            pairs[f"{r['worker']}->{r['verifier']}"].extend(c["ratio"] for c in r["checks"])
        res["honest_pairs"] = {k: {"n": len(v), "median": float(np.median(v)), "max": float(np.max(v))} for k, v in sorted(pairs.items())}
        cheats = defaultdict(lambda: defaultdict(list))
        cheats_w = defaultdict(lambda: defaultdict(list))  # same, split by the worker that made the cheat
        for r in recs:
            if r["kind"] == "honest" or not r["worker"].startswith(GPU_PREFIXES) or cls_of(r["worker"]) != cls:
                continue
            if cls == "fp32" and cls_of(r["verifier"]) != "fp32":
                continue
            for rule in rules:
                tau = tau_for(rule, r["prompt"], r["verifier"]) if r["prompt"] in taus else {}
                f = sum(c["ratio"] > tau.get(c["i"], np.inf) for c in r["checks"])
                for d, key in ((cheats, f"{r['kind']}|{r['verifier']}|{rule}"),
                               (cheats_w, f"{r['kind']}|{r['worker']}|{r['verifier']}|{rule}")):
                    d[key]["flag_frac"].append(f / len(r["checks"]))
                    for k in (1, 2, 4, 8):
                        d[key][f"k{k}"].append(det(len(r["checks"]), f, k))
        def summ(d):
            return {key: {"n": len(v["flag_frac"]), "flag_frac": float(np.mean(v["flag_frac"])),
                          **{f"P_k{k}": float(np.nanmean(v[f"k{k}"])) for k in (1, 2, 4, 8)},
                          **{f"P_k{k}_min": float(np.nanmin(v[f"k{k}"])) for k in (1, 2, 4, 8)}}
                    for key, v in sorted(d.items())}
        res["cheats"], res["cheats_by_worker"] = summ(cheats), summ(cheats_w)
        out["classes"][cls] = res
        flags = ", ".join(f"{rule} {res[f'honest_false_flags_{rule}']}/{n_x}" for rule in rules)
        md += [f"## {cls.upper()} class", "",
               f"Honest cross-device steps: {n_x} (plus {res['honest_steps'] - n_x} same-backend re-run steps, all r = 0). "
               f"Leave-one-prompt-out false flags: {flags}"
               + ("" if conformal_ok else f"; conformal n/a (smallest fold has {n_fold} calibration steps; needs >= {math.ceil(KCAL / ALPHA) - 1})"),
               "Rules: max = pooled over verifier stacks; maxV = per verifier stack (protocol rule).", "",
               "| honest pair | n | median r | max r |", "|---|---|---|---|"]
        md += [f"| {k} | {v['n']} | {v['median']:.2e} | {v['max']:.2e} |" for k, v in res["honest_pairs"].items()]
        md += ["", "| cheat / verifier / rule | n | flagged | P k=1 | k=2 | k=4 | k=8 |", "|---|---|---|---|---|---|---|"]
        md += [f"| {k} | {v['n']} | {v['flag_frac']:.2f} | {v['P_k1']:.2f} | {v['P_k2']:.2f} | {v['P_k4']:.2f} | {v['P_k8']:.2f} |"
               for k, v in res["cheats"].items()]
        md.append("")
    secs = defaultdict(list)
    for r in recs:
        secs[r["verifier"]].append(r["secs"] / max(len(r["checks"]), 1))
    out["verify_secs_per_step"] = {k: float(np.median(v)) for k, v in secs.items()}
    md += ["## Verification time per checked step (median, s)", ""] + [f"- {k}: {v:.2f}" for k, v in out["verify_secs_per_step"].items()]
    (RUN / "summary.json").write_text(json.dumps(out, indent=2))
    (RUN / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
