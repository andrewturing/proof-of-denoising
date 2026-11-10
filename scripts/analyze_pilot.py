"""Summarize the pilot: honest per-step drift, calibrated thresholds, cheat detection, and
output-level drift. Writes runs/pilot/summary.json and runs/pilot/summary.md.

Thresholds are calibrated by leave-one-prompt-out: for each held-out prompt, tau_i is the
max honest ratio at step i over all other prompts and backend pairs (times a margin).
That gives an honest false-accusation estimate that does not reuse the calibration data.
"""
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

RUN = ROOT / "runs" / "pilot"
MARGIN = 1.5  # tau = MARGIN x max honest ratio seen in calibration


def load():
    recs = [json.loads(l) for l in (RUN / "checks.jsonl").read_text().splitlines() if l.strip()]
    return recs


def ratios(rec) -> dict[int, float]:
    return {c["i"]: c["ratio"] for c in rec["checks"]}


def det_prob(flagged: int, total: int, k: int) -> float:
    k = min(k, total)
    if total - flagged < k:
        return 1.0
    return 1 - comb(total - flagged, k) / comb(total, k)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    recs = load()
    honest = [r for r in recs if r["kind"] == "honest"]
    cheats = [r for r in recs if r["kind"] != "honest"]
    out: dict = {"n_records": len(recs)}

    # 1. Honest drift by (worker, verifier) pair.
    pair = defaultdict(list)
    for r in honest:
        pair[(r["worker"], r["verifier"])].extend(ratios(r).values())
    out["honest_pairs"] = {f"{w}->{v}": {"n": len(x), "median": float(np.median(x)), "p99": float(np.quantile(x, 0.99)),
                                        "max": float(np.max(x))} for (w, v), x in sorted(pair.items())}

    # 2. Leave-one-prompt-out thresholds and honest false-accusation rate.
    by_prompt_step = defaultdict(lambda: defaultdict(list))  # prompt -> step -> ratios
    for r in honest:
        for i, x in ratios(r).items():
            by_prompt_step[r["prompt"]][i].append(x)
    prompts = sorted(by_prompt_step)

    def tau_without(p_out: int) -> dict[int, float]:
        acc = defaultdict(list)
        for p in prompts:
            if p != p_out:
                for i, xs in by_prompt_step[p].items():
                    acc[i].extend(xs)
        return {i: MARGIN * max(xs) for i, xs in acc.items()}

    taus = {p: tau_without(p) for p in prompts}
    fa = [x > taus[r["prompt"]].get(i, np.inf) for r in honest for i, x in ratios(r).items()]
    out["honest_step_false_accusation_rate"] = float(np.mean(fa)) if fa else None
    out["honest_steps_checked"] = len(fa)
    tau_all = defaultdict(list)
    for p in prompts:
        for i, xs in by_prompt_step[p].items():
            tau_all[i].extend(xs)
    out["tau_by_step"] = {i: MARGIN * max(xs) for i, xs in sorted(tau_all.items())}

    # 3. Cheat detection per (cheat, verifier).
    cs = defaultdict(lambda: {"flag_frac": [], "sep": [], "p_k": defaultdict(list)})
    for r in cheats:
        tau = taus.get(r["prompt"], {})
        rs = ratios(r)
        flagged = [i for i, x in rs.items() if x > tau.get(i, np.inf)]
        key = f"{r['kind']}@{r['verifier']}"
        cs[key]["flag_frac"].append(len(flagged) / len(rs))
        cs[key]["sep"].append(max(x / (tau.get(i, np.inf) / MARGIN) for i, x in rs.items()))
        for k in (1, 2, 4):
            cs[key]["p_k"][k].append(det_prob(len(flagged), len(rs), k))
    out["cheats"] = {key: {"flagged_step_frac_mean": float(np.mean(v["flag_frac"])),
                           "max_ratio_over_honest_max_median": float(np.median(v["sep"])),
                           **{f"P_detect_k{k}_mean": float(np.mean(v["p_k"][k])) for k in (1, 2, 4)},
                           **{f"P_detect_k{k}_min": float(np.min(v["p_k"][k])) for k in (1, 2, 4)},
                           "n": len(v["flag_frac"])} for key, v in sorted(cs.items())}

    # 4. Output-level drift (images decoded on one fixed backend).
    img_dir = RUN / "img"
    imgs = {p.stem: np.asarray(Image.open(p)) for p in img_dir.glob("*.png")} if img_dir.exists() else {}
    workers = sorted({k.split("__")[0] for k in imgs if "__honest__" in k})
    hon_pairs = defaultdict(list)
    for p in prompts:
        for a in workers:
            for b in workers:
                ka, kb = f"{a}__honest__{p:02d}", f"{b}__honest__{p:02d}"
                if a < b and ka in imgs and kb in imgs:
                    hon_pairs[f"{a}|{b}"].append(psnr(imgs[ka], imgs[kb]))
    out["output_psnr_honest_pairs"] = {k: {"median": float(np.median(v)), "min": float(np.min(v))} for k, v in hon_pairs.items()}
    cheat_psnr = defaultdict(list)
    for k, im in imgs.items():
        w, kind, p = k.split("__")
        ref = imgs.get(f"{w}__honest__{p}")
        if kind != "honest" and ref is not None:
            cheat_psnr[kind].append(psnr(im, ref))
    out["output_psnr_cheat_vs_honest"] = {k: {"median": float(np.median(v)), "max": float(np.max(v))} for k, v in sorted(cheat_psnr.items())}

    (RUN / "summary.json").write_text(json.dumps(out, indent=2))
    lines = ["# Pilot summary", "", f"Records: {out['n_records']}; honest steps checked: {out['honest_steps_checked']}; "
             f"honest step false-accusation rate (leave-one-prompt-out, margin {MARGIN}x): {out['honest_step_false_accusation_rate']}", "",
             "## Honest per-step drift ratio by worker -> verifier", "", "| pair | n | median | p99 | max |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['median']:.2e} | {v['p99']:.2e} | {v['max']:.2e} |" for k, v in out["honest_pairs"].items()]
    lines += ["", "## Cheats", "", "| cheat@verifier | n | flagged steps | max ratio / honest max (median) | P(detect) k=1 | k=2 | k=4 (min) |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['flagged_step_frac_mean']:.2f} | {v['max_ratio_over_honest_max_median']:.1f} | "
              f"{v['P_detect_k1_mean']:.2f} | {v['P_detect_k2_mean']:.2f} | {v['P_detect_k4_min']:.2f} |" for k, v in out["cheats"].items()]
    lines += ["", "## Output-level PSNR (dB)", "", "| comparison | median | worst |", "|---|---|---|"]
    lines += [f"| honest {k} | {v['median']:.1f} | min {v['min']:.1f} |" for k, v in out["output_psnr_honest_pairs"].items()]
    lines += [f"| {k} vs honest | {v['median']:.1f} | max {v['max']:.1f} |" for k, v in out["output_psnr_cheat_vs_honest"].items()]
    (RUN / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
