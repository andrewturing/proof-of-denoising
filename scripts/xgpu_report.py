"""Further NVIDIA GPUs as workers and verifiers, judged with thresholds frozen before they joined.

gtc_kit/run_xgpu_vm.sh ran the SD1.5 job on other NVIDIA GPUs (tags such as t4, a100): each rendered the 48 honest
jobs and re-checked the L4's honest, lower-precision and shortcut jobs. Check logs live in runs/xg_<verifier>_on_<worker>
and runs/xg_<tag>_self; a verifier or worker is named <tag>-<precision> (the L4 keeps its study name, cuda-*).

Every check is judged with the frozen pooled FP32 thresholds (runs/storybook_fresh/taus_fp32.json, written on
2026-11-09 from the Apple, L4, x86 and browser logs), so the new GPUs are held-out hardware: nothing is recalibrated.

    python scripts/xgpu_report.py
Writes runs/xgpu_report.json (read by export_macros_cvpr.py) and prints a summary.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
SHORTCUTS = ["sub-bk-sdm-small", "sub-bk-sdm-tiny", "w8", "w4", "cfg-skip", "cache2", "cache2-late", "step-skip", "neg-strip"]


def det(c: int, f: int, k: int) -> float:
    from math import comb
    return float("nan") if k > c else 1.0 - comb(c - f, k) / comb(c, k)


def cls(name: str) -> str:
    return name.rsplit("-", 1)[-1]


def gpu(name: str) -> str:
    return name.rsplit("-", 1)[0]


def main():
    taus = {int(i): v for i, v in json.loads((RUNS / "storybook_fresh" / "taus_fp32.json").read_text())["tau"].items()}
    recs = []
    for d in sorted(RUNS.glob("xg_*")):
        f = d / "checks.jsonl"
        if f.exists():
            recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not recs:
        sys.exit("no runs/xg_*/checks.jsonl")
    out = {"taus": "runs/storybook_fresh/taus_fp32.json", "gpus": {}, "honest_fp32": {}, "precision": {}, "shortcuts": {}}
    for d in sorted(x for x in RUNS.glob("xgpu_*") if x.is_dir()):
        tag = d.name[5:]
        g = {}
        if (d / "gpu.txt").exists():
            g["name"] = (d / "gpu.txt").read_text().splitlines()[1].split(",")[0].strip()
        if (d / "torch_env.txt").exists():
            g["software"] = (d / "torch_env.txt").read_text().strip()
        if (d / "bench.json").exists():
            g["step_s"] = {k: v["step_s"] for k, v in json.loads((d / "bench.json").read_text()).items() if isinstance(v, dict) and "step_s" in v}
        out["gpus"][tag] = g

    over = lambda c: c["ratio"] / taus[c["i"]]
    # honest FP32 jobs checked by a different FP32 GPU (held-out pairs), and same-GPU re-runs
    pairs = defaultdict(list)
    for r in recs:
        if r["kind"] == "honest" and cls(r["worker"]) == "fp32" and cls(r["verifier"]) == "fp32":
            pairs[f"{r['worker']}->{r['verifier']}"] += [(c["ratio"], over(c), r["prompt"]) for c in r["checks"]]
    tot = fl = 0
    worst = 0.0
    for k, v in sorted(pairs.items()):
        ra, ov = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        same = gpu(k.split("->")[0]) == gpu(k.split("->")[1])
        out["honest_fp32"][k] = {"n": len(v), "prompts": len({x[2] for x in v}), "median": float(np.median(ra)), "max": float(ra.max()),
                                 "flagged": int((ov > 1).sum()), "max_over_tau": float(ov.max()), "same_gpu": same}
        if not same:
            tot, fl, worst = tot + len(v), fl + int((ov > 1).sum()), max(worst, float(ov.max()))
    out["honest_fp32_cross"] = {"steps": tot, "flagged": fl, "max_over_tau": worst,
                                "max": max((v["max"] for v in out["honest_fp32"].values() if not v["same_gpu"]), default=None),
                                "median_of_medians": float(np.median([v["median"] for v in out["honest_fp32"].values() if not v["same_gpu"]])) if tot else None}
    # lower-precision honest jobs judged as FP32 jobs by an FP32 verifier
    low = defaultdict(list)
    for r in recs:
        if r["kind"] == "honest" and cls(r["worker"]) != "fp32" and cls(r["verifier"]) == "fp32":
            low[f"{r['worker']}->{r['verifier']}"] += [(c["ratio"], over(c)) for c in r["checks"]]
    for k, v in sorted(low.items()):
        ra, ov = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        out["precision"][k] = {"n": len(v), "median": float(np.median(ra)), "min": float(ra.min()), "flagged": int((ov > 1).sum()),
                               "bit_identical": bool(ra.max() == 0)}
    # the L4's shortcut jobs checked by the new GPUs
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["kind"] in SHORTCUTS and cls(r["verifier"]) == "fp32":
            c, f = len(r["checks"]), sum(over(x) > 1 for x in r["checks"])
            by[r["verifier"]][r["kind"]].append((f / c, det(c, f, 2)))
    for ver, d in sorted(by.items()):
        out["shortcuts"][ver] = {k: {"n": len(v), "flag_frac": float(np.mean([x[0] for x in v])), "P_k2": float(np.mean([x[1] for x in v]))}
                                 for k, v in d.items()}
        out["shortcuts"][ver]["caught_k2"] = sum(v["P_k2"] >= 0.99 for k, v in out["shortcuts"][ver].items() if isinstance(v, dict))
    (RUNS / "xgpu_report.json").write_text(json.dumps(out, indent=1))
    for tag, g in out["gpus"].items():
        print(tag, g.get("name"), g.get("step_s"), "|", g.get("software"))
    for k, v in out["honest_fp32"].items():
        print(f"honest FP32 {k:28s} n={v['n']:5d} median {v['median']:.1e} max {v['max']:.1e} flagged {v['flagged']} max r/tau {v['max_over_tau']:.2f}")
    print("cross-GPU honest FP32:", out["honest_fp32_cross"])
    for k, v in out["precision"].items():
        print(f"precision {k:28s} n={v['n']:5d} median {v['median']:.1e} min {v['min']:.1e} flagged {v['flagged']}/{v['n']}" + (" (bit-identical to FP32)" if v["bit_identical"] else ""))
    for ver, d in out["shortcuts"].items():
        print("shortcuts checked by", ver, {k: round(v["P_k2"], 3) for k, v in d.items() if isinstance(v, dict)}, "caught", d["caught_k2"])


if __name__ == "__main__":
    main()
