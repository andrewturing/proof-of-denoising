"""Are shortcuts detectable exactly where they cut? Uses trajectories recomputed on ALL 25 steps by the worker's own
stack (NVIDIA L4, FP32), so honestly computed steps reproduce bit-exactly and every non-zero r comes from the shortcut.

For each deviation, the steps it changes are known from its definition (podlab/workers.py), not inferred from flags:
  every-step deviations (substitution, W8/W4, CFG skip, safety strip): all 25 steps
  cache (period 2 from step 0): steps that reuse a prediction, i = 1, 3, ..., 23
  late cache (period 2 from step 12): i = 13, 15, ..., 23
  step skip (two-step jumps with interpolated states): i = 0 .. 23
Reports the smallest r / tau over changed steps and the largest r / tau over unchanged steps, against the frozen
pooled thresholds (runs/storybook_fresh/taus_fp32.json).

    python scripts/dense_margin.py
Writes runs/dense_margin.json.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
N = 25
EVERY = {"sub-bk-sdm-small", "sub-bk-sdm-tiny", "w8", "w4", "cfg-skip", "neg-strip"}


def changed(kind):
    if kind in EVERY:
        return set(range(N))
    if kind == "cache2":
        return {i for i in range(N) if i % 2 == 1}
    if kind == "cache2-late":
        return {i for i in range(N) if i >= 12 and (i - 12) % 2 == 1}
    if kind == "step-skip":
        return set(range(N - 1))
    raise KeyError(kind)


def main():
    taus = {int(i): v for i, v in json.loads((RUNS / "storybook_fresh" / "taus_fp32.json").read_text())["tau"].items()}
    steps = defaultdict(dict)
    for run in ("nvidia48", "dense_nvidia48"):
        f = RUNS / run / "checks.jsonl"
        if f.exists():
            for r in map(json.loads, f.read_text().splitlines()):
                if r["worker"] == "cuda-fp32" and r["verifier"] == "cuda-fp32" and r["kind"] != "honest":
                    for c in r["checks"]:
                        steps[(r["kind"], r["prompt"])][c["i"]] = c["ratio"]
    full = {k: v for k, v in steps.items() if len(v) == N}
    per = defaultdict(lambda: {"trajectories": 0, "min_over_changed": float("inf"), "max_over_unchanged": 0.0,
                               "changed_not_flagged": 0, "unchanged_flagged": 0})
    for (kind, p), st in full.items():
        ch = changed(kind)
        d = per[kind]
        d["trajectories"] += 1
        for i, r in st.items():
            o = r / taus[i]
            if i in ch:
                d["min_over_changed"] = min(d["min_over_changed"], o)
                d["changed_not_flagged"] += o <= 1
            else:
                d["max_over_unchanged"] = max(d["max_over_unchanged"], o)
                d["unchanged_flagged"] += o > 1
    out = {"thresholds": "runs/storybook_fresh/taus_fp32.json", "trajectories": len(full), "by_kind": dict(per),
           "min_over_detectable": min((d["min_over_changed"] for d in per.values()), default=None),
           "changed_not_flagged": sum(d["changed_not_flagged"] for d in per.values()),
           "unchanged_flagged": sum(d["unchanged_flagged"] for d in per.values())}
    (RUNS / "dense_margin.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "by_kind"}, indent=1))
    for k, d in sorted(per.items()):
        print(f"{k:18s} n={d['trajectories']:3d} min r/tau changed={d['min_over_changed']:.1f} max unchanged={d['max_over_unchanged']:.3f} "
              f"misses={d['changed_not_flagged']} false={d['unchanged_flagged']}")


if __name__ == "__main__":
    main()
