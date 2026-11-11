"""The nine shortcuts on 240 further prompts (scripts/shortcut_scale.py -> runs/scale240), judged with the frozen
pooled FP32 thresholds (runs/storybook_fresh/taus_fp32.json). Every step of every shortcut trajectory was recomputed
on the GPU that rendered it, so a step the shortcut did not change is bit-identical (r = 0) and a changed step has
r > 0. For each shortcut the report gives the share of steps over the threshold, the exact two-check detection
probability per trajectory, and the smallest margin r / tau over the steps it changed.

    python scripts/scale_report.py [run]
Writes runs/<run>_report.json (read by export_macros_cvpr.py).
"""
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "scale240"
    taus = {int(i): v for i, v in json.loads((RUNS / "storybook_fresh" / "taus_fp32.json").read_text())["tau"].items()}
    recs = [json.loads(l) for l in (RUNS / run / "checks.jsonl").read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in recs:
        by[r["kind"]].append(r)
    out = {"run": run, "prompts": len({r["prompt"] for r in recs}), "taus": "runs/storybook_fresh/taus_fp32.json", "shortcuts": {}}
    honest = [c["ratio"] for r in by.pop("honest", []) for c in r["checks"]]
    out["honest_same_device"] = {"steps": len(honest), "nonzero": int(sum(x != 0 for x in honest))}
    tot_traj = tot_changed = tot_missed = 0
    margin = np.inf
    for kind, rs in sorted(by.items()):
        frac, pk2, changed, missed, lo = [], [], 0, 0, np.inf
        for r in rs:
            over = np.array([c["ratio"] / taus[c["i"]] for c in r["checks"]])
            ch = np.array([c["ratio"] > 0 for c in r["checks"]])
            f, n = int((over > 1).sum()), len(over)
            frac.append(f / n)
            pk2.append(1.0 - comb(n - f, 2) / comb(n, 2))
            changed += int(ch.sum())
            missed += int((ch & (over <= 1)).sum())
            if ch.any():
                lo = min(lo, float(over[ch].min()))
        out["shortcuts"][kind] = {"trajectories": len(rs), "steps_per_trajectory": len(rs[0]["checks"]), "flag_frac": float(np.mean(frac)),
                                  "P_k2": float(np.mean(pk2)), "P_k2_min": float(min(pk2)), "below_099": int(sum(p < 0.99 for p in pk2)),
                                  "changed_steps": changed, "changed_not_flagged": missed, "min_over_changed": lo}
        tot_traj, tot_changed, tot_missed, margin = tot_traj + len(rs), tot_changed + changed, tot_missed + missed, min(margin, lo)
    out.update(trajectories=tot_traj, changed_steps=tot_changed, changed_not_flagged=tot_missed, min_over_changed=margin,
               caught_k2=sum(v["P_k2"] >= 0.99 for v in out["shortcuts"].values()))
    (RUNS / f"{run}_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "shortcuts"}, indent=1))
    for k, v in out["shortcuts"].items():
        print(f"{k:18s} n={v['trajectories']:4d} flagged steps {100 * v['flag_frac']:5.1f}%  P(k=2) mean {v['P_k2']:.3f} min {v['P_k2_min']:.3f}"
              f"  changed-but-passed {v['changed_not_flagged']}/{v['changed_steps']}  min r/tau on changed steps {v['min_over_changed']:.1f}")


if __name__ == "__main__":
    main()
