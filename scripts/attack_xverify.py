"""Cross-device check of the full-cost attack trajectories (rendered on the NVIDIA L4 by scripts/adaptive_attacks.py,
recomputed on all 25 steps by another FP32 stack, e.g. the x86 CPU: runs/xadaptive_cpu). Counts steps over the frozen
pooled thresholds (runs/storybook_fresh/taus_fp32.json) per attack and prompt.

    python scripts/attack_xverify.py [--run xadaptive_cpu]
Writes runs/adaptive/xverify_summary.json (read by make_attack_fig.py, build_poster.py and export_macros_cvpr.py).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="xadaptive_cpu")
    ap.add_argument("--out", default="adaptive/xverify_summary.json", help="summary file under runs/")
    a = ap.parse_args()
    taus = {int(i): v for i, v in json.loads((RUNS / "storybook_fresh" / "taus_fp32.json").read_text())["tau"].items()}
    per = defaultdict(dict)
    verifier = None
    for r in map(json.loads, (RUNS / a.run / "checks.jsonl").read_text().splitlines()):
        verifier = r["verifier"]
        over = [c["i"] for c in r["checks"] if c["ratio"] > taus[c["i"]]]
        per[r["kind"]][str(r["prompt"])] = {"over": len(over), "steps": len(r["checks"]), "over_steps": over,
                                            "max_over": max(c["ratio"] / taus[c["i"]] for c in r["checks"]), "verifier": verifier}
    summary = {k: {"trajectories": len(v), "steps_over": sum(x["over"] for x in v.values()),
                   "steps": sum(x["steps"] for x in v.values()),
                   "max_over_tau": max(x["max_over"] for x in v.values())} for k, v in per.items()}
    out = {"run": a.run, "verifier": verifier, "summary": summary, **per}
    (RUNS / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
