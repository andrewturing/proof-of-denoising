"""Two-tier verification on held-out honest jobs: the light verifier (the reader's browser) checks; any step
over its threshold escalates to a reference FP32 verifier (NVIDIA L4) that re-checks the same step. A final
false accusation needs both tiers to exceed tau.

Thresholds: FP32 class, pooled over every stack, tau_i = 1.5 x max honest cross-device r_i,
leave-one-prompt-out (the same rule as analyze_study.py "max").

    python scripts/escalation.py [POOL] [--light webgpu-fp32] [--ref cuda-fp32]
Writes runs/escalation_<light>_to_<ref>.json.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
MARGIN = 1.5


def cls(b):
    return b.rsplit("-", 1)[-1]


def load(names):
    merged = {}
    for n in names:
        f = RUNS / n / "checks.jsonl"
        if not f.exists():
            continue
        for r in map(json.loads, f.read_text().splitlines()):
            key = (r["verifier"], Path(r["file"]).name)
            if key not in merged:
                merged[key] = {**r, "checks": list(r["checks"])}
            else:
                have = {c["i"] for c in merged[key]["checks"]}
                merged[key]["checks"] += [c for c in r["checks"] if c["i"] not in have]
    return list(merged.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", nargs="?", default="study+calib_webgpu-fp32+nvidia+cross_nvidia+cross_mac+arbiter_cuda")
    ap.add_argument("--light", default="webgpu-fp32")
    ap.add_argument("--ref", default="cuda-fp32")
    a = ap.parse_args()
    recs = load(a.pool.split("+"))
    hon = [r for r in recs if r["kind"] == "honest" and cls(r["worker"]) == "fp32" and cls(r["verifier"]) == "fp32"
           and r["worker"] != r["verifier"]]
    prompts = sorted({r["prompt"] for r in hon})
    taus = {}
    for p in prompts:
        acc = defaultdict(list)
        for r in hon:
            if r["prompt"] != p:
                for c in r["checks"]:
                    acc[c["i"]].append(c["ratio"])
        taus[p] = {i: MARGIN * max(v) for i, v in acc.items()}
    ref = {(Path(r["file"]).name, c["i"]): c["ratio"] for r in recs if r["verifier"] == a.ref for c in r["checks"]}
    light = [r for r in hon if r["verifier"] == a.light]
    n_checks = sum(len(r["checks"]) for r in light)
    esc, confirmed, missing = [], 0, 0
    for r in light:
        for c in r["checks"]:
            if c["ratio"] > taus[r["prompt"]][c["i"]]:
                rr = ref.get((Path(r["file"]).name, c["i"]))
                over_ref = None if rr is None else rr / taus[r["prompt"]][c["i"]]
                esc.append({"file": Path(r["file"]).name, "prompt": r["prompt"], "step": c["i"],
                            "light_over": c["ratio"] / taus[r["prompt"]][c["i"]], "ref_over": over_ref})
                if rr is None:
                    missing += 1
                elif over_ref > 1:
                    confirmed += 1
    out = {"pool": a.pool, "light": a.light, "ref": a.ref, "light_honest_checks": n_checks,
           "light_honest_jobs": len(light), "prompts": len(prompts), "escalated": len(esc), "confirmed": confirmed,
           "no_reference_check": missing,
           "escalated_prompts": sorted({e["prompt"] for e in esc}),
           "ref_over_max": max((e["ref_over"] for e in esc if e["ref_over"] is not None), default=None),
           "escalations": esc}
    (RUNS / f"escalation_{a.light}_to_{a.ref}.json").write_text(json.dumps(out, indent=1))
    print(f"{a.light}: {n_checks} held-out honest checks on {len(light)} jobs ({len(prompts)} prompts); "
          f"{len(esc)} over tau -> escalated to {a.ref}: {confirmed} confirmed, {missing} without a reference check; "
          f"max reference ratio {out['ref_over_max']}")


if __name__ == "__main__":
    main()
