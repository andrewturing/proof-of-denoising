"""Request-level split-conformal thresholds (podlab/audit_policy.py), the calibration the review asked for.

One score per independent honest REQUEST (never per step): S_j = max_i r_ji / a_i over all 25 steps, with per-step
scales a_i fitted on a disjoint split. With n calibration requests, tau_i = q * a_i where q is the
ceil((n + 1)(1 - alpha))-th smallest score; under exchangeability of honest requests, an honest request is flagged on
ANY of its steps with probability <= alpha, so every subset of checked steps inherits the bound.

Calibration requests: runs/calib240 (240 PartiPrompts disjoint from the study's 48; rendered on the NVIDIA L4, FP32)
checked on all 25 steps by the Apple GPU (FP32), runs/xcal240_mps. Split: the first FIT prompts fit the scales.
Test requests (never used above): every honest FP32 cross-device job in the study pool checked on all 25 steps,
grouped by (worker, verifier); stacks the calibration never saw (x86 CPU, browser) are stress tests.
Power: the deviating trajectories of the study pool against the same thresholds.

    python scripts/conformal_requests.py [--alpha 0.01] [--fit 40]
Writes runs/conformal_requests.json.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from podlab.audit_policy import DriftGroup, calibrate_request_max  # noqa: E402

N = 25
TEST_RUNS = ["study", "calib_webgpu-fp32", "cross_mac", "nvidia48", "cross_nvidia", "arbiter_cuda", "xcal_cuda_fp32",
             "xcpu_nvidia48", "dense_nvidia48"]


def cls(b):
    return b.rsplit("-", 1)[-1]


def full_records(runs):
    """Merge records by (verifier, file); keep those with all N steps."""
    merged = {}
    for run in runs:
        f = ROOT / "runs" / run / "checks.jsonl"
        if not f.exists():
            continue
        for r in map(json.loads, f.read_text().splitlines()):
            key = (r["verifier"], Path(r["file"]).name, r.get("traj_run", run))
            m = merged.setdefault(key, {**r, "steps": {}})
            for c in r["checks"]:
                m["steps"][c["i"]] = c["ratio"]
    return [m for m in merged.values() if len(m["steps"]) == N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--fit", type=int, default=40)
    a = ap.parse_args()
    cal = [r for r in full_records(["xcal240_mps"]) if r["kind"] == "honest"]
    cal.sort(key=lambda r: r["prompt"])
    fit = [DriftGroup(f"calib240/{r['prompt']}", tuple(r["steps"][i] for i in range(N))) for r in cal if r["prompt"] < a.fit]
    cg = [DriftGroup(f"calib240/{r['prompt']}", tuple(r["steps"][i] for i in range(N))) for r in cal if r["prompt"] >= a.fit]
    out = {"alpha": a.alpha, "fit_requests": len(fit), "calibration_requests": len(cg),
           "calibration": "runs/calib240 (NVIDIA L4 FP32 render) checked by the Apple GPU FP32 on all 25 steps"}
    if not fit or not cg:
        out["status"] = "no calibration data yet"
        print(out)
        return
    c = calibrate_request_max(cg, steps=N, alpha=a.alpha, scale_fit=fit)
    out.update(rank=c.rank, usable=c.usable)
    if not c.usable:
        out["status"] = f"inconclusive: rank {c.rank} > n = {c.n_groups}; alpha = {a.alpha} needs >= {math.ceil(1 / a.alpha) - 1} groups"
        (ROOT / "runs" / "conformal_requests.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
    tau = c.thresholds
    out["tau"] = list(tau)
    test = [r for r in full_records(TEST_RUNS) if r["kind"] == "honest" and cls(r["worker"]) == "fp32"
            and cls(r["verifier"]) == "fp32" and r["worker"] != r["verifier"]]
    by = defaultdict(list)
    for r in test:
        by[f"{r['worker']}->{r['verifier']}"].append(any(r["steps"][i] > tau[i] for i in range(N)))
    out["test_requests_flagged"] = {k: {"flagged": sum(v), "requests": len(v)} for k, v in sorted(by.items())}
    seen = {"mps-fp32", "cuda-fp32"}
    native = [x for k, v in by.items() if set(k.split("->")) <= seen for x in v]
    out["test_same_stacks"] = {"flagged": sum(native), "requests": len(native)}
    # power: deviating trajectories (any number of checked steps), pooled thresholds above
    power = defaultdict(list)
    for run in TEST_RUNS:
        f = ROOT / "runs" / run / "checks.jsonl"
        if f.exists():
            for r in map(json.loads, f.read_text().splitlines()):
                if r["kind"] != "honest" and cls(r["worker"]) == "fp32" and cls(r["verifier"]) == "fp32":
                    st = {x["i"]: x["ratio"] for x in r["checks"]}
                    cnt, fl = len(st), sum(v > tau[i] for i, v in st.items())
                    if cnt >= 2:
                        power[r["kind"]].append(1 - math.comb(cnt - fl, 2) / math.comb(cnt, 2))
    out["power_k2_mean"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(power.items())}
    out["caught_k2_ge_099"] = sum(v >= 0.99 for v in out["power_k2_mean"].values())
    (ROOT / "runs" / "conformal_requests.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "tau"}, indent=1))


if __name__ == "__main__":
    main()
