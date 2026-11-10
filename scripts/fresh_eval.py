"""Outcomes of the pre-registered fresh storybook evaluation (runs/storybook_fresh/PREREG.json).

Reads only logs written after the thresholds were frozen:
  verify_cuda-fp32.all.jsonl          reference (L4, FP32) ratios on all 25 steps of every page
  verify_webgpu-fp32.jsonl            tier 1: the browser's k = 2 random checks per page
  escalate_webgpu-fp32_to_cuda-fp32.jsonl   tier 2: exactly the flagged steps, re-checked on the reference
Checks first that PREREG.json, the thresholds and the code still match the hashes recorded before rendering.

    python scripts/fresh_eval.py
Writes runs/storybook_fresh/result.json.
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs" / "storybook_fresh"
N = 25


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pre = json.loads((R / "PREREG.json").read_text())
    assert sha(R / "taus_fp32.json") == pre["thresholds"]["taus_fp32_json_sha256"], "thresholds changed after pre-registration"
    assert sha(R / "book.json") == pre["book"]["book_json_sha256"], "book changed after pre-registration"
    for s, h in pre["thresholds"]["sources"].items():
        assert sha(ROOT / s) == h, f"calibration source changed: {s}"
    code_changed = sorted(f for f, h in pre["code_sha256"].items() if sha(ROOT / f) != h)
    render = json.loads((R / "render.json").read_text())
    taus = {int(i): v for i, v in json.loads((R / "taus_fp32.json").read_text())["tau"].items()}
    kind = lambda page, node: render["pages"][page][node]

    ref = {(r["page"], r["node"]): r for r in map(json.loads, (R / "verify_cuda-fp32.all.jsonl").read_text().splitlines())}
    light = {(r["page"], r["node"]): r for r in map(json.loads, (R / "verify_webgpu-fp32.jsonl").read_text().splitlines())}
    ef = R / "escalate_webgpu-fp32_to_cuda-fp32.jsonl"
    esc = {(e["page"], e["node"]): e for e in map(json.loads, ef.read_text().splitlines())} if ef.exists() else {}

    pages, by_kind = [], defaultdict(list)
    for (page, node), r in sorted(ref.items()):
        k = kind(page, node)
        over = [c["ratio"] / taus[c["i"]] for c in sorted(r["checks"], key=lambda c: c["i"])]
        m = sum(o > 1 for o in over)
        lt = light.get((page, node))
        e = esc.get((page, node))
        final = e["final_accepted"] if e else (lt["accepted"] if lt else None)
        rec = {"page": page, "node": node, "kind": k, "m_detectable": m,
               "P_detect": {kk: 1 - math.comb(N - m, kk) / math.comb(N, kk) for kk in (1, 2, 4, 8)},
               "max_over_honest_steps" if k == "honest" else "min_over_detectable": (max(over) if k == "honest" else
                                                                                    min((o for o in over if o > 1), default=None)),
               "flag_mask": "".join("x" if o > 1 else "." for o in over),
               "tier1": None if lt is None else {"challenged": lt["challenged"], "accepted": lt["accepted"], "secs": lt["secs"],
                                                 "max_over": max(c["over"] for c in lt["checks"])},
               "tier2": None if not e or not e.get("escalated_steps") else {"steps": e["escalated_steps"], "accepted": e["final_accepted"]},
               "final_accepted": final, "ref_check_secs": r["secs"]}
        pages.append(rec)
        by_kind[k].append(rec)
    honest = by_kind.pop("honest")
    dev = [p for v in by_kind.values() for p in v]
    done = [p for p in pages if p["final_accepted"] is not None]
    out = {
        "prereg_sha256": sha(R / "PREREG.json"), "code_changed_since_prereg": code_changed,
        "honest_pages": len(honest), "deviating_pages": len(dev),
        "tier1_done": sum(p["tier1"] is not None for p in pages),
        "honest_accepted_final": sum(p["final_accepted"] is True for p in honest),
        "honest_flagged_tier1": sum(p["tier1"] is not None and not p["tier1"]["accepted"] for p in honest),
        "deviating_rejected_final": sum(p["final_accepted"] is False for p in dev),
        "deviating_flagged_tier1": sum(p["tier1"] is not None and not p["tier1"]["accepted"] for p in dev),
        "deviating_cleared_by_tier2": sum(p["tier2"] is not None and p["tier2"]["accepted"] for p in dev),
        "missed": sorted((p["kind"], p["page"]) for p in dev if p["final_accepted"] is True),
        "honest_steps_over_tau_reference": sum(p["m_detectable"] for p in honest),
        "honest_max_over_reference": max(p["max_over_honest_steps"] for p in honest),
        "detectable_min_over_reference": min(p["min_over_detectable"] for p in dev if p["min_over_detectable"] is not None),
        "expected_rejected_k2": sum(p["P_detect"][2] for p in dev),
        "by_kind": {k: {"pages": len(v), "m": [p["m_detectable"] for p in v], "P_k2": [round(p["P_detect"][2], 3) for p in v],
                        "rejected": sum(p["final_accepted"] is False for p in v)} for k, v in sorted(by_kind.items())},
        "median_secs": {"browser_check": sorted(p["tier1"]["secs"] for p in pages if p["tier1"])[len(done) // 2] if done else None},
        "pages": pages,
    }
    (R / "result.json").write_text(json.dumps(out, indent=1))
    show = {k: v for k, v in out.items() if k not in ("pages", "by_kind")}
    print(json.dumps(show, indent=1))
    for k, v in out["by_kind"].items():
        print(f"{k:13s} m={v['m']} P(k=2)={v['P_k2']} rejected {v['rejected']}/{v['pages']}")


if __name__ == "__main__":
    main()
