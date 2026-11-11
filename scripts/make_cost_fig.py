"""Detection versus audit cost, from logged per-step checks (no hand-typed numbers).

For every cheat trajectory rendered and checked on the NVIDIA L4 (FP32), the flagged steps against the frozen pooled
thresholds (runs/storybook_fresh/taus_fp32.json) give P(detect | k) = 1 - C(c-f, k) / C(c, k) for c observed steps
(c = 25 once the dense all-step pass is merged in, so the value is exact for that trajectory). Lines: mean over prompts;
bands: 95% bootstrap over prompts. The dashed line is a trajectory with exactly one detectable step (one spliced
transition): P = k / 25.

    python scripts/make_cost_fig.py
Writes ../poster/figs/detect_vs_cost.png (300 dpi, poster layout), ../paper-pod/fig/detect_vs_cost.pdf (drawn at the
paper's column width so its text prints at 6.5-7.5 pt) and runs/detect_vs_cost.json.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import GREY, INK, MUTED, plt  # noqa: E402  (shared poster style)
from _paths import paper_dir, poster_figs  # noqa: E402  (../paper-pod and ../poster, or figures/ standalone)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
N, KS = 25, list(range(1, 13))
EVERY = ["sub-bk-sdm-small", "sub-bk-sdm-tiny", "w8", "w4", "cfg-skip", "step-skip", "neg-strip"]
SERIES = [("every", "#2a78d6", "o", "-"), ("cache2", "#eb6834", "s", "-"), ("cache2-late", "#eb6834", "^", "--")]
# per layout: figure size (in), font sizes (pt), line widths, label text and positions (data coordinates)
LAYOUT = {
    "poster": dict(size=(6.4, 3.9), text=9.5, tick=9, axis=10, lw=2, ms=5,
                   adjust=dict(left=0.1, right=0.98, top=0.97, bottom=0.2),
                   labels={"every": ("{n} deviations that change every step (weakest shown)", 2.2, 1.055),
                           "cache2": ("cache,\nevery other step", 3.6, 0.885),
                           "cache2-late": ("cache, late steps only", 6.6, 0.7)},
                   splice=("one spliced step (exact: k / 25)", 7.6, 0.2),
                   xlabel="checked steps k  (cost: k / 25 of the network evaluations)"),
    "paper": dict(size=(3.3, 2.0), text=7, tick=6.5, ytick=6.5, axis=7.5, lw=1.3, ms=3,
                  adjust=dict(left=0.13, right=0.985, top=0.975, bottom=0.25),
                  labels={"every": ("{n} every-step deviations (weakest)", 2.0, 1.1),
                          "cache2": ("cache, every other step", None, None),
                          "cache2-late": ("cache, late steps only", None, None)},
                  key_at=(5.9, 0.77), ylim=1.18,  # unplaced labels go in a key in the empty middle right
                  splice=("one spliced step (k / 25)", 6.9, 0.17),
                  xlabel="checked steps k  (cost: % of network evaluations)"),
}


def p_detect(c, f, k):
    return 1 - math.comb(c - f, k) / math.comb(c, k) if k <= c else float("nan")


def draw(curves, n_every, L):
    fig, ax = plt.subplots(figsize=L["size"])
    for key, col, mk, ls in SERIES:
        if key not in curves:
            continue
        m, lo, hi = curves[key]
        ax.fill_between(KS, lo, hi, color=col, alpha=0.16, linewidth=0)
        ax.plot(KS, m, color=col, linewidth=L["lw"], marker=mk, markersize=L["ms"], linestyle=ls)
        label, lx, ly = L["labels"][key]
        if lx is None:
            ax.lines[-1].set_label(label)
        else:
            ax.text(lx, ly, label.format(n=n_every), color=INK, fontsize=L["text"], va="center", ha="left")
    ax.plot(KS, [k / N for k in KS], color=INK, linewidth=L["lw"], linestyle=(0, (4, 3)))
    label, lx, ly = L["splice"]
    ax.text(lx, ly, label, color=INK, fontsize=L["text"], va="center")
    ax.axhline(0.99, color=GREY, linewidth=L["lw"] / 2, linestyle=":")
    # dotted line: P = 0.99 (named in the caption)
    ax.set_xticks(KS)
    ax.set_xticklabels([f"{k}\n{100 * k / N:.0f}%" for k in KS], fontsize=L["tick"])
    if L.get("ytick"):
        ax.tick_params(axis="y", labelsize=L["ytick"])
    ax.set_xlabel(L["xlabel"], fontsize=L["axis"])
    ax.set_ylabel("P(detect)", fontsize=L["axis"])
    ax.set_ylim(0, L.get("ylim", 1.1))
    if L.get("key_at"):
        ax.legend(loc="upper left", bbox_to_anchor=L["key_at"], bbox_transform=ax.transData, fontsize=L["text"],
                  frameon=False, handlelength=2.6, borderaxespad=0, borderpad=0, labelspacing=0.35)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0.7, KS[-1] + 0.8)
    fig.subplots_adjust(**L["adjust"])
    return fig


def main():
    taus = {int(i): v for i, v in json.loads((RUNS / "storybook_fresh" / "taus_fp32.json").read_text())["tau"].items()}
    steps = defaultdict(dict)  # (kind, prompt) -> {i: ratio}
    for run in ("nvidia48", "dense_nvidia48"):
        f = RUNS / run / "checks.jsonl"
        if not f.exists():
            continue
        for r in map(json.loads, f.read_text().splitlines()):
            if r["worker"] == "cuda-fp32" and r["verifier"] == "cuda-fp32" and r["kind"] != "honest":
                for c in r["checks"]:
                    steps[(r["kind"], r["prompt"])][c["i"]] = c["ratio"]
    curves, rng = {}, np.random.default_rng(0)
    out = {"thresholds": "runs/storybook_fresh/taus_fp32.json (pooled max, frozen)", "series": {}}
    for kind in EVERY + ["cache2", "cache2-late"]:
        rows = []
        for (k_, p), st in sorted(steps.items()):
            if k_ == kind:
                c, fl = len(st), sum(r > taus[i] for i, r in st.items())
                rows.append([p_detect(c, fl, k) for k in KS])
        if not rows:
            continue
        a = np.array(rows)
        boot = np.array([np.nanmean(a[rng.integers(0, len(a), len(a))], axis=0) for _ in range(2000)])
        curves[kind] = (np.nanmean(a, axis=0), np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0))
        out["series"][kind] = {"prompts": len(a), "observed_steps": sorted({len(st) for (k_, _), st in steps.items() if k_ == kind}),
                               "mean": dict(zip(KS, curves[kind][0].round(4).tolist())),
                               "ci95_lo": dict(zip(KS, curves[kind][1].round(4).tolist()))}
    every = [k for k in EVERY if k in curves]
    if every:  # the weakest every-step deviation at each k
        lo = np.min([curves[k][0] for k in every], axis=0)
        lo_ci = np.min([curves[k][1] for k in every], axis=0)
        hi_ci = np.min([curves[k][2] for k in every], axis=0)
        curves["every"] = (lo, lo_ci, hi_ci)
    n = max((v["prompts"] for v in out["series"].values()), default=0)
    out["prompts"] = n
    draw(curves, len(every), LAYOUT["poster"]).savefig(poster_figs() / "detect_vs_cost.png", dpi=300)
    draw(curves, len(every), LAYOUT["paper"]).savefig(paper_dir() / "fig" / "detect_vs_cost.pdf")
    (RUNS / "detect_vs_cost.json").write_text(json.dumps(out, indent=1))
    print("prompts", n, {k: (v["observed_steps"], v["mean"][2]) for k, v in out["series"].items()})


if __name__ == "__main__":
    main()
