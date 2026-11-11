"""Overview figure for the paper (full text width): the protocol, and why a threshold separates honest drift from
shortcuts. Every plotted value comes from logged per-step checks; nothing is typed.

  left   the commit -> challenge -> recompute protocol (a schematic)
  right  per-step mismatch r on a log axis, as min-max bars with the median marked:
           honest FP32 stacks checking each other (every cross-device pair in the pooled logs)
           honest TF32 / FP16 trajectories checked as FP32 jobs (silent precision downgrades)
           each shortcut on the steps it changes (NVIDIA L4, all 25 steps recomputed)
         and the band of thresholds tau_i (runs/storybook_fresh/taus_fp32.json)

    python scripts/make_overview_fig.py
Writes ../paper-pod/fig/overview.pdf (or figures/paper/fig/ in a standalone checkout).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import GREY, INK, MUTED, plt  # noqa: E402
from _paths import paper_dir  # noqa: E402
from dense_margin import changed  # noqa: E402  (steps each deviation changes, by its definition)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2e8b57"
ROWS = [("honest", "Honest FP32 stacks", MUTED),
        ("tf32", "TF32 claimed as FP32", ORANGE), ("fp16", "FP16 claimed as FP32", ORANGE),
        ("w8", "8-bit weights", BLUE), ("cache2-late", "Cache (late only)", BLUE), ("cache2", "Cache (every other)", BLUE),
        ("step-skip", "Step skip", BLUE), ("neg-strip", "Safety strip", BLUE), ("cfg-skip", "CFG skip", BLUE),
        ("w4", "4-bit weights", BLUE), ("sub-bk-sdm-tiny", "BK-SDM-Tiny", BLUE), ("sub-bk-sdm-small", "BK-SDM-Small", BLUE)]


def cls(backend: str) -> str:
    return backend.rsplit("-", 1)[-1]


def collect():
    """Per-step r by row, from the largest pooled analysis' runs."""
    pool = max((d.name for d in RUNS.iterdir() if d.is_dir() and d.name.startswith("study+")), key=lambda n: n.count("+"))
    r = defaultdict(list)
    for run in pool.split("+"):
        f = RUNS / run / "checks.jsonl"
        if not f.exists():
            continue
        for rec in map(json.loads, f.read_text().splitlines()):
            w, v, kind = rec["worker"], rec["verifier"], rec["kind"]
            if cls(v) != "fp32":
                continue
            vals = [(c["i"], c["ratio"]) for c in rec["checks"]]
            if kind == "honest" and cls(w) == "fp32" and w != v:
                r["honest"] += [x for _, x in vals]
            elif kind == "honest" and w == "cuda-tf32":
                r["tf32"] += [x for _, x in vals]
            elif kind == "honest" and w == "cuda-fp16":
                r["fp16"] += [x for _, x in vals]
            elif kind != "honest" and w == "cuda-fp32" and v == "cuda-fp32":
                ch = changed(kind)
                r[kind] += [x for i, x in vals if i in ch]
    return pool, r


def box(ax, x, y, w, h, text, fc="#f4f3ef", ec=GREY, fs=7, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight, linespacing=1.15)


def arrow(ax, a, b, color=INK, lw=0.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=6, lw=lw, color=color, shrinkA=0, shrinkB=0))


def protocol(ax):
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 56)
    ax.axis("off")
    ax.text(0, 54, "Worker (untrusted GPU)", fontsize=7.5, weight="bold", color=INK, va="center")
    # the trajectory z_0 ... z_N
    xs, labels = [2, 15, 40, 53, 79], ["$z_0$", "$z_1$", "$z_i$", "$z_{i+1}$", "$z_N$"]
    for x, lab in zip(xs, labels):
        chal = lab in ("$z_i$", "$z_{i+1}$")
        ax.add_patch(Rectangle((x, 40), 10, 9, fc="#fde7dc" if chal else "#f4f3ef", ec=ORANGE if chal else GREY, lw=0.9 if chal else 0.7))
        ax.text(x + 5, 44.5, lab, ha="center", va="center", fontsize=7.5, color=INK)
    for a, b in [(12, 15), (50, 53)]:
        arrow(ax, (a, 44.5), (b, 44.5))
    for a, b in [(25, 40), (63, 79)]:
        ax.text((a + b) / 2, 44.5, "···", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(51.5, 51.6, "Step", ha="center", va="center", fontsize=6.3, color=MUTED)
    arrow(ax, (89.6, 44.5), (92.4, 44.5), color=MUTED, lw=0.6)
    ax.text(93.2, 44.5, "image", ha="left", va="center", fontsize=6.3, color=MUTED)
    # 1: commitment (every state feeds the root)
    ax.plot([7, 84], [38, 38], color=GREY, lw=0.7)
    for x in xs:
        ax.plot([x + 5, x + 5], [38, 40], color=GREY, lw=0.7)
    ax.plot([16, 16], [36.4, 38], color=GREY, lw=0.7)
    box(ax, 1, 27, 38, 9, "1  commit: Merkle root over\nthe header + every latent", fs=6.2)
    # verifier
    ax.text(7, 23.3, "Verifier (any device)", fontsize=7.5, weight="bold", color=INK, va="center")
    box(ax, 1, 10.5, 39, 9, "2  after the root, draw\n$k$ random steps $i$", fs=6.2)
    box(ax, 44.5, 10.5, 27, 9, "3  recompute\n$\\hat z_{i+1}=\\mathrm{Step}(z_i)$", fs=6.2)
    box(ax, 76, 10.5, 23, 9, "4  $r_i\\leq\\tau_i$ ?\n(Eq. 1)", fs=6.2)
    arrow(ax, (40.8, 15), (43.7, 15))
    arrow(ax, (72.3, 15), (75.2, 15))
    arrow(ax, (4, 26.2), (4, 20.4), color=MUTED, lw=0.7)               # the challenge comes after the root
    arrow(ax, (51.5, 39.3), (51.5, 20.4), color=ORANGE, lw=0.9)       # opened states go to the verifier
    ax.text(53.3, 30.5, "open $z_i,\\,z_{i+1}$\nwith Merkle paths", fontsize=6.2, color=ORANGE, va="center", linespacing=1.15)
    box(ax, 59, 0, 13, 6, "accept", fc="#e6f2ea", ec=GREEN, fs=6.2, color="#1f6b43")
    box(ax, 76, 0, 23, 6, "re-check on a\nreference GPU", fc="#fde7dc", ec=ORANGE, fs=5.8)
    arrow(ax, (79, 9.9), (71, 6.8), color=GREEN)
    arrow(ax, (90, 9.9), (90, 6.8), color=ORANGE)
    ax.text(70.5, 8.9, "yes", fontsize=6, color="#1f6b43", va="center", ha="right")
    ax.text(91.5, 8.4, "no: flag", fontsize=6, color=ORANGE, va="center")


def separation(ax, r, taus):
    ys = np.arange(len(ROWS))[::-1]
    ax.axvspan(min(taus), max(taus), color="#d9d7cf", alpha=0.6, lw=0)
    for y, (key, label, col) in zip(ys, ROWS):
        v = np.array(r[key])
        v = v[v > 0]
        ax.plot([v.min(), v.max()], [y, y], color=col, lw=3.2, solid_capstyle="round", alpha=0.85)
        ax.plot([np.median(v)], [y], marker="|", color="white", ms=5.5, mew=1.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in ROWS], fontsize=6.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 20)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("per-step mismatch $r$ (log scale)", fontsize=7.2)
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.text(np.sqrt(min(taus) * max(taus)), len(ROWS) - 0.42, "thresholds $\\tau_i$", ha="center", va="bottom", fontsize=6.3, color=MUTED)
    # groups, bracketed in the right margin
    tr = ax.get_yaxis_transform()  # x in axes fraction, y in data
    for a, b, text, col in [(0, 0, "honest", MUTED), (1, 2, "silent\ndowngrades", ORANGE),
                            (3, 11, "shortcuts,\non the steps\nthey change", BLUE)]:
        ya, yb = ys[a] + 0.36, ys[b] - 0.36
        ax.plot([1.03, 1.03], [ya, yb], color=col, lw=0.9, clip_on=False, transform=tr)
        ax.text(1.055, (ya + yb) / 2, text, transform=tr, fontsize=6.2, color=col, va="center", ha="left", linespacing=1.1)


def main():
    pool, r = collect()
    taus = list(json.loads((RUNS / "storybook_fresh" / "taus_fp32.json").read_text())["tau"].values())
    missing = [k for k, _, _ in ROWS if not r[k]]
    assert not missing, missing
    fig = plt.figure(figsize=(6.875, 1.95))
    protocol(fig.add_axes([0.005, 0.03, 0.475, 0.95]))
    separation(fig.add_axes([0.652, 0.2, 0.245, 0.72]), r, taus)
    out = paper_dir() / "fig" / "overview.pdf"
    fig.savefig(out)
    stats = {k: {"n": len(r[k]), "min": float(np.min([x for x in r[k] if x > 0])), "median": float(np.median(r[k])),
                 "max": float(np.max(r[k]))} for k, _, _ in ROWS}
    (RUNS / "overview_fig.json").write_text(json.dumps({"pool": pool, "tau_min": min(taus), "tau_max": max(taus), "rows": stats}, indent=1))
    print(out)
    for k, v in stats.items():
        print(f"  {k:18s} n={v['n']:6d} min={v['min']:.1e} median={v['median']:.1e} max={v['max']:.1e}")


if __name__ == "__main__":
    main()
