"""Figures from logged runs only (checks.jsonl and decoded images). Missing inputs are
skipped, never faked.

Usage: python scripts/make_figures.py --run runs/pilot --png-out runs/figs [--pdf-out DIR]
Colors: honest = neutral grey; cheats grouped into three families using the first three
validated categorical slots (blue, orange, aqua), which pass all-pairs color-vision checks.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "pilot"
OUT_P: Path = ROOT / "runs" / "figs"
OUT_C: "Path | None" = None

GREY, INK, MUTED = "#8a8984", "#0b0b0b", "#52514e"
FAMILY = {  # cheat -> (family, color)
    "sub-bk-small": ("every step", "#2a78d6"), "sub-bk-tiny": ("every step", "#2a78d6"),
    "w8": ("every step", "#2a78d6"), "w4": ("every step", "#2a78d6"), "cfg-skip": ("every step", "#2a78d6"),
    "cache2": ("shortcut steps", "#eb6834"), "cache2-late": ("shortcut steps", "#eb6834"),
    "step-skip": ("shortcut steps", "#eb6834"),
    "neg-strip": ("safety strip", "#1baf7a"),
}
FAMILY["sub-bk-sdm-small"], FAMILY["sub-bk-sdm-tiny"] = FAMILY["sub-bk-small"], FAMILY["sub-bk-tiny"]
LABEL = {"sub-bk-small": "BK-SDM-S", "sub-bk-tiny": "BK-SDM-T", "w8": "W8", "w4": "W4", "cfg-skip": "CFG skip",
         "cache2": "cache", "cache2-late": "late cache", "step-skip": "step skip", "neg-strip": "safety strip"}
ORDER = list(LABEL)
LABEL.update({"sub-bk-sdm-small": "BK-SDM-S", "sub-bk-sdm-tiny": "BK-SDM-T"})
ORDER = ORDER[:2] + ["sub-bk-sdm-small", "sub-bk-sdm-tiny"] + ORDER[2:]

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8})


def save(fig, name):
    OUT_P.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_P / f"{name}.png", dpi=300, bbox_inches="tight")
    if OUT_C is not None:
        OUT_C.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_C / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load_checks():
    path = RUN / "checks.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


MARGIN = 1.5


def thresholds(recs):
    """Leave-one-prompt-out per-step thresholds, exactly as in scripts/analyze_pilot.py:
    tau_i = MARGIN x max honest r at step i over all other prompts and backend pairs."""
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["kind"] == "honest":
            for c in r["checks"]:
                by[r["prompt"]][c["i"]].append(c["ratio"])
    taus = {}
    for p in by:
        acc = defaultdict(list)
        for q, steps in by.items():
            if q != p:
                for i, xs in steps.items():
                    acc[i].extend(xs)
        taus[p] = {i: MARGIN * max(xs) for i, xs in acc.items()}
    return taus


def honest_class(r):
    if r["worker"] == r["verifier"]:
        return "same backend"
    if "fp16" in (r["worker"] + r["verifier"]):
        return "FP16 vs FP32"
    return "CPU vs GPU, FP32"


HCOL = {"same backend": "#b9b8b2", "CPU vs GPU, FP32": "#8a8984", "FP16 vs FP32": "#52514e"}


def fig_drift_vs_cheats(recs, verifier="mps-fp32"):
    """Per trajectory: max over steps of r_i / tau_i (1 = detection threshold)."""
    recs = [r for r in recs if r["verifier"] == verifier]
    if not recs:
        return
    taus = thresholds(recs)

    def norm(r):
        tau = taus.get(r["prompt"], {})
        return max(c["ratio"] / tau[c["i"]] for c in r["checks"] if c["i"] in tau)

    cats, vals, cols = [], [], []
    for hc in ("same backend", "CPU vs GPU, FP32", "FP16 vs FP32"):
        v = [norm(r) for r in recs if r["kind"] == "honest" and honest_class(r) == hc]
        if v:
            short = {"same backend": "honest\nsame\nbackend", "CPU vs GPU, FP32": "honest\nCPU↔GPU\nFP32",
                     "FP16 vs FP32": "honest\nFP16↔\nFP32"}[hc]
            cats.append(short); vals.append(v); cols.append(HCOL[hc])
    for k in ORDER:
        v = [norm(r) for r in recs if r["kind"] == k]
        if v:
            cats.append({"sub-bk-small": "BK-SDM\nSmall", "sub-bk-tiny": "BK-SDM\nTiny"}.get(k, LABEL[k].replace(" ", "\n")))
            vals.append(v); cols.append(FAMILY[k][1])
    fig, ax = plt.subplots(figsize=(9.9, 4.3))
    rng = np.random.default_rng(0)
    for j, (v, col) in enumerate(zip(vals, cols)):
        v = np.maximum(np.array(v), 1e-7)
        ax.scatter(j + rng.uniform(-0.2, 0.2, len(v)), v, s=36, color=col, edgecolor="white", linewidth=0.9, zorder=3)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(-0.45, 1.35, "detection threshold", ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(5e-8, 2e3)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("max over steps of  r_i / τ_i", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.text(0, 1.2e-7, "bit-identical\n(plotted at floor)", ha="center", va="bottom", fontsize=9.5, color=MUTED)
    save(fig, "drift_vs_cheats")


def fig_per_step(recs, verifier="mps-fp32"):
    """r_i by step: honest bands per precision class, plus three cheats (medians)."""
    recs = [r for r in recs if r["verifier"] == verifier]
    fig, ax = plt.subplots(figsize=(9.9, 3.6))
    drew = False
    for hc, alpha in (("CPU vs GPU, FP32", 0.45), ("FP16 vs FP32", 0.55)):
        per = defaultdict(list)
        for r in recs:
            if r["kind"] == "honest" and honest_class(r) == hc:
                for c in r["checks"]:
                    per[c["i"]].append(c["ratio"])
        if per:
            xs = sorted(per)
            ax.fill_between(xs, [min(per[i]) for i in xs], [max(per[i]) for i in xs], color=HCOL[hc], alpha=alpha,
                            linewidth=0, label=f"honest, {hc} (min–max)")
            drew = True
    if not drew:
        plt.close(fig)
        return
    for kind in ("w8", "cache2-late", "neg-strip"):
        per = defaultdict(list)
        for r in recs:
            if r["kind"] == kind:
                for c in r["checks"]:
                    per[c["i"]].append(c["ratio"])
        if per:
            xs = sorted(per)
            ax.plot(xs, [np.median(per[i]) for i in xs], color=FAMILY[kind][1], linewidth=2, marker="o", markersize=4,
                    label=LABEL[kind] + " (median)")
    ax.set_yscale("log")
    ax.set_xlabel("denoising step i (0 = noisiest)", fontsize=13)
    ax.set_ylabel("mismatch r_i", fontsize=13)
    ax.legend(fontsize=10.5, frameon=False, ncol=2, loc="lower right")
    save(fig, "per_step")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def load_imgs():
    d = RUN / "img"
    return {p.stem: np.asarray(Image.open(p)) for p in d.glob("*.png")} if d.exists() else {}


def fig_output_psnr(imgs):
    if not imgs:
        return
    workers = sorted({k.split("__")[0] for k in imgs if "__honest__" in k})
    prompts = sorted({k.split("__")[2] for k in imgs})
    hon = [psnr(imgs[f"{a}__honest__{p}"], imgs[f"{b}__honest__{p}"]) for p in prompts for a in workers for b in workers
           if a < b and f"{a}__honest__{p}" in imgs and f"{b}__honest__{p}" in imgs]
    by = defaultdict(list)
    for k, im in imgs.items():
        w, kind, p = k.split("__")
        if kind != "honest" and f"{w}__honest__{p}" in imgs:
            by[kind].append(psnr(im, imgs[f"{w}__honest__{p}"]))
    if not hon:
        return
    cats = ["honest"] + [k for k in ORDER if by.get(k)]
    fig, ax = plt.subplots(figsize=(9.9, 2.6))
    rng = np.random.default_rng(1)
    for j, cat in enumerate(cats):
        vals = np.array(hon if cat == "honest" else by[cat])
        ax.scatter(j + rng.uniform(-0.18, 0.18, len(vals)), vals, s=30, color=GREY if cat == "honest" else FAMILY[cat][1],
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(min(hon), color=INK, linestyle="--", linewidth=1.5)
    ax.text(len(cats) - 0.5, min(hon) + 0.5, "lowest honest PSNR", ha="right", va="bottom", fontsize=12)
    ax.set_xticks(range(len(cats)))
    short = {"sub-bk-small": "BK-SDM\nSmall", "sub-bk-tiny": "BK-SDM\nTiny"}
    ax.set_xticklabels(["honest\npairs"] + [short.get(c, LABEL[c].replace(" ", "\n")) for c in cats[1:]], fontsize=10)
    ax.set_ylabel("PSNR vs honest (dB)", fontsize=13)
    save(fig, "output_psnr")


def fig_drift_examples(imgs, p="00"):
    keys = [f"{w}__honest__{p}" for w in ("mps-fp16", "mps-fp32", "cpu-fp32")]
    if not all(k in imgs for k in keys):
        return
    a, b, c = (imgs[k] for k in keys)
    diff = np.clip(np.abs(a.astype(int) - c.astype(int)).mean(axis=2) * 10, 0, 255).astype(np.uint8)
    fig, axs = plt.subplots(1, 4, figsize=(9.9, 3.0))
    for ax, im, t in zip(axs, (a, b, c, diff), ("Apple GPU FP16", "Apple GPU FP32", "CPU FP32", "|FP16 − CPU| ×10")):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=12, color=INK)
        ax.axis("off")
    fig.suptitle("same weights, prompt and seed", fontsize=12, color=MUTED, y=0.12)
    save(fig, "drift_examples")


def main(argv=None):
    global RUN, OUT_P, OUT_C
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--png-out", type=Path, default=OUT_P)
    ap.add_argument("--pdf-out", type=Path, default=None)
    a = ap.parse_args(argv)
    RUN, OUT_P, OUT_C = a.run, a.png_out, a.pdf_out
    recs, imgs = load_checks(), load_imgs()
    print(f"{len(recs)} check records, {len(imgs)} images", file=sys.stderr)
    fig_drift_vs_cheats(recs)
    fig_per_step(recs)
    fig_output_psnr(imgs)
    fig_drift_examples(imgs)


if __name__ == "__main__":
    main()
