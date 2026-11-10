"""Which steps does each deviation make detectable? Per-step r / tau on all 25 steps, from the reference (NVIDIA L4,
FP32) all-step log of the pre-registered fresh storybook (runs/storybook_fresh/verify_cuda-fp32.all.jsonl).
Rows: each deviation (its first page) and the honest pages (largest value over all 16 at each step).
Cells at or below tau are neutral; cells above tau use one sequential hue by log10(r / tau).

    python scripts/make_mask_fig.py
Writes ../poster/figs/step_masks.png (300 dpi, poster layout) and ../paper-pod/fig/step_masks.pdf (drawn at the paper's
column width, deviation names as in the paper's tables, text at 6.5-7 pt).
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, MUTED, plt  # noqa: E402
from _paths import paper_dir, poster_figs  # noqa: E402  (../paper-pod and ../poster, or figures/ standalone)

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs" / "storybook_fresh"
ORDER = ["sub-bk-small", "sub-bk-tiny", "w8", "w4", "cfg-skip", "neg-strip", "step-skip", "cache2", "cache2-late", "honest"]
LAYOUT = {
    "poster": dict(size=(6.4, 2.75), tick=8, xlabel=8.5, small=7.5, grid=1.2,
                   adjust=dict(left=0.27, right=0.93, top=0.97, bottom=0.17),
                   names={"sub-bk-small": "smaller model (Small)", "sub-bk-tiny": "smaller model (Tiny)", "w8": "8-bit weights",
                          "w4": "4-bit weights", "cfg-skip": "CFG skip", "neg-strip": "safety-prompt strip",
                          "step-skip": "step skip", "cache2": "cache, every other step",
                          "cache2-late": "cache, late steps only", "honest": "honest (max of {n} pages)"},
                   xtitle="denoising step (all 25 recomputed on the L4)", note=(0.99, 0.015, "right")),
    "paper": dict(size=(3.3, 1.95), tick=6.5, xlabel=7, small=6.5, grid=0.8,
                  adjust=dict(left=0.335, right=0.9, top=0.985, bottom=0.2),
                  names={"sub-bk-small": "BK-SDM-Small", "sub-bk-tiny": "BK-SDM-Tiny", "w8": "8-bit weights",
                         "w4": "4-bit weights", "cfg-skip": "CFG skip", "neg-strip": "Safety strip", "step-skip": "Step skip",
                         "cache2": "Cache (every other)", "cache2-late": "Cache (late only)",
                         "honest": "Honest (max of {n})"},
                  xtitle="denoising step (all 25 recomputed)", note=(0.01, 0.025, "left")),
}


def main():
    render = json.loads((R / "render.json").read_text())
    taus = {int(i): v for i, v in json.loads((R / "taus_fp32.json").read_text())["tau"].items()}
    rows, n_honest = {}, 0
    for r in map(json.loads, (R / "verify_cuda-fp32.all.jsonl").read_text().splitlines()):
        k = render["pages"][r["page"]][r["node"]]
        n_honest += k == "honest"
        over = np.array([c["ratio"] / taus[c["i"]] for c in sorted(r["checks"], key=lambda c: c["i"])])
        rows[k] = np.maximum(rows[k], over) if k == "honest" and k in rows else rows.get(k, over)
    M = np.array([rows[k] for k in ORDER])
    draw(M, n_honest, LAYOUT["poster"]).savefig(poster_figs() / "step_masks.png", dpi=300)
    draw(M, n_honest, LAYOUT["paper"]).savefig(paper_dir() / "fig" / "step_masks.pdf")
    print({k: int((rows[k] > 1).sum()) for k in ORDER}, "honest pages", n_honest)


def draw(M, n_honest, L):
    shade = np.where(M > 1, np.log10(np.maximum(M, 1)), np.nan)
    cmap = LinearSegmentedColormap.from_list("over", ["#f9d2bf", "#eb6834", "#7a2a08"])
    fig, ax = plt.subplots(figsize=L["size"])
    ax.imshow(np.where(M > 1, np.nan, 0), cmap=LinearSegmentedColormap.from_list("ok", ["#ecebe6", "#ecebe6"]),
              aspect="auto", vmin=0, vmax=1)
    im = ax.imshow(shade, cmap=cmap, aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(np.arange(-0.5, 25, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=L["grid"])
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks([0, 4, 9, 14, 19, 24])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "25"], fontsize=L["tick"])
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([L["names"][k].format(n=n_honest) for k in ORDER], fontsize=L["tick"])
    ax.set_xlabel(L["xtitle"], fontsize=L["xlabel"])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["τ", "10τ", "100τ", "10³τ", "10⁴τ"], fontsize=L["small"])
    cb.outline.set_visible(False)
    nx, ny, nha = L["note"]
    fig.text(nx, ny, "grey: r ≤ τ (passes)", ha=nha, fontsize=L["small"], color=MUTED)
    fig.subplots_adjust(**L["adjust"])
    return fig


if __name__ == "__main__":
    main()
