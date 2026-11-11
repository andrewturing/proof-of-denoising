"""Supplement figure: what each deviation does to an SDXL image, next to how often two step checks catch it.

Rows: the first two prompts of the study (not selected). Columns: the honest render and the nine deviations, all
rendered on the L4 at 1024 x 1024 (runs/sdxl/img, decoded by scripts/family_study.py). Under each image: PSNR to the
honest render of the same job. Under each column: the family-level P(detect) with two checks (runs/sdxl_report.json).

    python scripts/make_family_fig.py
Writes ../paper-pod/fig/sdxl_examples.pdf (two-column width).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, MUTED, plt  # noqa: E402
from _paths import paper_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "sdxl"
COLS = [("honest", "honest"), ("cache2-late", "late\ncache"), ("cache2", "every-other\ncache"), ("step-skip", "step\nskip"),
        ("cfg-skip", "guidance\nskip"), ("neg-strip", "safety\nstrip"), ("w8", "8-bit\nweights"), ("w4", "4-bit\nweights"),
        ("sub-ssd1b", "SSD-1B"), ("sub-vega", "Segmind-\nVega")]
PX = 300  # pixels per panel in the PDF (0.66 in wide: about 450 dpi)


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    rep = json.loads((ROOT / "runs" / "sdxl_report.json").read_text())["fp32_detection"]
    prompts = [0, 1]
    fig, axes = plt.subplots(len(prompts), len(COLS), figsize=(6.875, 0.69 * len(prompts) + 0.75))
    for r, p in enumerate(prompts):
        hon = Image.open(RUN / "img" / f"cuda-fp32__honest__{p:02d}.png").convert("RGB")
        for c, (key, title) in enumerate(COLS):
            im = Image.open(RUN / "img" / f"cuda-fp32__{key}__{p:02d}.png").convert("RGB")
            ax = axes[r][c]
            ax.imshow(im.resize((PX, PX), Image.LANCZOS))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=6.5, color=INK, pad=3)
            label = "" if key == "honest" else f"{psnr(im, hon):.0f} dB"
            if r == len(prompts) - 1 and key != "honest":
                label += f"\n$P$ = {rep[key]['P_k2']:.2f}"
            ax.set_xlabel(label, fontsize=6.5, color=MUTED, labelpad=2)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.83, bottom=0.17, wspace=0.05, hspace=0.3)
    out = paper_dir() / "fig" / "sdxl_examples.pdf"
    fig.savefig(out, dpi=450)
    print("wrote", out)


if __name__ == "__main__":
    main()
