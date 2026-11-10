"""Poster/paper figure for the full-cost attacks (scripts/adaptive_attacks.py), from logged outputs only.
Rows: the first N prompts (not selected). Columns: honest image | steered inside the thresholds on every step
(c = 0.95) | one spliced step (s = 20) | the target the attacker steers towards. Each panel carries its PSNR to the
honest image and how many steps a verifier can detect.

    python scripts/make_attack_fig.py [--prompts 2]
Writes ../poster/figs/attacks.png (300 dpi, poster layout) and ../paper-pod/fig/attacks.pdf (drawn at the paper's column
width with short labels; the paper caption explains them).
"""
import argparse
import json
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, MUTED, plt  # noqa: E402
from _paths import paper_dir, poster_figs  # noqa: E402  (../paper-pod and ../poster, or figures/ standalone)

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "runs" / "adaptive"
COLS = ["honest", "steer95", "splice20", "target"]
LAYOUT = {
    "poster": dict(width=6.4, row_h=1.75, extra_h=0.55, title=8.5, label=7.5,
                   adjust=dict(left=0.01, right=0.99, top=0.86, bottom=0.12, wspace=0.05, hspace=0.45),
                   titles={"honest": "honest render", "steer95": "steered on every step,\ninside every threshold",
                           "splice20": "one spliced step\n(step 20 of 25)", "target": "attacker's target"},
                   psnr="PSNR {db:.0f} dB vs honest", short=False),
    "paper": dict(width=3.3, row_h=0.8, extra_h=0.62, title=7, label=6.5,
                  adjust=dict(left=0.005, right=0.995, top=0.8, bottom=0.2, wspace=0.06, hspace=0.45),
                  titles={"honest": "honest", "steer95": "steered\n(every step)", "splice20": "one spliced\nstep (20 of 25)",
                          "target": "attacker's\ntarget"},
                  psnr="{db:.0f} dB", short=True),  # caption: PSNR to honest; steps over tau on the x86 CPU
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=2)
    a = ap.parse_args()
    recs = {r["prompt"]: r for r in map(json.loads, (A / "outputs.jsonl").read_text().splitlines())}
    ps = [p for p in sorted(recs) if (A / "img" / f"{p:02d}__honest.png").exists()][: a.prompts]
    xv = A / "xverify_summary.json"  # cross-device check of these trajectories, when it has run
    xv = json.loads(xv.read_text()) if xv.exists() else {}
    draw(recs, ps, xv, LAYOUT["poster"]).savefig(poster_figs() / "attacks.png", dpi=300)
    draw(recs, ps, xv, LAYOUT["paper"]).savefig(paper_dir() / "fig" / "attacks.pdf", dpi=400)  # images at 400 dpi
    print("prompts", ps)


def draw(recs, ps, xv, L):
    fig, axes = plt.subplots(len(ps), len(COLS), figsize=(L["width"], L["row_h"] * len(ps) + L["extra_h"]))
    for r, p in enumerate(ps):
        for c, key in enumerate(COLS):
            ax = axes[r][c] if len(ps) > 1 else axes[c]
            ax.imshow(Image.open(A / "img" / f"{p:02d}__{key}.png"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(L["titles"][key], fontsize=L["title"], color=INK)
            if key in ("steer95", "splice20"):
                at = recs[p]["attacks"][key]
                det = ("each step at r = 0.95 τ" if key == "steer95" else "1 inconsistent step of 25")
                v = xv.get(key, {}).get(str(p))
                if v is not None:
                    x86 = v["verifier"].startswith("x86")
                    assert x86 or not L["short"], "the paper caption names the x86 CPU as this check's verifier"
                    det = f"{v['over']}/{v['steps']} steps > τ" + (" (x86 CPU)" if x86 and not L["short"] else "")
                ax.set_xlabel(L["psnr"].format(db=at["psnr_vs_honest"]) + f"\n{det}", fontsize=L["label"], color=MUTED)
            elif key == "target":
                ax.set_xlabel(L["psnr"].format(db=recs[p]["psnr_target_vs_honest"]), fontsize=L["label"], color=MUTED)
    fig.subplots_adjust(**L["adjust"])
    return fig


if __name__ == "__main__":
    main()
