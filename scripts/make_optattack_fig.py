"""Paper figures for the full-cost attacks, including optimised steering (scripts/optimal_steer.py).

Images are decoded here from the committed final latents (SD1.5 VAE). PSNR labels come from the attack logs, where
each attack image is compared with the honest render of the same run.

  fig/attacks.pdf        main paper, one row: the prompt on which optimised steering changes the image most.
                         honest | fixed-direction steering | optimised steering | one spliced step | target
  fig/attacks_more.pdf   supplement: the first prompt (not selected) and the worst prompts, with both budgets.

    python scripts/make_optattack_fig.py [--device mps]
The poster's attacks.png (scripts/make_attack_fig.py) is not touched.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, MUTED, plt  # noqa: E402
from _paths import paper_dir  # noqa: E402

RUNS = ROOT / "runs"
OLD, NEW, HON = RUNS / "adaptive", RUNS / "adaptive_opt", RUNS / "nvidia48"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--more", type=int, default=3, help="worst prompts shown in the supplement figure")
    a = ap.parse_args()
    from podlab.model import Backend, Denoiser
    old = {r["prompt"]: r for r in map(json.loads, (OLD / "outputs.jsonl").read_text().splitlines()) if r}
    new = {r["prompt"]: r for r in map(json.loads, (NEW / "outputs.jsonl").read_text().splitlines()) if r}
    n = len(new)
    have = [p for p in new if (NEW / "traj" / f"cuda-fp32__max95__{p:02d}.npz").exists()]
    worst = sorted(have, key=lambda p: new[p]["attacks"]["max95"]["psnr_vs_honest"])
    den = Denoiser(Backend(a.device, "fp32"), load_vae=True)
    last = lambda f: den.decode(np.load(f)["states"][-1])

    def images(p):
        t = new[p]["target"]
        return {"honest": last(HON / "traj" / f"cuda-fp32__honest__{p:02d}.npz"),
                "steer95": last(OLD / "traj" / f"cuda-fp32__steer95__{p:02d}.npz"),
                "max33": last(NEW / "traj" / f"cuda-fp32__max33__{p:02d}.npz"),
                "max95": last(NEW / "traj" / f"cuda-fp32__max95__{p:02d}.npz"),
                "splice20": last(OLD / "traj" / f"cuda-fp32__splice20__{p:02d}.npz"),
                "target": last(HON / "traj" / f"cuda-fp32__honest__{t:02d}.npz")}

    def db(p, key):
        if key == "honest":
            return ""
        if key == "target":
            return f"{new[p]['psnr_target_vs_honest']:.0f} dB"
        src = new if key.startswith("max") else old
        return f"{src[p]['attacks'][key]['psnr_vs_honest']:.0f} dB"

    def draw(prompts, cols, titles, width, cell, title_size, label_size, top, bottom, hspace):
        fig, axes = plt.subplots(len(prompts), len(cols), figsize=(width, cell * len(prompts) + (0.62 if len(prompts) == 1 else 0.75)), squeeze=False)
        for r, p in enumerate(prompts):
            ims = images(p)
            for c, key in enumerate(cols):
                ax = axes[r][c]
                ax.imshow(ims[key])
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if r == 0:
                    ax.set_title(titles[key], fontsize=title_size, color=INK, pad=3)
                ax.set_xlabel(db(p, key), fontsize=label_size, color=MUTED, labelpad=2)
        fig.subplots_adjust(left=0.005, right=0.995, top=top, bottom=bottom, wspace=0.06, hspace=hspace)
        return fig

    w = worst[0]
    draw([w], ["honest", "steer95", "max95", "splice20", "target"],
         {"honest": "honest", "steer95": "fixed\nsteering", "max95": "optimized\nsteering", "splice20": "one spliced\nstep", "target": "attacker's\ntarget"},
         3.3, 0.64, 6.5, 6.5, 0.74, 0.16, 0.3).savefig(paper_dir() / "fig" / "attacks.pdf", dpi=400)
    first = min(have)
    rows = [first] + [p for p in worst if p != first][: a.more]
    draw(rows, ["honest", "steer95", "max33", "max95", "splice20", "target"],
         {"honest": "honest", "steer95": "fixed steering\n(c = 0.95)", "max33": "optimized steering\n(c = 0.33)", "max95": "optimized steering\n(c = 0.95)",
          "splice20": "one spliced step\n(step 20 of 25)", "target": "attacker's target"},
         6.875, 1.12, 7, 7, 0.93, 0.04, 0.22).savefig(paper_dir() / "fig" / "attacks_more.pdf", dpi=300)
    (RUNS / "optattack_fig.json").write_text(json.dumps({"prompts": n, "main_prompt": w, "main_psnr": new[w]["attacks"]["max95"]["psnr_vs_honest"],
                                                          "supplement_prompts": rows}, indent=1))
    print("main figure: prompt", w, "| supplement rows:", rows)


if __name__ == "__main__":
    main()
