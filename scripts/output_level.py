"""Output-level baseline: can comparing final images separate honest drift from cheating?

The threshold is the lowest PSNR between honest images of the same job on different
backends (any precision), so it accepts every honest pair. Then count how many cheat
images (vs the honest image from the same worker) clear that threshold.
Writes runs/pilot/output_level.json.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(__file__).resolve().parents[1] / "runs" / "pilot"
WORKERS = ["cpu-fp32", "mps-fp16", "mps-fp32"]
CHEATS = ["sub-bk-small", "sub-bk-tiny", "w8", "w4", "cfg-skip", "cache2", "cache2-late", "step-skip", "neg-strip"]


def psnr(a, b) -> float:
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if m == 0 else float(10 * np.log10(255.0**2 / m))


def main():
    imgs = {p.stem: np.asarray(Image.open(p)) for p in (RUN / "img").glob("*.png")}
    prompts = sorted({int(k.split("__")[2]) for k in imgs})
    honest_pairs = {}
    for i, a in enumerate(WORKERS):
        for b in WORKERS[i + 1:]:
            honest_pairs[f"{a}|{b}"] = [psnr(imgs[f"{a}__honest__{p:02d}"], imgs[f"{b}__honest__{p:02d}"]) for p in prompts]
    thr = min(min(v) for v in honest_pairs.values())
    out = {"threshold_min_honest_psnr": thr,
           "honest_pairs": {k: {"median": float(np.median(v)), "min": float(min(v))} for k, v in honest_pairs.items()},
           "cheats": {}}
    for kind in CHEATS:
        v = [psnr(imgs[f"mps-fp16__{kind}__{p:02d}"], imgs[f"mps-fp16__honest__{p:02d}"]) for p in prompts]
        out["cheats"][kind] = {"median": float(np.median(v)), "accepted": int(sum(x >= thr for x in v)), "n": len(v)}
    out["accepted_total"] = sum(c["accepted"] for c in out["cheats"].values())
    out["cheat_images_total"] = sum(c["n"] for c in out["cheats"].values())
    (RUN / "output_level.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
