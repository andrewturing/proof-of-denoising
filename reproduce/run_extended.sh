#!/usr/bin/env bash
# Extended L4 session for the CVPR additions (2026-11-09). Every stage resumes where it stopped.
#   A: SANA (GPU checks + x86 CPU verifier) and the fresh storybook chapter (resume after the VM restart)
#   B: 240 disjoint calibration prompts (request-level split conformal), adaptive attacks (steering, splice),
#      dense all-step L4 checks of every SD1.5 FP32 cheat
#   bash reproduce/run_extended_vm.sh            (from the kit's pod/ directory)
set -uo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 POD_DETERMINISTIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
CHEATS=sub-bk-sdm-small,sub-bk-sdm-tiny,w8,w4,cfg-skip,cache2,cache2-late,step-skip,neg-strip

echo "== repair $(date -u +%H:%M:%S)"   # files cut by the stop: unreadable npz, partial last log line
$PY - <<'EOF'
import json, numpy as np
from pathlib import Path
for f in list(Path("runs").glob("*/traj/*.npz")) + list(Path("runs").glob("*/states/*.npz")):
    try:
        np.load(f)["states"].shape
    except Exception as e:
        print("drop", f, type(e).__name__); f.unlink()
for f in Path("runs").glob("*/*.jsonl"):
    lines = f.read_text().splitlines()
    good = []
    for l in lines:
        try:
            json.loads(l); good.append(l)
        except Exception:
            print("cut partial line in", f)
    if len(good) != len(lines):
        f.write_text("".join(l + "\n" for l in good))
EOF

echo "== A $(date -u +%H:%M:%S)"
(HF_HUB_OFFLINE=0 $PY scripts/sana_study.py --role gpu --prompts 24 --run-name sana > ~/sana_gpu2.out 2>&1; echo "sana gpu $?" >> ~/ext.log) &
SANA_GPU=$!
(HF_HUB_OFFLINE=0 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= $PY scripts/sana_study.py --role cpu --prompts 24 --run-name sana \
   --tag x86cpu-fp32 --cpu-k 8 > ~/sana_cpu2.out 2>&1; echo "sana cpu $?" >> ~/ext.log) &
SANA_CPU=$!
$PY scripts/storybook_case.py --run storybook_fresh --chapters 1,2 --plan every --device cuda --verifier cuda-fp32 \
   --stages render,decode,refall > ~/fresh2.out 2>&1; echo "fresh $?" >> ~/ext.log
tar -czf ~/fresh_results.tgz runs/storybook_fresh && touch ~/fresh.done

echo "== B $(date -u +%H:%M:%S)"
($PY scripts/gen_honest.py --backend cuda-fp32 --from 0 --to 240 --run-name calib240 \
   --prompts-file prompts/parti_calib.json --seed-base 9000 > ~/calib.out 2>&1; echo "calib $?" >> ~/ext.log) &
($PY scripts/adaptive_attacks.py --device cuda --honest-run nvidia48 --prompts 48 > ~/adaptive.out 2>&1; echo "adaptive $?" >> ~/ext.log) &
($PY scripts/cross_verify.py --traj-run nvidia48 --verifier cuda-fp32 --k 25 --out-run dense_nvidia48 \
   --only-workers cuda-fp32 --only-kinds $CHEATS > ~/dense.out 2>&1; echo "dense $?" >> ~/ext.log) &
wait

echo "== analyze + pack $(date -u +%H:%M:%S)"
$PY scripts/analyze_study.py sana > runs/sana_summary.txt 2>&1
tar -czf ~/ext_results.tgz runs/sana runs/sana_summary.txt runs/calib240 runs/adaptive runs/dense_nvidia48 runs/storybook_fresh
ls -la ~/ext_results.tgz && echo "EXT DONE $(date -u +%H:%M:%S)"
