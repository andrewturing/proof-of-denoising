#!/usr/bin/env bash
# Two further model families on an NVIDIA L4 (24 GB): SDXL (1024 px) and Wan 2.1 text-to-video 1.3B.
#   bash reproduce/run_families.sh [sdxl_prompts] [wan_prompts]        (from the kit's pod/ directory)
# Every stage resumes where it stopped. Stage markers and logs are written to ~/fam/.
#   GPU: SDXL study, then Wan study (honest runs in each precision class, every deviation in FP32, all checked
#        on cuda-fp32).
#   CPU: Wan text embeddings first, before anything else loads (the UMT5 encoder needs ~23 GB of RAM); then, in the
#        background, the host CPU re-checks random steps of the honest FP32 jobs as a second FP32 stack (x86cpu-fp32).
# Results are packed after each stage (~/fam_results.tgz: logs, example images, honest FP32 trajectories).
set -uo pipefail
cd "$(dirname "$0")/.."
NS=${1:-16}; NW=${2:-8}
L=~/fam; mkdir -p "$L" runs
export CUBLAS_WORKSPACE_CONFIG=:4096:8 POD_DETERMINISTIC=1 HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
WAN_ARGS=${WAN_ARGS:-}   # e.g. "--frames 5" or "--size 320x576", chosen from the timing run

pack() {
  tar -czf ~/fam_results.tmp.tgz --exclude '*.tmp.npz' \
    $(ls -d runs/sdxl/spec.json runs/sdxl/checks.jsonl runs/sdxl/img runs/wan/spec.json runs/wan/checks.jsonl runs/wan/img 2>/dev/null) \
    $(ls runs/sdxl/traj/cuda-fp32__honest__*.npz 2>/dev/null) $(ls runs/wan/traj/cuda-fp32__honest__0[0-3].npz 2>/dev/null) \
    $(ls runs/sdxl_summary.txt runs/wan_summary.txt 2>/dev/null) 2>/dev/null
  mv ~/fam_results.tmp.tgz ~/fam_results.tgz
}

# Wan text embeddings first, alone: the UMT5 encoder needs ~23 GB of the host's 32 GB while it runs
$PY scripts/family_study.py --family wan --role text --prompts "$NW" $WAN_ARGS > "$L/wan_text.out" 2>&1; echo "wan text $?" >> "$L/log"

# CPU chain (background): the second FP32 stack for SDXL, then for Wan
(
  until [ "$(ls runs/sdxl/traj/cuda-fp32__honest__*.npz 2>/dev/null | grep -vc tmp)" -ge "$NS" ]; do sleep 30; done
  OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= $PY scripts/family_study.py --family sdxl --role verify --verifier cpu-fp32 \
    --tag x86cpu-fp32 --k 4 --prompts "$NS" > "$L/sdxl_cpu.out" 2>&1; echo "sdxl cpu $?" >> "$L/log"
  until [ "$(ls runs/wan/traj/cuda-fp32__honest__*.npz 2>/dev/null | grep -vc tmp)" -ge "$NW" ]; do sleep 30; done
  OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= $PY scripts/family_study.py --family wan --role verify --verifier cpu-fp32 \
    --tag x86cpu-fp32 --k 2 --prompts "$NW" $WAN_ARGS > "$L/wan_cpu.out" 2>&1; echo "wan cpu $?" >> "$L/log"
  touch "$L/cpu.done"
) &

# GPU chain (each study renders its honest FP32 jobs first; the CPU chain waits for those files)
$PY scripts/family_study.py --family sdxl --role gpu --prompts "$NS" > "$L/sdxl_gpu.out" 2>&1; echo "sdxl gpu $?" >> "$L/log"
$PY scripts/analyze_study.py sdxl > runs/sdxl_summary.txt 2>&1
pack; touch "$L/sdxl.done"

$PY scripts/family_study.py --family wan --role gpu --prompts "$NW" $WAN_ARGS > "$L/wan_gpu.out" 2>&1; echo "wan gpu $?" >> "$L/log"
$PY scripts/analyze_study.py wan > runs/wan_summary.txt 2>&1
pack; touch "$L/wan.done"

wait
$PY scripts/analyze_study.py sdxl > runs/sdxl_summary.txt 2>&1
$PY scripts/analyze_study.py wan > runs/wan_summary.txt 2>&1
pack; touch "$L/all.done"
echo "ALL DONE $(date -u +%H:%M:%S)" >> "$L/log"
