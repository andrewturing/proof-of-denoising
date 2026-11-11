#!/usr/bin/env bash
# CVPR-scale run on an NVIDIA VM (after run_nvidia.sh has set up .venv and the models).
#   bash reproduce/run_cvpr.sh          (from the kit's pod/ directory)
# 1. Full 48-prompt PartiPrompts study on CUDA: honest FP16/TF32/FP32 + 18 cheats; cheats checked on
#    12 random steps, honest on all 25 (verifier cuda-fp32).
# 2. Dense cross-vendor calibration: every step of the Mac's honest FP32 trajectories on NVIDIA FP32,
#    TF32 and TensorRT FP16.
# 3. A second FP32 stack on the same honest jobs: the VM's x86 CPU checks every step of the 48 honest
#    CUDA-FP32 trajectories (logged as x86cpu-fp32, distinct from the Mac's cpu-fp32). This gives
#    the held-out false-accusation test at CVPR scale.
# Packs after each stage (cvpr_results.tgz), so a partial run is still usable. Honest CUDA-FP32
# trajectories ship too (~80 MB) for the reverse direction (verified on the Mac's GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 POD_DETERMINISTIC=1
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
LOG=runs/cvpr_vm.log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=================== $* ($(date -u +%H:%M:%S) UTC)"; }
pack() {
  tar -czf cvpr_results.tmp.tgz --exclude '*.npz' runs/nvidia48 runs/xcal_cuda_fp32 runs/xcal_cuda_tf32 \
      runs/xcal_trt_fp16 runs/xcpu_nvidia48 runs/nvidia48_summary.txt runs/xcpu_nvidia48_summary.txt runs/cvpr_vm.log \
      $(ls runs/nvidia48/traj/cuda-fp32__honest__*.npz 2>/dev/null) 2>/dev/null
  mv cvpr_results.tmp.tgz cvpr_results.tgz && ls -la cvpr_results.tgz
}

step "1/3 Study: 48 prompts on CUDA"
POD_K_RANDOM=12 $PY scripts/study.py --role gpu --device cuda --run-name nvidia48 --prompts 48
$PY scripts/analyze_study.py nvidia48 | tee runs/nvidia48_summary.txt
pack

step "2/3 Dense cross-vendor calibration (all steps of the Mac's honest FP32 trajectories)"
$PY scripts/cross_verify.py --traj-run study --verifier cuda-fp32 --k 25 --only-kinds honest --out-run xcal_cuda_fp32
$PY scripts/cross_verify.py --traj-run study --verifier cuda-tf32 --k 25 --only-kinds honest --out-run xcal_cuda_tf32
$PY scripts/cross_verify.py --traj-run study --verifier trt-fp16 --k 25 --only-kinds honest --out-run xcal_trt_fp16 || echo "TensorRT calibration skipped"
pack

step "3/3 x86 CPU FP32 verifies every step of the honest CUDA-FP32 jobs"
$PY scripts/cross_verify.py --traj-run nvidia48 --verifier cpu-fp32 --tag x86cpu-fp32 --k 25 \
    --only-kinds honest --only-workers cuda-fp32 --only-prompts "${XCPU_PROMPTS:-48}" --out-run xcpu_nvidia48
$PY scripts/analyze_study.py xcpu_nvidia48 | tee runs/xcpu_nvidia48_summary.txt
pack
echo "DONE"
