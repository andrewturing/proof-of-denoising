#!/usr/bin/env bash
# A further NVIDIA GPU (T4, A100, ...) joins the SD1.5 study as a worker and as a verifier.
#   bash reproduce/run_more_gpus.sh <tag> [low_precision_prompts=48]      e.g. t4, a100   (from the kit's pod/ directory)
# The kit carries the L4's trajectories (runs/nvidia48: honest FP32 / TF32 / FP16 jobs for 48 prompts and the nine
# FP32 shortcut jobs for 16 prompts) and the L4's honest SDXL jobs (runs/sdxl). Every stage resumes.
#   1  setup: Python environment, models, GPU and software versions, step-time benchmark
#   2  honest jobs on this GPU in FP32 (48 prompts), TF32 and FP16     -> runs/xgpu_<tag>/traj/<tag>-<prec>__honest__pp.npz
#   3  this GPU (FP32) re-checks every step of the L4's honest FP32 jobs                       -> runs/xg_<tag>_on_l4
#   4  this GPU (FP32) re-checks its own honest jobs: FP32 (4 steps), TF32 and FP16 (8 steps)    -> runs/xg_<tag>_self
#   5  this GPU (FP32) re-checks the L4's honest TF32 / FP16 jobs (8 steps) and shortcut jobs (6 steps) -> runs/xg_<tag>_on_l4
#   6  SDXL (skip with SDXL=0): every step of the L4's 16 honest FP32 jobs                     -> runs/sdxl/checks_<tag>.jsonl
# Packs ~/xg_<tag>.tgz after each stage (logs plus this GPU's honest FP32 trajectories). Markers and logs: ~/xg/.
# The trajectories may arrive after the code: stage 3 waits for ~/xg/data.ready, stage 6 for ~/xg/sdxl.ready.
set -uo pipefail
cd "$(dirname "$0")/.."
T=${1:?tag, e.g. t4}; NLOW=${2:-48}
L=~/xg; mkdir -p "$L" runs/xgpu_$T
export CUBLAS_WORKSPACE_CONFIG=:4096:8 POD_DETERMINISTIC=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
exec >> "$L/run.out" 2>&1
step() { echo; echo "=================== $* ($(date -u +%H:%M:%S) UTC)"; echo "$* $(date -u +%H:%M:%S)" >> "$L/log"; }
pack() {
  tar -czf ~/xg_$T.tmp.tgz $(ls -d runs/xgpu_$T/spec.json runs/xgpu_$T/gpu.txt runs/xgpu_$T/torch_env.txt runs/xgpu_$T/bench.json \
      runs/xg_${T}_on_l4 runs/xg_${T}_self runs/xg_${T}_on_* runs/sdxl/checks_$T.jsonl 2>/dev/null | sort -u) \
      $(ls runs/xgpu_$T/traj/$T-fp32__honest__*.npz 2>/dev/null) 2>/dev/null
  mv ~/xg_$T.tmp.tgz ~/xg_$T.tgz
}

step "1 setup"
until nvidia-smi > /dev/null 2>&1; do echo "waiting for the NVIDIA driver"; sleep 20; done
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv > runs/xgpu_$T/gpu.txt
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  python3 -m venv .venv 2>/dev/null || { sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv && python3 -m venv .venv; }
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q torch
  $PY -m pip install -q "diffusers>=0.30" "transformers>=4.44" huggingface_hub safetensors accelerate numpy pillow
fi
$PY -c "import torch, diffusers, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)" | tee runs/xgpu_$T/torch_env.txt
$PY scripts/fetch_models.py || exit 1
[ -f runs/xgpu_$T/bench.json ] || $PY scripts/bench_backends.py --backends cuda-fp16,cuda-tf32,cuda-fp32 --out runs/xgpu_$T/bench.json

step "2 honest jobs on this GPU"
$PY scripts/gen_honest.py --backend cuda-fp32 --name $T-fp32 --from 0 --to 48 --run-name xgpu_$T
pack; touch "$L/honest.done"
$PY scripts/gen_honest.py --backend cuda-tf32 --name $T-tf32 --from 0 --to "$NLOW" --run-name xgpu_$T
$PY scripts/gen_honest.py --backend cuda-fp16 --name $T-fp16 --from 0 --to "$NLOW" --run-name xgpu_$T

until [ -f "$L/data.ready" ]; do sleep 15; done
step "3 every step of the L4's honest FP32 jobs"
$PY scripts/cross_verify.py --traj-run nvidia48 --verifier cuda-fp32 --tag $T-fp32 --k 25 --only-kinds honest --only-workers cuda-fp32 --out-run xg_${T}_on_l4
pack

step "4 this GPU's own honest jobs"
$PY scripts/cross_verify.py --traj-run xgpu_$T --verifier cuda-fp32 --tag $T-fp32 --k 4 --only-workers $T-fp32 --out-run xg_${T}_self
$PY scripts/cross_verify.py --traj-run xgpu_$T --verifier cuda-fp32 --tag $T-fp32 --k 8 --only-workers $T-tf32,$T-fp16 --out-run xg_${T}_self
pack

step "5 the L4's lower-precision and shortcut jobs"
$PY scripts/cross_verify.py --traj-run nvidia48 --verifier cuda-fp32 --tag $T-fp32 --k 8 --only-kinds honest --only-workers cuda-tf32,cuda-fp16 --out-run xg_${T}_on_l4
$PY scripts/cross_verify.py --traj-run nvidia48 --verifier cuda-fp32 --tag $T-fp32 --k 6 --only-workers cuda-fp32 \
    --only-kinds sub-bk-sdm-small,sub-bk-sdm-tiny,w8,w4,cfg-skip,cache2,cache2-late,step-skip,neg-strip --out-run xg_${T}_on_l4
pack; touch "$L/sd15.done"

n=0; until [ -f "$L/sdxl.ready" ] || [ $n -gt 40 ]; do sleep 15; n=$((n+1)); done
if [ "${SDXL:-1}" = 1 ] && [ -f "$L/sdxl.ready" ]; then
  step "6 SDXL: every step of the L4's honest FP32 jobs"
  HF_HUB_OFFLINE=0 $PY scripts/family_study.py --family sdxl --role verify --verifier cuda-fp32 --tag $T-fp32 --k 25 --prompts 16 --checks checks_$T.jsonl
  pack
fi
touch "$L/all.done"; step "done"
