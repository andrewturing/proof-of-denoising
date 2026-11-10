#!/usr/bin/env bash
# Proof-of-Denoising: the NVIDIA run for the GTC 2027 poster. One command on any Linux machine
# with an NVIDIA GPU (Colab/Kaggle notebook terminal, RunPod/Lambda/Vast VM, cloud VM).
#
#   bash reproduce/run_nvidia.sh            # from the unpacked kit's pod/ directory
#
# Produces nvidia_results.tgz (bring it back to the Mac). Takes ~30-40 min on an L4/A10/4090,
# ~1-1.5 h on a T4. Optional steps (TensorRT) are skipped automatically if unavailable.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
LOG=runs/nvidia_kit.log
export CUBLAS_WORKSPACE_CONFIG=:4096:8 POD_DETERMINISTIC=1   # bit-identical same-GPU reruns
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=================== $* ($(date -u +%H:%M:%S) UTC)"; }

step "GPU"
nvidia-smi || { echo "no NVIDIA GPU visible"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv > runs/gpu.txt

step "Python environment"
PY=python3
if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  $PY -m venv .venv 2>/dev/null || { sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv && $PY -m venv .venv; }
  PY=.venv/bin/python
  $PY -m pip install -q --upgrade pip
  # torch-tensorrt first, so pip chooses a matching torch (PyPI Linux wheels bundle CUDA 12.x);
  # if that fails, plain torch. TensorRT steps are skipped automatically when it is missing.
  $PY -m pip install -q torch-tensorrt || echo "torch-tensorrt not installable; continuing without TensorRT"
  $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || $PY -m pip install -q torch
fi
$PY -m pip install -q "diffusers>=0.30" "transformers>=4.44" huggingface_hub safetensors accelerate numpy pillow matplotlib
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" | tee runs/torch_env.txt
$PY -c "import tensorrt; print('tensorrt', tensorrt.__version__)" 2>/dev/null | tee -a runs/torch_env.txt || true
$PY -m unittest discover -s tests -t . || { echo "unit tests failed"; exit 1; }

step "Models (SD1.5 + BK-SDM UNets, safetensors)"
$PY scripts/fetch_models.py || exit 1

step "1/6 Benchmark: CUDA FP16 / TF32 / FP32, TensorRT FP16, CPU"
$PY scripts/bench_backends.py --backends cuda-fp16,cuda-tf32,cuda-fp32,trt-fp16,cpu-fp32 --out runs/bench_nvidia.json

step "2/6 Study on CUDA: honest FP16/TF32/FP32 + 18 cheats, 8 prompts, verified on cuda-fp32"
$PY scripts/study.py --role gpu --device cuda --run-name nvidia --prompts 8

step "3/6 Cross-vendor: verify the Apple-GPU/CPU/browser trajectories on NVIDIA"
$PY scripts/cross_verify.py --traj-run study --verifier cuda-fp32 --k 8 --out-run cross_nvidia
$PY scripts/cross_verify.py --traj-run study --verifier cuda-tf32 --k 4 --only-kinds honest --out-run cross_nvidia
$PY scripts/cross_verify.py --traj-run study --verifier trt-fp16 --k 4 --only-kinds honest --out-run cross_nvidia || echo "TensorRT cross-verify skipped"

step "4/6 Storybook case study: NVIDIA edge node renders the chapter; verify on CUDA"
$PY scripts/storybook_case.py --run storybook_cuda --device cuda --stages taus,render,decode,verify --verifier cuda-fp32
$PY scripts/storybook_case.py --run storybook --stages verify --verifier cuda-fp32 || true

step "5/6 Analysis"
$PY scripts/analyze_study.py nvidia | tee runs/nvidia_summary.txt
$PY scripts/analyze_study.py cross_nvidia | tee runs/cross_nvidia_summary.txt

step "6/6 Pack results"
tar -czf nvidia_results.tgz runs/gpu.txt runs/torch_env.txt runs/bench_nvidia.json runs/nvidia runs/cross_nvidia runs/storybook_cuda \
  runs/storybook/verify_cuda-fp32.jsonl runs/nvidia_summary.txt runs/cross_nvidia_summary.txt runs/nvidia_kit.log 2>/dev/null
ls -la nvidia_results.tgz && echo "DONE: copy nvidia_results.tgz to your analysis machine and untar it in the repository root."
