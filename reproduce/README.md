# Reproducing the NVIDIA experiments

These scripts produced the NVIDIA results on a single NVIDIA L4 (24 GB). Any Linux machine with a recent NVIDIA GPU
and CUDA drivers should work. TensorRT steps are skipped when TensorRT is unavailable. Run the scripts from the
repository root, in this order:

| Script | What it runs |
|---|---|
| `run_nvidia.sh` | Environment and models; timing for CUDA FP16/TF32/FP32, TensorRT FP16 and CPU; the study on 8 prompts (honest FP16/TF32/FP32 plus the deviations); cross-vendor checks of the Apple GPU, CPU and browser trajectories; the storybook pages; analysis. Writes `nvidia_results.tgz`. Takes ~30–40 min on an L4. |
| `run_cvpr.sh` | The 48-prompt study on CUDA; every step of the Apple-GPU honest trajectories checked on CUDA FP32, TF32 and TensorRT FP16; an x86 CPU as a second FP32 verifier for all 48 honest CUDA trajectories. |
| `run_extended.sh` | SANA on the GPU and the x86 CPU; the unseen storybook chapters; 240 disjoint calibration prompts; full-cost attacks (steering, splicing); the all-step check of every deviating trajectory. |
| `run_families.sh` | Two further models on a 24 GB GPU: SDXL at 1024² (substitutes SSD-1B and Segmind-Vega) and Wan 2.1 text-to-video (33 frames at 480×832). Honest runs in every precision class, every deviation in FP32, all checked on the GPU; the host CPU re-checks a few steps of the honest jobs. Wan's text encoder runs once on the CPU and needs about 23 GB of RAM. Takes ~4 h for SDXL and ~15 h for Wan on an L4. |
| `run_more_gpus.sh <tag>` | Another NVIDIA GPU (we used a T4 and an A100) joins as a worker and as a verifier: it renders the 48 honest jobs, then re-checks the first GPU's honest, lower-precision and shortcut jobs and its honest SDXL jobs. It needs the first GPU's trajectories under `runs/nvidia48/traj` and `runs/sdxl/traj`. `scripts/xgpu_report.py` then judges every check with thresholds frozen beforehand. |

Every stage resumes where it stopped. For bit-exact reruns on the same GPU, the scripts set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `POD_DETERMINISTIC=1`.

When a run finishes, copy its results into `runs/` on your analysis machine and run the analysis commands in the main
README.
