# Proof-of-Denoising (PoD)

**Spot-checking diffusion inference on GPUs you don't own.**

Image generation increasingly runs on hardware the requester does not control: GPU marketplaces, decentralized networks, third-party APIs and other users' devices. A paid worker can quietly cut corners:
- substitute a smaller model or quantize its weights;
- cache or skip denoising steps;
- drop classifier-free guidance or a required safety prompt;
- run in a lower precision than the job specifies.

Re-running the job costs as much as the job. It also compares outputs that already differ between honest GPUs.

PoD relies on one property: **a diffusion sampler is a Markov chain in its latent**.
1. The worker pins the sampler and a precision class, and commits (a Merkle root) to the job header and every latent z_0 … z_N.
2. After the commitment, the verifier draws k random steps and asks the worker to open them.
3. The verifier recomputes each opened step on its own device. It accepts if r_i = ‖Step(z_i) − z_{i+1}‖ / ‖Step(z_i) − z_i‖ stays under a per-step threshold τ_i, calibrated on honest cross-device drift.
4. A light verifier (for example a browser) escalates any flag to a reference GPU. The reference must confirm the flag before a worker is rejected.

Checking k of N steps costs k/N of the network evaluations (2 of 25 = 8%). End-to-end cost also includes text encoding, state transfer, hashing and model loading.

## Results

`scripts/export_macros_cvpr.py` regenerates every number below from the logs in `runs/`. The setup is SD1.5 at 512², DDIM with 25 steps and FP32 jobs, unless noted. Stacks: NVIDIA T4, L4 and A100 (CUDA; TensorRT on the L4), an Apple GPU and CPU, an x86 CPU, and WebGPU in a browser.

| Question | Result |
|---|---|
| How far do honest GPUs drift? | Apple GPU ↔ NVIDIA L4, FP32: median 9.2×10^-6, max 4.8×10^-5 per step. FP32 thresholds pool 5 stacks. |
| Do the thresholds hold on GPUs they were not calibrated on? | A T4 and an A100 joined after the thresholds were frozen. Across 8 worker→verifier pairs, 0 of 9,600 honest FP32 steps were flagged (largest r/τ: 0.29). As verifiers of the L4's shortcut jobs they catch 7 of 9 and 7 of 9. |
| How often are honest workers flagged? (held out per prompt) | Native FP32 stacks: 0 of 2,680 checks. Browser (WebGPU): 2 of 1,200 checks, all cleared by the L4 re-check. |
| Which shortcuts does k = 2 catch with P ≥ 0.99? | 7 of 9 on the L4, 7 of 9 on the Apple GPU, 7 of 9 in the browser. |
| Are shortcuts detectable on the steps they change? | Across 2,608 all-step renders (2,160 of them on 240 prompts outside the study), every changed step exceeded τ at least 24-fold, and no unchanged step exceeded it. A node saving a fraction f of the compute is therefore caught with P ≥ f. |
| How does it compare with replaying whole jobs at the same budget? | Replaying a random 8% of jobs catches a deviating job 8% of the time. Even a full replay that compares final latents detects TF32 run as FP32 on only 29% of jobs. |
| Is a silent precision downgrade caught? | A TF32 or FP16 run claimed as FP32 was flagged on 1,248/1,248 steps. TF32 drifts 105× more than cross-device FP32 and saves 35%. |
| Does it hold on unseen, pre-registered pages? | 16/16 honest pages accepted, 16/16 deviating pages rejected (14.3 expected). |
| What does request-level split conformal give (α = 1%)? | 0 of 32 held-out jobs flagged between the same two stacks; 7 of 9 shortcuts still caught. |
| What can a full-cost attacker do inside τ? | A fixed steering direction moved the final latent at most 3.9% of the way to a target (48 prompts). Gradient-optimized steering is about 27× stronger: at 95% of the tolerance the image falls below 30 dB on 12 of 48 prompts (worst 17.8 dB), yet it covers at most 4.9% of the distance to a chosen target. A verifier on the worker's own stack is bit-exact and leaves no such budget. |
| What do sampled checks miss? | Splicing one step replaces the image (median PSNR 9 dB to the honest render), yet k checks catch it only with P = k/N. |
| Does it transfer to other models? | **SDXL at 1024²** (16 prompts): 7 of 9 shortcuts caught at k = 2; TF32 and FP16 runs flagged on 400/400 and 400/400 steps; 0 of 1,264 honest steps flagged. **SANA-1.6B** (flow matching, 24 prompts): 6 of 8 caught; BF16 and TF32 runs flagged on 600/600 and 600/600 steps; 4 of 600 honest steps flagged. **Wan 2.1 text-to-video** (33 frames, 8 prompts): 5 of 7 caught; TF32 and BF16 runs flagged on 200/200 and 200/200 steps; 6 of 219 honest steps flagged. The honest flags on SANA and Wan come from calibrating on few prompts. |
| How long does a check take? | On the NVIDIA L4, rendering a page takes 5.7 s and checking 2 steps takes 0.5 s. A laptop browser takes 4.2 s. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_models.py      # SD1.5 + BK-SDM UNets, safetensors only (~6 GB)
python -m unittest discover -s tests -t .
```

After `fetch_models.py`, everything runs offline; weights load only from the local cache. SANA, SDXL and Wan 2.1 are fetched on first use (`podlab/flow.py`, `podlab/sdxl.py`; set `HF_HUB_OFFLINE=0`).

## Use the protocol

```python
from podlab.model import Backend, Denoiser
from podlab.protocol import Job, run_and_commit, draw_challenges, open_steps, verify_openings

job = Job(prompt="a red bicycle leaning against a brick wall", negative="nsfw, gore", seed=7, precision_class="fp32")

# worker (any device)
states, tree = run_and_commit(Denoiser(Backend("cuda", "fp32")), job)          # publish tree.root

# verifier (another device), challenges drawn after the root is published
idx = draw_challenges(job.spec.steps, k=2)
ok, checks = verify_openings(Denoiser(Backend("cpu", "fp32")), job, tree.root,
                             open_steps(states, tree, idx), taus=my_fp32_thresholds,
                             challenges=idx, header_path=tree.path(0),
                             initial_state=states[0], initial_path=tree.path(1))
```

Follow these rules when you use the protocol:
- Keep the published root and challenge list on the verifier. Never replace either with values from the worker's response. The verifier rejects missing, duplicate or substituted challenges, and always verifies the header and seed proofs.
- Use seeded challenge generators only for reproducible experiments.
- Resolve `job.model_id` to trusted local weights before verifying. The API does not authenticate the denoiser object.
- **Pin the precision class in the job.** Honest FP16 drift is far larger than cross-device FP32 drift. A worker that could claim the loosest class would hide inside its tolerance.

Calibrate `taus` (per-step thresholds for the job's precision class) on honest cross-device runs. Two rules are included: the empirical maximum-times-margin rule (`scripts/analyze_study.py`) and request-level split conformal (`podlab/audit_policy.py`, `scripts/conformal_requests.py`).

## Reproduce

All scripts resume and skip finished outputs. Results go to `runs/`, figures and LaTeX tables to `figures/`. Scripts that draw images or re-check trajectories need the trajectories from your own run; this repository ships the logs, not the latents.

| What | Command |
|---|---|
| Pilot on one laptop (Apple GPU, CPU, browser) | `python scripts/pilot.py && python scripts/analyze_pilot.py && python scripts/output_level.py` |
| Study on Apple silicon | `python scripts/study.py --role mps` and, in parallel, `--role cpu` |
| NVIDIA: CUDA FP32/TF32/FP16, TensorRT FP16, cross-vendor | `bash reproduce/run_nvidia.sh` (see `reproduce/README.md`) |
| NVIDIA: calibration prompts, all-step checks, attacks, SANA, storybook | `bash reproduce/run_cvpr.sh`, then `bash reproduce/run_extended.sh` |
| Pooled analysis (held out per prompt) | `python scripts/analyze_study.py <pool>`; a pool is a `runs/` folder name such as `study+calib_webgpu-fp32+…` |
| Shortcut margins, detection vs cost, step masks | `python scripts/dense_margin.py`, `scripts/make_cost_fig.py`, `scripts/make_mask_fig.py` |
| Full-cost attacks | Fixed steering and splicing: `python scripts/adaptive_attacks.py`, `scripts/attack_xverify.py`, `scripts/make_attack_fig.py`. Optimized steering: `python scripts/optimal_steer.py --device cuda --prompts 48`, a second stack with `scripts/cross_verify.py --traj-run adaptive_opt …`, then `scripts/attack_xverify.py --run <that run> --out adaptive_opt/xverify_summary.json` and `scripts/make_optattack_fig.py` |
| Request-level conformal | `python scripts/conformal_requests.py --alpha 0.01 --fit 40` |
| Further models | SANA: `python scripts/sana_study.py`, then `scripts/sana_report.py`. SDXL and Wan 2.1: `bash reproduce/run_families.sh`, then `python scripts/sana_report.py sdxl` and `python scripts/sana_report.py wan` |
| Further NVIDIA GPUs under frozen thresholds | `bash reproduce/run_more_gpus.sh <tag>` on each GPU, then `python scripts/xgpu_report.py` |
| Nine shortcuts on 240 further prompts, every step recomputed | `python scripts/shortcut_scale.py --device cuda`, then `scripts/scale_report.py` |
| Replay baseline; what each deviation saves and how visible it is | `python scripts/baseline_compare.py`, `scripts/perceptual_table.py` |
| Every number above, as LaTeX macros and tables | `python scripts/export_macros_cvpr.py` |

### Storybook case study

The case study renders an illustrated story whose pages are not part of this release. Its logs are included (page ids, which deviation ran on each page, per-step ratios), so its figures and numbers regenerate. The pages' text, and the pre-registration record that hash-stamps it, are not included, so `scripts/fresh_eval.py` documents the frozen evaluation but cannot run here. To run the case study on your own pages, write them in the format of `prompts/storybook_example.json` and pass `--story`.

### In-browser WebGPU backend

1. Export the model: `python scripts/export_onnx.py` writes `runs/onnx/unet_fp32_single.onnx` (+ `.data`) and the text encoder. For an FP16 graph, run `python scripts/export_unet_fp16.py`.
2. Install the runtime: `cd web && npm install`.
3. Run the harness: `python scripts/browser_backend.py --task smoke --port 8092`.
4. Open `http://127.0.0.1:8092/?unet=unet_fp32_single.onnx&ext=1` in a WebGPU browser.

In FP32 the browser joins the FP32 class. The sampler arithmetic, commitments and decisions run in the local Python host; the browser runs the network.

## What is in here

| Path | What |
|---|---|
| `podlab/sampler.py` | Explicit Markov DDIM sampler, matching diffusers to 1e-5. Float64 update with FP32 stored states; Philox noise keyed by (seed, step) |
| `podlab/flow.py` | Flow-matching Euler sampler for SANA and for Wan 2.1 (video latents) |
| `podlab/sdxl.py` | SDXL on the DDIM path: two text encoders, 1024², distilled substitutes SSD-1B and Segmind-Vega |
| `podlab/model.py` | Text encoder, UNet and VAE on `cpu`, `mps` or `cuda` in FP16, TF32 or FP32; substitute UNets; simulated weight quantization |
| `podlab/trt.py` | TensorRT FP16 engine for the UNet (optional) |
| `podlab/commit.py` | SHA-256 Merkle commitment over the job header and every latent |
| `podlab/protocol.py` | Commit → challenge → open → verify, with hardened openings |
| `podlab/audit_policy.py` | Request-level calibration and exact audit-budget planning |
| `podlab/workers.py` | Honest worker and deviations: substitution, quantization, CFG skip, caching, step skipping, safety-prompt stripping |
| `podlab/verify.py` | Step recomputation and the test statistic r_i |
| `scripts/` | Experiments, analyses, figures, ONNX export, browser harness, model fetching |
| `reproduce/` | The shell scripts that ran the NVIDIA experiments |
| `runs/` | The JSON/JSONL logs behind every reported number (no trajectories or images) |
| `web/` | WebGPU page (ONNX Runtime Web) |
| `prompts/` | PartiPrompts samples: 48 for the study, 240 disjoint ones for calibration (Apache-2.0) |

## Threat model and limits

**In scope:** rational workers who deviate to save compute or to remove a safety component, through substitution, quantization, guidance skipping, caching, step skipping, safety-prompt stripping or a silent precision downgrade.

**Required:** challenges must be fresh, drawn *after* the commitment. Predictable or hash-derived challenges are evaded.

**Out of scope:** worker–verifier collusion, side channels and prompt privacy (the verifier sees the prompt and the opened latents).

**Limits:**
- Step checks secure the *computation*, not the output. A worker that pays full cost and splices one step is caught only with probability k/N.
- The tolerance that absorbs honest cross-stack drift is also an attacker's budget. Gradient-optimized steering inside it visibly alters details of the image on about a quarter of our prompts, although it cannot choose the result. A verifier on the worker's own stack is bit-exact and needs no tolerance.
- Thresholds are empirical, and each stack pair needs its own calibration.
- Few-step (1–4 step) models cannot be verified cheaply, because cost scales as k/N.
- The prototype does not bind delivered image bytes to a verified VAE decode, run an output safety checker or sign a provenance record. Those steps are required before claiming an output attestation.

**Ethics:** we generate only benign prompts. The "safety strip" deviation removes a negative prompt, and no unsafe content is produced.

## License and citation

Apache License 2.0; see `LICENSE` and `NOTICE`. The licence includes an explicit patent grant for this code. Third-party material is listed in `THIRD_PARTY_NOTICES.md`. Model weights are not redistributed; their own licences apply.

A paper describing this work is under review. Until it is published, please cite this repository (`CITATION.cff`).
