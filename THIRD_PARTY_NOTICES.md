# Third-party notices

This repository's code is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`). It ships no model weights. It uses, or downloads on request, the following third-party material.

| Component | Use | License | Redistributed here? |
|---|---|---|---|
| PartiPrompts (github.com/google-research/parti) | 48-prompt sample in `prompts/parti_sample.json`; 240 disjoint calibration prompts in `prompts/parti_calib.json` | Apache-2.0 | Yes, derived samples with attribution |
| Stable Diffusion v1.5 weights | Model under test | CreativeML OpenRAIL-M | No; fetched by `scripts/fetch_models.py` |
| BK-SDM-Small / BK-SDM-Tiny UNets (nota-ai) | Substitution cheats | CreativeML OpenRAIL-M | No; fetched by `scripts/fetch_models.py` |
| SANA-1.6B / SANA-0.6B (Efficient-Large-Model, NVlabs) | Second model family and its substitute | Apache-2.0 per the model cards, which also state the models are intended for research; the Gemma-2 text encoder is under the Gemma Terms of Use | No; fetched on first use by `podlab/flow.py` |
| NVIDIA TensorRT | Optional FP16 UNet engine (`podlab/trt.py`) | NVIDIA software licence | No; install it yourself |
| ONNX Runtime Web | In-browser WebGPU backend | MIT | No; installed with `npm install` in `web/` |
| PyTorch, diffusers, transformers, huggingface_hub, ONNX, ONNX Runtime | Python dependencies | BSD-3 / Apache-2.0 / MIT | No; installed with pip |

**Sources:**
- PartiPrompts: Yu et al., "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation", 2022.
- BK-SDM: Kim et al., ECCV 2024.

**Model licence terms.** Using the model weights is subject to their licences. These restrict harmful uses. This research generates only benign prompts. Its "safety strip" cheat removes a *negative* prompt, and the study never generates unsafe content.
