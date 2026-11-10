"""Download the model files PoD needs into the local Hugging Face cache (safetensors only).

    python scripts/fetch_models.py            # SD1.5 base + BK-SDM substitutes (about 6 GB)
    python scripts/fetch_models.py --base-only

After this, all PoD code runs offline. Licenses: SD1.5 and BK-SDM are CreativeML OpenRAIL-M;
by downloading you accept their terms. No weights are redistributed with this repository.
"""
import argparse
import os

os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import snapshot_download  # noqa: E402

BASE = os.environ.get("POD_BASE_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5")
BASE_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
              "text_encoder/model.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
              "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
SUBS = ["nota-ai/bk-sdm-small", "nota-ai/bk-sdm-tiny"]
SUB_FILES = ["unet/config.json", "unet/diffusion_pytorch_model.safetensors", "model_index.json"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-only", action="store_true")
    a = ap.parse_args()
    print("base:", snapshot_download(BASE, allow_patterns=BASE_FILES))
    if not a.base_only:
        for repo in SUBS:
            print(repo, snapshot_download(repo, allow_patterns=SUB_FILES))
