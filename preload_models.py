"""Preload models for the experimental AI kit.

This script downloads the required Hugging Face model checkpoints ahead of
time so that inference is ready to use on first launch.  It relies on
`huggingface_hub.snapshot_download` to fetch and cache all files for a
repository into the directory specified by the HF_HOME environment
variable.  Downloading models in advance reduces cold‑start latency and
avoids repeated downloads when the pod restarts【486638170667719†L503-L549】.

Environment variables:
    HF_HOME (str): Root directory for the Hugging Face cache.  Defaults
        to `~/.cache/huggingface` if unset.  When running inside our
        Docker image, HF_HOME points to `/data/hf_cache`, which should
        be mounted on a persistent volume.
    HUGGING_FACE_HUB_TOKEN (str): Optional token for downloading gated
        models.  Set this if you need access to private repos.
"""

import os
from typing import List

from huggingface_hub import snapshot_download


def preload(repo_id: str) -> None:
    """Download a model repository from Hugging Face Hub.

    Args:
        repo_id: The repository identifier, e.g. "Wan-AI/Wan2.2-T2V-5B".
    """
    print(f"Downloading {repo_id}…")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=os.environ.get("HF_HOME"),
        token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        local_files_only=False,
        ignore_patterns=["*.msgpack"]  # skip accelerated weight formats we don't use
    )
    print(f"Finished downloading {repo_id}.")


def main() -> None:
    # List of models referenced in app.py.  Feel free to extend this list
    # if you add more models to the interface.
    models: List[str] = [
        # Wan 2.2 models (text→video, image→video, text+image→video).  Both
        # 5B and 14B versions are included so you can switch sizes without
        # additional downloads.  14B models require ~24 GB VRAM【526278201854702†L305-L310】.
        "Wan-AI/Wan2.2-T2V-5B",
        "Wan-AI/Wan2.2-T2V-A14B",
        "Wan-AI/Wan2.2-I2V-5B",
        "Wan-AI/Wan2.2-I2V-A14B",
        "Wan-AI/Wan2.2-TI2V-5B",
        "Wan-AI/Wan2.2-TI2V-A14B",
        # Flux image model for text→image generation
        "black-forest-labs/FLUX.1-dev",
        # Qwen image model for alternative text→image generation
        "Qwen/Qwen-Image",
    ]
    for model in models:
        preload(model)
    print("All requested models are downloaded.")


if __name__ == "__main__":
    main()