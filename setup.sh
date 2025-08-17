#!/usr/bin/env bash
#
# setup.sh — helper script to build and run the experimental AI kit locally
#
# This script streamlines building the Docker image, creating a
# persistent volume for model caches and outputs, and running the
# container.  It is intended for local testing or for environments
# where you control Docker directly, such as a Runpod pod with
# sudo privileges.  When used on Runpod, you can copy and modify the
# docker run invocation below in your template settings.
#
# USAGE:
#   bash setup.sh [--build] [--run] [--models]
#
# Options:
#   --build   Build the Docker image (tagged as experimental-ai-kit).
#   --run     Launch a container with GPU support, persistent cache and open port 7860.
#   --models  Pre‑download models into the cache before first run.  Requires internet access.
#
# The script uses environment variables for optional configuration:
#   HF_HOME, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE
#       Override the default cache location (/data/hf_cache).  When not set,
#       the Docker image defaults to /data/hf_cache — ensure you mount a volume
#       at /data to make the cache persistent【486638170667719†L503-L549】.
#   HUGGING_FACE_HUB_TOKEN
#       Your personal Hugging Face token (optional).  Needed to download
#       private or gated models.  Export it before running if required.
#   GRADIO_AUTH
#       Comma‑separated "username:password" for basic authentication on the UI.
#       Leave unset for open access (not recommended on public pods).

set -eo pipefail

IMAGE_NAME="experimental-ai-kit"
VOLUME_NAME="experimental_kit_data"

build_image() {
    echo "[+] Building Docker image as $IMAGE_NAME..."
    docker build -t "$IMAGE_NAME" -f Dockerfile .
    echo "[+] Build complete."
}

create_volume() {
    if ! docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1; then
        echo "[+] Creating persistent volume $VOLUME_NAME for cache and outputs..."
        docker volume create "$VOLUME_NAME"
    else
        echo "[+] Volume $VOLUME_NAME already exists."
    fi
}

run_container() {
    create_volume
    echo "[+] Launching container..."
    # Note: `--gpus all` requires the NVIDIA Container Runtime on the host
    docker run --gpus all --rm -it \
        -p 7860:7860 \
        -v "$VOLUME_NAME":/data \
        -e HF_HOME="${HF_HOME:-/data/hf_cache}" \
        -e HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/hf_cache/hub}" \
        -e TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/hf_cache/hub}" \
        -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
        -e GRADIO_AUTH="${GRADIO_AUTH:-}" \
        "$IMAGE_NAME"
}

preload_models() {
    echo "[+] Pre‑downloading models into the cache..."
    # Ensure the image is built before running preload script
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "Image $IMAGE_NAME not found.  Building it first."
        build_image
    fi
    create_volume
    # Run the preload script in a temporary container with mounted cache
    docker run --gpus all --rm \
        -v "$VOLUME_NAME":/data \
        -e HF_HOME="${HF_HOME:-/data/hf_cache}" \
        -e HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/hf_cache/hub}" \
        -e TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/hf_cache/hub}" \
        -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
        "$IMAGE_NAME" \
        python preload_models.py
    echo "[+] Model download complete."
}

# Parse arguments
while (( "$#" )); do
    case "$1" in
        --build)
            ACTION_BUILD=1
            shift
            ;;
        --run)
            ACTION_RUN=1
            shift
            ;;
        --models)
            ACTION_MODELS=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "${ACTION_BUILD:-}" ]]; then
    build_image
fi

if [[ -n "${ACTION_MODELS:-}" ]]; then
    preload_models
fi

if [[ -n "${ACTION_RUN:-}" ]]; then
    run_container
fi

if [[ -z "${ACTION_BUILD:-}" && -z "${ACTION_RUN:-}" && -z "${ACTION_MODELS:-}" ]]; then
    echo "No action specified.  Available options: --build, --run, --models" >&2
    exit 1
fi