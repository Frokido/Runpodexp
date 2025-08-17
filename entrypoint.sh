#!/usr/bin/env bash
# Entrypoint script for the experimental AI kit container
#
# This script optionally pre‑downloads models if the environment
# variable PRELOAD_MODELS is set to a non‑empty value.  It then
# launches the Gradio application.

set -euo pipefail

# If a preload flag is passed via environment, run the preload script.
if [[ -n "${PRELOAD_MODELS:-}" ]]; then
    echo "[entrypoint] Preloading models..."
    python preload_models.py
fi

exec python app.py