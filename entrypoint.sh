#!/usr/bin/env bash
# Entrypoint script for the experimental AI kit container
#
# This script ensures the Gradio app always launches on port 7860
# and optionally pre-downloads models if requested

set -euo pipefail

# Set default port if not specified
export PORT=${PORT:-7860}

# If a preload flag is passed via environment, run the preload script
if [[ -n "${PRELOAD_MODELS:-}" ]]; then
    echo "[entrypoint] Preloading models..."
    python preload_models.py
fi

# Always launch the Gradio app on the specified port
echo "[entrypoint] Launching Gradio on port $PORT..."
exec python app.py
