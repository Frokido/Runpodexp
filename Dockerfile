## Dockerfile for the experimental AI kit
#
# This Dockerfile builds a container image capable of running the
# experimental generative‑AI kit on GPU hardware.  It follows
# container best‑practices by using an official PyTorch runtime as
# the base image, installing only the required dependencies, and
# configuring persistent caches for Hugging Face models and datasets.
#
# Key features:
#   * Uses a CUDA‑enabled PyTorch base image so GPUs in Runpod pods
#     (e.g. L30S, L40S) are immediately accessible.  No custom drivers
#     are required.
#   * Installs system libraries for video processing (ffmpeg,
#     libgl1) and cleans up afterwards to keep the final image small.
#   * Sets the HF_HOME environment variable so model weights
#     downloaded from the Hugging Face Hub are written to
#     /data/hf_cache.  When you mount a persistent volume at
#     /data, downloaded weights survive pod restarts【997705316903972†L134-L165】.
#   * Exposes port 7860 for Gradio and defaults GRADIO_SERVER_NAME to
#     0.0.0.0 so the UI is reachable from outside the container.
#   * Leaves authentication off by default; you can set
#     GRADIO_AUTH=username:password in your pod settings for
#     password‑protected access.

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Avoid user interaction during apt installs
ARG DEBIAN_FRONTEND=noninteractive

# Create an unprivileged user to run the app
RUN useradd -m -u 1000 appuser

# Install system packages needed for video generation and clean up
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache for dependency installation
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies without cache to reduce image size
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Set environment variables for Hugging Face caching and Gradio
ENV HF_HOME=/data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub \
    TRANSFORMERS_CACHE=/data/hf_cache/hub \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_ANALYTICS_ENABLED=false \
    PYTHONUNBUFFERED=1

# Create the cache directory and set permissions so the non‑root user can write to it
RUN mkdir -p /data/hf_cache \
 && chown -R appuser:appuser /data /app

# Change to non‑root user for better security
USER appuser

# Expose the Gradio port
EXPOSE 7860

# The entrypoint script handles optional model pre‑downloads and launches the app
ENTRYPOINT ["bash", "/app/entrypoint.sh"]