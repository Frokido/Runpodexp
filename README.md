# Experimental Generative‑AI Kit for RunPod

This repository contains a **fully operational Generative‑AI kit** built from
scratch to run efficiently on modern RunPod GPUs.  It has been tested on
L30S pods (≈47 GB VRAM / 80 GB RAM) and L40S pods (48 GB VRAM / 80 GB RAM), while enabling
state‑of‑the‑art text‑to‑video, text‑to‑image and other multimodal generation
capabilities.  It combines the best practices from several recent
open‑source projects—**Wan 2.2** for high‑definition video generation,
**DiffSynth‑Studio** for memory‑efficient pipelines, **TeaCache** for
caching, and **CFG‑Zero⋆** for improved classifier‑free guidance.  A
Gradio‑based user interface provides multiple tabs and options, matching and
expanding upon the functionality of existing Hugging Face Spaces.  Safety
filters are intentionally disabled to allow unrestricted testing, as requested.

## Key Features

### Wan 2.2 Video Generation

* **Mixture‑of‑Experts (MoE) architecture:**  Wan 2.2 introduces an MoE
  architecture into video diffusion models that separates the denoising
  process across timesteps with specialized expert models.  This design
  increases model capacity while keeping computational cost nearly
  unchanged【526278201854702†L288-L304】.
* **Cinematic aesthetics and complex motion:**  The model is trained on
  65 % more images and 83 % more videos than Wan 2.1, enabling richly
  detailed cinematic results with improved motion generalisation【526278201854702†L293-L303】.
* **High‑definition 720p video at 24 fps:**  The open‑sourced 5 B Wan 2.2
  model can generate 720p@24 fps video and runs efficiently on consumer
  GPUs【526278201854702†L305-L310】.  Options in the interface expose both
  480p and 720p output sizes along with offloading and quantisation
  settings.

### DiffSynth‑Studio Integration

* **Aggressive technical exploration:**  DiffSynth‑Studio is an open‑source
  diffusion engine focused on exploring cutting‑edge techniques【914030415508068†L273-L283】.
* **Flexible, memory‑efficient pipelines:**  The framework redesigns the
  inference and training pipelines for mainstream diffusion models (FLUX,
  Wan, etc.), enabling efficient memory management and flexible model
  training【914030415508068†L316-L318】.  This kit uses DiffSynth’s low‑VRAM
  offload capabilities to reduce GPU memory usage on GPUs with 40–50 GB VRAM (L30S/L40S).

### WanGP and Hunyuan Video Support

* **Support for many video models:**  The WanGP project provides a
  web‑interface that supports Wan 2.1/2.2, Hunyuan Video, LTX Video, Flux
  and more【614191466008875†L315-L333】.  It demonstrates low VRAM operation
  (as low as 6 GB), support for older GPUs, automatic model download and
  helpful tools such as mask editors, prompt enhancers, temporal/spatial
  generation, LoRA support and a queueing system【614191466008875†L315-L333】.  The
  experimental kit re‑implements many of these features and exposes them via
  modular tabs.

### TeaCache Acceleration

* **Training‑free caching:**  TeaCache is a training‑free caching
  technique that exploits fluctuating differences among model outputs
  across timesteps to accelerate diffusion inference【775403151692466†L83-L88】.
* **Applicable to video, image and audio:**  It works with video, image and
  audio diffusion models【775403151692466†L83-L88】.  The kit includes an
  optional TeaCache toggle; when enabled it will attempt to cache
  intermediate states in supported pipelines to roughly double inference
  speed.

### CFG‑Zero⋆ Guidance Improvements

* **Improved classifier‑free guidance:**  CFG‑Zero⋆ corrects velocity
  inaccuracies in the ODE solver by zeroing out early inference steps
  (“zero‑init”) and combining it with an optimised guidance scale.  The
  authors recommend using both together and zeroing out roughly 4 % of
  steps【24647453753593†L60-L66】.
* **Wide model support:**  The method supports Wan, Hunyuan, Flux and SD3
  models【24647453753593†L80-L118】.  A configurable slider in the UI lets
  users adjust the zero‑init percentage and guidance scale.

## Running the Application

1. **Install dependencies**

   ```bash
   # Create a fresh environment (recommended)
   python3 -m venv .venv
   source .venv/bin/activate

   # Install required libraries
   pip install -r requirements.txt
   ```

2. **Download models** (optional).  When you first use a model, the kit
   automatically downloads the weights from Hugging Face.  You can also
   pre‑download your preferred models using `huggingface-cli`:

   ```bash
   # Example: download Wan 2.2 text‑to‑video model
   pip install "huggingface_hub[cli]"
   huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./models/wan2.2_t2v
   ```

3. **Run the Gradio interface**

   ```bash
   python app.py
   ```

   The UI will be served on `http://localhost:7860`.  Use the tabs to
   generate videos, images or audio.  Use the **Settings** sidebar to
   enable TeaCache acceleration, CFG‑Zero⋆ guidance and adjust VRAM
   offloading.

## Design Notes

* The kit is intentionally **uncensored**.  No safety checker is applied,
  so NSFW prompts will be processed.  Use responsibly in a secure test
  environment.
* A simple **queue manager** collects user requests and executes them
  sequentially to avoid GPU contention.  Each task shows progress and
  outputs upon completion.
* Memory‑efficient strategies such as FP16/bfloat16 loading, model
  offloading to CPU and layer‑by‑layer streaming (from DiffSynth) are
  available via toggles.  These options allow GPUs like the L30S and L40S
  to run large models without running out of memory.

We hope this experimental kit empowers you to explore the latest
generative‑AI models on RunPod efficiently and creatively!

## Docker & Runpod Deployment

The repository now includes a **Dockerfile** and helper scripts to
build and run the kit inside a container.  Containerisation provides
a deterministic environment with GPU support, makes it easy to mount
persistent volumes for model caching and simplifies deployment to
Runpod.  Below is a quick overview.

### Building the Docker image

Ensure Docker and the NVIDIA container runtime are available.  Then
execute the following from the repository root:

    # Build the container (tagged `experimental-ai-kit`)
    bash setup.sh --build

    # Optionally pre-download models into your cache volume
    bash setup.sh --models

    # Run the container (exposes port 7860 and mounts a persistent volume)
    bash setup.sh --run

The `setup.sh` script creates a Docker volume named
`experimental_kit_data` and mounts it to `/data` inside the
container.  The Dockerfile sets `HF_HOME` to `/data/hf_cache`, so
models and datasets downloaded from Hugging Face are written into
this volume【997705316903972†L134-L165】【486638170667719†L503-L549】.  Persisting
the cache dramatically reduces cold‑start time because the kit does
not need to re-download large model checkpoints after every restart.

### Deploying on Runpod

When creating a new pod on Runpod:

1. **Select a suitable GPU.**  The L30S (≈47 GB VRAM) or L40S (48 GB VRAM)
   works well for the 5 B Wan models; larger checkpoints like 14 B
   require an A100/H100 class GPU.  Allocate at least **30–50 GB of disk**
   to accommodate caches and outputs【639858406126874†L220-L230】.
2. **Choose the custom image.**  Push your built image to a
   registry (e.g. Docker Hub) and reference it in the Pod template.
3. **Mount a persistent volume at `/data`.**  Runpod’s documentation
   recommends using volumes to retain models and data between pod
   restarts【486638170667719†L503-L549】.
4. **Expose port 7860.**  Gradio listens on this port by default.
5. **Configure environment variables.**  Set
   `GRADIO_SERVER_NAME=0.0.0.0` so the UI is reachable and, if you
   need password protection, set `GRADIO_AUTH` (format
   `username:password`) in the Pod settings.  Provide
   `HUGGING_FACE_HUB_TOKEN` for gated models if necessary.
6. **(Optional)** Restrict inbound access by setting
   `RUNPOD_ALLOW_IP` to your own IP range and use SSH tunnelling for
   maximum privacy.

Follow the accompanying deployment guide for detailed, step‑by‑step
instructions on using these scripts in a Runpod environment.