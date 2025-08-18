# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional video processing dependencies
RUN pip install --no-cache-dir opencv-python-headless

# Copy application files
COPY app.py .
COPY preload_models.py .
COPY entrypoint.sh .
COPY setup.sh .
COPY resource_monitor.sh .

# Make shell scripts executable
RUN chmod +x entrypoint.sh setup.sh resource_monitor.sh

# Create necessary directories
RUN mkdir -p /tmp /app/models /app/cache

# Set proper permissions
RUN chown -R 1000:1000 /app /tmp

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["python", "app.py"]
