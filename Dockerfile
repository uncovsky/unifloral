FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    libosmesa6 libgl1 libglu1-mesa \
    libxrender1 libxext6 libsm6

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install jaxlib with CUDA support and jax BEFORE other packages
RUN pip install --upgrade \
    jaxlib==0.4.16+cuda12.cudnn89 \
    jax==0.4.16 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
ENV MUJOCO_GL=osmesa


# Set working directory
WORKDIR /work/rl

COPY . /work/rl

RUN pip install --no-cache-dir -e .

# Default command
CMD ["python"]
