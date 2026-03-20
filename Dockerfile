FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libfontconfig1 \
    build-essential \
    libopenexr-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

RUN pip install --no-cache-dir psutil ninja packaging

RUN rm -rf /usr/lib/python3/dist-packages/blinker*

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN python -c "import torch; print(f'PRE-FLASH CHECK: torch {torch.__version__} CUDA {torch.version.cuda}')"

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;9.0a;10.0;10.0a"
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash_attn==2.8.3

RUN python -c "\
import torch; print(f'torch {torch.__version__} CUDA {torch.version.cuda}'); \
import flash_attn; print(f'flash_attn {flash_attn.__version__}'); \
print('ALL IMPORTS OK')"

WORKDIR /workspace
ENV HYDRA_FULL_ERROR=1

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
