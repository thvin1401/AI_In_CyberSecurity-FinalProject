# CPU-optimised image — no CUDA required
# Uses official TensorFlow CPU image built with AVX2/MKL optimisations
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Tell TF to use all available CPU threads
    TF_NUM_INTEROP_THREADS=0 \
    TF_NUM_INTRAOP_THREADS=0 \
    # Suppress TF info/warning logs
    TF_CPP_MIN_LOG_LEVEL=2 \
    # OpenMP thread count — matches Xeon W-10855M (6 cores / 12 threads)
    OMP_NUM_THREADS=12

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    libgomp1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# rclone — for syncing data/checkpoints with Google Drive
RUN curl -fsSL https://rclone.org/install.sh | bash

WORKDIR /workspace

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download NLTK data at build time so containers start instantly
RUN python -c "\
import nltk; \
nltk.download('punkt',      quiet=True); \
nltk.download('punkt_tab',  quiet=True); \
nltk.download('stopwords',  quiet=True)"

# Copy project source
COPY src/     ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

CMD ["bash"]
