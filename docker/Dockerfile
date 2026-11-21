FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Nsight Compute CLI tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    cuda-nsight-compute-12-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt