#!/bin/bash
set -e

# ==========================================
# 1. PROFILER & HARDWARE TOGGLES
# ==========================================
USE_DOCKER=true
GPU_DEVICE="device=0"   # Target a single GPU (e.g., GPU 0)

ENABLE_TORCH_PROF=true
ENABLE_NSYS=false
ENABLE_NCU=false        # Set to true for Nsight Compute kernel profiling

# ==========================================
# 2. HUGGING FACE TOKEN
# ==========================================
HF_TOKEN="hf_your_actual_token_here"  # Replace with your actual HF token

# ==========================================
# 3. TRAINING & WINDOW PARAMETERS
# ==========================================
BATCH_SIZE=1
TOTAL_STEPS=10
PROFILE_START_STEP=3
PROFILE_STEPS=3
MAX_PIXELS=$((64 * 64))
MAX_LENGTH=512

DOCKER_IMAGE="qwen2vl-profiler:latest"

# Build Docker image if enabled
if [ "$USE_DOCKER" = true ]; then
    echo "==> Building Docker image..."
    docker build -t $DOCKER_IMAGE docker/
fi

# Construct Python arguments
PY_ARGS="\
--batch_size $BATCH_SIZE \
--total_steps $TOTAL_STEPS \
--profile_start_step $PROFILE_START_STEP \
--profile_steps $PROFILE_STEPS \
--max_pixels $MAX_PIXELS \
--max_length $MAX_LENGTH"

if [ "$ENABLE_TORCH_PROF" = true ]; then
    PY_ARGS="$PY_ARGS --enable_torch_profiler"
fi

if [ "$ENABLE_NSYS" = true ] || [ "$ENABLE_NCU" = true ]; then
    PY_ARGS="$PY_ARGS --enable_cuda_profiler_api"
fi

CMD="python train_qwen.py $PY_ARGS"

# Wrap with Nsight Systems (nsys)
if [ "$ENABLE_NSYS" = true ]; then
    echo "==> Wrapping with Nsight Systems..."
    CMD="nsys profile \
        --trace=cuda,nvtx,osrt \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output=nsys_qwen_report \
        --force-overwrite=true \
        $CMD"

# Wrap with Nsight Compute (ncu)
elif [ "$ENABLE_NCU" = true ]; then
    echo "==> Wrapping with Nsight Compute..."
    CMD="ncu \
        --set full \
        --profile-from-start off \
        --target-processes all \
        --output=ncu_qwen_report \
        --force-overwrite \
        $CMD"
fi

# Execute inside Docker (Single GPU) or Locally
if [ "$USE_DOCKER" = true ]; then
    echo "==> Running inside Docker container on single GPU ($GPU_DEVICE)..."
    docker run --rm --gpus "$GPU_DEVICE" --ipc=host \
        --cap-add=SYS_ADMIN \
        -v $(pwd):/workspace \
        -e HF_HOME=/workspace/hf_cache \
        -e HF_TOKEN=$HF_TOKEN \
        $DOCKER_IMAGE \
        bash -c "$CMD"
else
    echo "==> Running locally on single GPU..."
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$HF_TOKEN eval $CMD
fi

echo "==> Done! Check your workspace directory for generated profile outputs."