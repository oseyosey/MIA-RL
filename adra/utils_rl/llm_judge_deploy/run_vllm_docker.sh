#!/bin/bash
# vLLM Server using Docker (Container-based approach)
# This runs the official vLLM server in a container

set -e

# Configuration
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
export PORT="${PORT:-8000}"
export DTYPE="${DTYPE:-float16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Create data directory for model cache
mkdir -p ~/vllm-cache

echo "vLLM Server (Docker) - Starting..."
echo "=================================="
echo "Model: $MODEL_ID"
echo "Port: $PORT"
echo "Data directory: ~/vllm-cache"
echo "Node: $(hostname)"
echo "IP addresses: $(hostname -I)"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found!"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "Warning: NVIDIA Docker runtime not detected."
    echo "GPU acceleration may not work properly."
    echo "Install nvidia-container-toolkit if needed."
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. You may need it for some models."
fi

# Run the vLLM server using Docker
echo "Starting vLLM server container..."
docker run --gpus all \
  --rm \
  -p ${PORT}:8000 \
  -v ~/vllm-cache:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model $MODEL_ID \
  --dtype $DTYPE \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-num-seqs $MAX_NUM_SEQS \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
