#!/bin/bash

# Optimized vLLM Server Startup Script for 8x H200 GPUs
# This script configures vLLM to use all 8 GPUs with optimal settings for LLM judge workload

set -e

echo "🚀 Starting Optimized 8-GPU vLLM Server for LLM Judge"
echo "============================================================"

# Server configuration
MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

# Multi-GPU optimizations
export TENSOR_PARALLEL_SIZE=8
export MAX_NUM_SEQS=16384                  # 8x higher for 8 GPUs
export MAX_NUM_BATCHED_TOKENS=262144       # 8x higher for 8 GPUs
export GPU_MEMORY_UTILIZATION=0.92       # Slightly lower for multi-GPU stability
export BLOCK_SIZE=16                      # Optimized for shorter judge sequences
export ENABLE_CHUNKED_PREFILL=true       # Better throughput for many requests
export ENABLE_PREFIX_CACHING=true        # Cache common prompt prefixes

# Model configuration
export DTYPE=float16                      # Best balance of speed/quality
export MAX_MODEL_LEN=16384                # Shorter for judge tasks
export TRUST_REMOTE_CODE=true

# Performance monitoring
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Configuration:"
echo "  Model: $MODEL_ID"
echo "  GPUs: 8 (tensor parallelism)"
echo "  Max concurrent sequences: $MAX_NUM_SEQS"
echo "  Max batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "  Memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  Server URL: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""

# Check GPU availability
echo "Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
echo ""


cd /gpfs/scrubbed/osey/Dataset_Distillation/ADRA/adra/utils_rl/llm_judge_deploy

# Start the optimized server
python run_vllm_server.py \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --host "$HOST" \
    --tensor-parallel-size 8 \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --block-size $BLOCK_SIZE \
    --dtype float16 \
    --max-model-len $MAX_MODEL_LEN

echo "🎉 vLLM server started successfully!"
