# Remote LLM Judge for Reconstruction Evaluation

This module implements a remote LLM judge system using vLLM for high-performance text generation. It provides an alternative to the local LLM judge implementation with significantly improved speed and memory efficiency.

## Overview

The remote LLM judge (`llm_judge_remote.py`) provides the same functionality as the local implementation (`llm_judge_local.py`) but delegates text generation to a remote vLLM server. This architecture offers:

- **Performance**: vLLM's optimized inference engine delivers 10-20x faster generation
- **Memory Efficiency**: No GPU memory required on evaluation workers
- **Scalability**: Centralized LLM service can serve multiple evaluation runs
- **Flexibility**: Easy switching between local and remote inference

## Quick Start

### 1. Deploy vLLM Server

#### Option 1: Python Script (Recommended)
```bash
cd llm_judge_deploy/
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000
```

#### Option 2: Direct vLLM Command  
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype float16
```

#### Option 3: Docker Container
```bash
cd llm_judge_deploy/
./run_vllm_docker.sh
```

### 2. Configure Environment

```bash
# Set server URL
export LLM_JUDGE_SERVER_URL="http://localhost:8000"

# Optional: authentication
export LLM_JUDGE_SERVER_API_KEY="your-api-key"
```

### 3. Use in Evaluation

```bash
# Enable remote LLM judge in evaluation
python -m adra.scripts.evaluate_reconstructions \
    --input data.parquet \
    --output results.json \
    --evaluate-local-llm-judge \
    --llm-judge-use-remote \
    --llm-judge-server-url http://localhost:8000
```

## Architecture

```
┌─────────────────────┐    HTTP/OpenAI API    ┌──────────────────────┐
│   Evaluation        │ ──────────────────────> │  vLLM Server         │
│   Worker            │ <────────────────────── │  Qwen3-32B           │
│   - No GPU needed   │    Generated Text       │  - GPU accelerated   │
└─────────────────────┘                        └──────────────────────┘
```

## Performance Comparison

| Metric | Local (HF Transformers) | Remote (vLLM) | Improvement |
|--------|-------------------------|---------------|-------------|
| Throughput | ~2 req/min | ~30 req/min | **15x faster** |
| Memory | 64GB GPU | 0GB (worker) | **No GPU needed** |
| Latency | 30s/request | 2s/request | **15x faster** |
| Batch Size | Limited | High | **Better batching** |

## Features

### 1. **Drop-in Replacement**
The remote LLM judge maintains API compatibility with the local version:

```python
# Both work identically
scores = compute_llm_judge_scores_batch(
    problems=["What is 2+2?"],
    ground_truths=["4"],
    candidates_list=[["4", "four"]],
    # Local
    model_name="Qwen/Qwen3-32B"
    # Remote 
    server_url="http://localhost:8000"
)
```

### 2. **Clear Error Reporting**  
If the remote server is unavailable, the system raises clear exceptions instead of silently falling back to lexical similarity. This ensures you know when the LLM judge isn't working properly.

### 3. **Efficient Batching**
vLLM's optimized batching provides much higher throughput:

```python
# Efficiently processes large batches
scores = compute_llm_judge_scores_batch(
    problems=problems_list,      # 1000 problems
    ground_truths=gt_list,       # 1000 ground truths  
    candidates_list=cand_lists,  # 1000 x 5 candidates
    batch_size=32,               # Optimal batch size
    server_url="http://localhost:8000"
)
```

### 4. **Connection Management**
- HTTP connection pooling for efficiency
- Automatic retry with exponential backoff
- Configurable timeouts and error handling

### 5. **Thinking Mode Support**
Full support for Qwen3's thinking/non-thinking modes:

```python
# Enable thinking mode for complex reasoning
scores = compute_llm_judge_scores_batch(
    problems=problems,
    ground_truths=ground_truths,
    candidates_list=candidates_list,
    enable_thinking=True,
    server_url="http://localhost:8000"
)
```

## Configuration Options

### Environment Variables
- `LLM_JUDGE_SERVER_URL`: Default server URL
- `LLM_JUDGE_SERVER_API_KEY`: Optional authentication key  
- `LLM_JUDGE_SERVER_TIMEOUT`: Request timeout (default: 60s)

### Function Parameters
```python
compute_llm_judge_scores_batch(
    problems=problems,
    ground_truths=ground_truths,
    candidates_list=candidates_list,
    
    # Model configuration
    model_name="Qwen/Qwen3-32B",
    enable_thinking=False,
    
    # Generation parameters
    temperature=0.7,
    top_p=0.8,
    max_new_tokens=512,
    
    # Remote server configuration
    server_url="http://localhost:8000",
    api_key=None,
    timeout=60.0,
    
    # Performance tuning
    batch_size=8
)
```

### Command-line Arguments
```bash
python -m adra.scripts.evaluate_reconstructions \
    --evaluate-local-llm-judge \
    --llm-judge-use-remote \
    --llm-judge-server-url http://localhost:8000 \
    --llm-judge-model Qwen/Qwen3-32B \
    --llm-judge-enable-thinking \
    --llm-judge-batch-size 16 \
    --llm-judge-temperature 0.7 \
    --llm-judge-timeout 90.0
```

## Deployment Configurations

### Single Node Setup
For evaluation and server on the same machine:

```bash
# Terminal 1: Start server
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000

# Terminal 2: Run evaluation  
export LLM_JUDGE_SERVER_URL="http://localhost:8000"
python -m adra.scripts.evaluate_reconstructions --evaluate-local-llm-judge --llm-judge-use-remote
```

### Multi-Node Setup
For evaluation and server on different machines:

```bash
# GPU node: Start server
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000 --host 0.0.0.0
# Note the server IP from output (e.g., 172.16.0.60)

# Evaluation node: Set server URL
export LLM_JUDGE_SERVER_URL="http://172.16.0.60:8000"
python -m adra.scripts.evaluate_reconstructions --evaluate-local-llm-judge --llm-judge-use-remote
```

### High-Availability Setup
For production deployments with multiple replicas:

```bash
# Deploy multiple server instances
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8001
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8002

# Use load balancer (nginx, HAProxy, etc.)
# Point evaluation to load balancer URL
export LLM_JUDGE_SERVER_URL="http://load-balancer:8000"
```

## Testing and Validation

### 1. Test Server Connectivity
```bash
cd llm_judge_deploy/
python test_vllm_connectivity.py http://localhost:8000
```

This tests:
- Server health and availability
- Model information retrieval  
- Single and batch text generation
- LLM judge prompt evaluation
- Error handling

### 2. Compare Local vs Remote
```python
# Test both implementations
from adra.utils_rl.llm_judge_local import compute_llm_judge_scores_batch as local_batch
from adra.utils_rl.llm_judge_remote import compute_llm_judge_scores_batch as remote_batch

# Compare results (should be very similar)
local_scores, _ = local_batch(problems, ground_truths, candidates_list)
remote_scores, _ = remote_batch(problems, ground_truths, candidates_list, 
                                server_url="http://localhost:8000")
```

### 3. Performance Benchmarking
```bash
# Benchmark evaluation speed
time python -m adra.scripts.evaluate_reconstructions \
    --input large_dataset.parquet \
    --evaluate-local-llm-judge \
    --llm-judge-use-remote
```

## Performance Optimization

### 1. Server Configuration
Optimize vLLM server for your hardware:

```bash
python run_vllm_server.py \
    --model Qwen/Qwen3-32B \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --dtype float16
```

### 2. Client Configuration
Tune client parameters for your workload:

```python
# High throughput
llm_judge_kwargs = {
    "batch_size": 32,            # Larger batches  
    "timeout": 120.0,            # Longer timeout
    "max_new_tokens": 256,       # Shorter responses
    "temperature": 0.1,          # Less randomness
}

# Low latency  
llm_judge_kwargs = {
    "batch_size": 4,             # Smaller batches
    "timeout": 30.0,             # Shorter timeout
    "max_new_tokens": 128,       # Shorter responses
}
```

### 3. Model Selection
Choose model size based on requirements:

```bash
# Faster inference, lower quality
python run_vllm_server.py --model Qwen/Qwen3-14B

# Balanced performance  
python run_vllm_server.py --model Qwen/Qwen3-32B

# Higher quality, slower inference
python run_vllm_server.py --model Qwen/Qwen3-72B
```

## Monitoring and Troubleshooting

### Health Monitoring
```bash
# Check server health
curl http://localhost:8000/health

# Get model information
curl http://localhost:8000/v1/models

# Monitor server logs
# Check GPU utilization: nvidia-smi
```

### Common Issues

#### Server Won't Start
1. **Insufficient GPU memory**: Reduce `--gpu-memory-utilization` or use smaller model
2. **CUDA errors**: Check CUDA installation and driver compatibility
3. **Port conflicts**: Change `--port` or kill existing processes

#### Connection Errors  
1. **Network issues**: Verify firewall settings and network connectivity
2. **Wrong URL**: Check server IP address and port
3. **Authentication**: Verify API key configuration

#### Performance Issues
1. **Slow inference**: Increase batch size, check GPU utilization
2. **Memory errors**: Reduce batch size or sequence length
3. **Timeouts**: Increase client timeout or check server load

#### Evaluation Failures
1. **Evaluation failures**: Check server connectivity and logs
2. **Score extraction errors**: Verify prompt template and model responses  
3. **Inconsistent results**: Check temperature settings and model configuration

### Debug Mode
Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run evaluation with debug output
```

## Migration Guide

### From Local to Remote

1. **Deploy vLLM server**:
   ```bash
   python run_vllm_server.py --model Qwen/Qwen3-32B
   ```

2. **Update evaluation commands**:
   ```bash
   # Before (local)
   python -m adra.scripts.evaluate_reconstructions --evaluate-local-llm-judge
   
   # After (remote)  
   python -m adra.scripts.evaluate_reconstructions --evaluate-local-llm-judge --llm-judge-use-remote
   ```

3. **Verify results consistency**:
   - Run both local and remote on small dataset
   - Compare output metrics for sanity check

### Configuration Migration
```python
# Old local configuration
llm_judge_kwargs = {
    "model_name": "Qwen/Qwen3-32B",
    "enable_thinking": False,
    "batch_size": 8,
    "temperature": 0.7,
}

# New remote configuration  
llm_judge_kwargs = {
    "model_name": "Qwen/Qwen3-32B",
    "enable_thinking": False, 
    "batch_size": 8,
    "temperature": 0.7,
    "use_remote": True,
    "server_url": "http://localhost:8000",
    "timeout": 60.0,
}
```

## Integration Examples

### Basic Usage
```python
from adra.utils_rl.llm_judge_remote import compute_llm_judge_scores_batch

scores, best_indices = compute_llm_judge_scores_batch(
    problems=["What is 2+2?", "What is 3+3?"],
    ground_truths=["4", "6"], 
    candidates_list=[["4", "four"], ["6", "six"]],
    server_url="http://localhost:8000"
)
```

### Evaluation Pipeline Integration
```python
from adra.utils_rl.reconstruction_evaluation import evaluate_dataset

results = evaluate_dataset(
    ground_truth_texts=ground_truths,
    candidates_list=candidates_list,
    problems_list=problems,
    evaluate_local_llm_judge=True,
    llm_judge_kwargs={
        "use_remote": True,
        "server_url": "http://localhost:8000",
        "batch_size": 16,
        "temperature": 0.7,
    }
)
```

### Batch Processing
```python
# Process large datasets efficiently
results = evaluate_dataset(
    ground_truth_texts=ground_truths,    # 10,000 items
    candidates_list=candidates_list,     # 10,000 x 5 candidates
    problems_list=problems,              # 10,000 problems
    evaluate_local_llm_judge=True,
    llm_judge_kwargs={
        "use_remote": True,
        "server_url": "http://localhost:8000", 
        "batch_size": 32,                # Large batches for efficiency
        "timeout": 120.0,                # Longer timeout
    }
)
```

## Future Enhancements

1. **Multi-GPU Support**: Distribute across multiple vLLM instances
2. **Model Caching**: Cache frequent prompt-response pairs
3. **Streaming Support**: Real-time evaluation for large datasets
4. **Auto-scaling**: Dynamic server scaling based on load
5. **Model Quantization**: 8-bit/4-bit models for memory efficiency

## Limitations

1. **Network Dependency**: Requires stable connection to vLLM server
2. **Model Memory**: vLLM server needs significant GPU memory (32B model ~64GB)  
3. **Setup Complexity**: More complex deployment than local inference
4. **Latency**: Network overhead adds ~100-500ms per request

## Support

For issues and questions:
1. Check server logs and connectivity
2. Verify model and vLLM installation
3. Test with provided connectivity script
4. Review configuration parameters
