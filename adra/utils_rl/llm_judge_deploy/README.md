# Remote LLM Judge Server Deployment

This directory contains scripts to deploy the vLLM server for the remote LLM judge functionality.

## Quick Start

### Option 1: Python vLLM Server (Recommended)
```bash
# Install dependencies
pip install vllm

# Run server
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000
```

### Option 2: Direct vLLM Command
```bash
# Run vLLM server directly
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype float16 \
    --max-model-len 32768
```

### Option 3: Docker Container
```bash
# Run vLLM server in container
./run_vllm_docker.sh
```

## Multi-Node Setup

1. **Start server** on GPU node
2. **Note the IP address** from server output (e.g., 172.16.0.60)
3. **Set environment variable** on evaluation node:
   ```bash
   export LLM_JUDGE_SERVER_URL="http://172.16.0.60:8000"
   ```

## Testing

```bash
# Test connectivity (from parent directory)
cd ../tests/
python test_llm_judge_connectivity.py http://YOUR_SERVER_IP:8000
```

## Configuration

All servers support environment variables:
- `MODEL_ID` - Model to use (default: Qwen/Qwen3-32B)
- `PORT` - Server port (default: 8000)  
- `HF_TOKEN` - Hugging Face token (for private models)
- `DTYPE` - Model precision (default: float16)
- `MAX_MODEL_LEN` - Maximum sequence length (default: 32768)
- `GPU_MEMORY_UTILIZATION` - GPU memory utilization (default: 0.9)

## Performance Tuning

### GPU Memory
- Adjust `GPU_MEMORY_UTILIZATION` based on available VRAM
- Use `--dtype float16` for memory efficiency
- Consider model quantization for larger models

### Throughput
- Increase `--max-num-batched-tokens` for higher throughput
- Adjust `--max-num-seqs` based on concurrent requests
- Use `--enable-prefix-caching` for similar prompts

### Latency  
- Use `--disable-log-stats` to reduce overhead
- Consider smaller models (e.g., Qwen3-14B) for faster inference
- Enable KV cache optimizations

## Files

- `run_vllm_server.py` - Python wrapper for vLLM server
- `run_vllm_docker.sh` - Docker-based deployment
- `test_vllm_connectivity.py` - Connection and functionality test

## Troubleshooting

### Server Won't Start
1. Check GPU availability: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check available VRAM (Qwen3-32B needs ~64GB+ VRAM)

### Connection Errors
1. Verify server URL is accessible
2. Check firewall settings
3. Test with curl: `curl http://server:8000/v1/models`

### Performance Issues
1. Monitor GPU utilization: `nvidia-smi`
2. Check server logs for bottlenecks
3. Adjust batch size and memory settings
4. Consider using multiple GPU instances
