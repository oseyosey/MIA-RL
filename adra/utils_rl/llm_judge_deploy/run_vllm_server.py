#!/usr/bin/env python3
"""
vLLM Server for Remote LLM Judge

This script starts a vLLM server with OpenAI-compatible API for LLM judge evaluation.
It provides a high-performance inference server optimized for text generation.

Usage:
    python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000

Features:
- OpenAI-compatible API endpoints
- GPU acceleration with vLLM optimizations
- Support for thinking/non-thinking modes
- Batch processing optimization
- Memory-efficient inference
"""

import os
import sys
import argparse
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed. Install with: pip install vllm")
        sys.exit(1)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            print("⚠️ CUDA not available - will use CPU (very slow)")
    except ImportError:
        print("❌ PyTorch not installed")
        sys.exit(1)

def get_server_config():
    """Get server configuration from environment variables."""
    config = {
        "model": os.getenv("MODEL_ID", "Qwen/Qwen3-32B"),
        "port": int(os.getenv("PORT", "8000")),
        "host": os.getenv("HOST", "0.0.0.0"),
        "dtype": os.getenv("DTYPE", "float16"),
        "max_model_len": int(os.getenv("MAX_MODEL_LEN", "32768")),
        "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),  # Reduced for multi-GPU
        "max_num_batched_tokens": int(os.getenv("MAX_NUM_BATCHED_TOKENS", "65536")),  # 8x higher for 8 GPUs
        "max_num_seqs": int(os.getenv("MAX_NUM_SEQS", "2048")),  # 8x higher for 8 GPUs
        "tensor_parallel_size": int(os.getenv("TENSOR_PARALLEL_SIZE", "8")),  # Use all 8 GPUs!
        "trust_remote_code": os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true",
        # Additional optimizations for high-throughput LLM judge workload
        "enable_chunked_prefill": os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true",
        "enable_prefix_caching": os.getenv("ENABLE_PREFIX_CACHING", "true").lower() == "true",
        "block_size": int(os.getenv("BLOCK_SIZE", "16")),  # Optimize for shorter sequences
    }
    return config

#? Need Check ?#
def estimate_memory_requirements(model_name: str, dtype: str) -> float:
    """Estimate memory requirements for the model."""
    # Rough estimates for common models
    model_sizes = {
        "Qwen/Qwen3-32B": 64,  # GB
        "Qwen/Qwen3-14B": 28,  # GB  
        "Qwen/Qwen3-7B": 14,   # GB
        "Qwen/Qwen3-1.8B": 4,  # GB
    }
    
    base_size = model_sizes.get(model_name, 32)  # Default estimate
    
    # Adjust for dtype
    if dtype == "float16":
        multiplier = 1.0
    elif dtype == "float32":
        multiplier = 2.0
    elif dtype == "int8":
        multiplier = 0.5
    elif dtype == "int4":
        multiplier = 0.25
    else:
        multiplier = 1.0
    
    return base_size * multiplier

def run_vllm_server(
    model: str = "Qwen/Qwen3-32B",
    port: int = 8000,
    host: str = "0.0.0.0",
    dtype: str = "float16",
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.85,
    max_num_batched_tokens: int = 65536,
    max_num_seqs: int = 2048,
    tensor_parallel_size: int = 8,
    trust_remote_code: bool = True,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = True,
    block_size: int = 16,
    additional_args: Optional[list] = None
):
    """Run the vLLM server with specified configuration."""
    
    print("vLLM Server for LLM Judge - Starting...")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Server: {host}:{port}")
    print(f"Precision: {dtype}")
    print(f"Max sequence length: {max_model_len}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Tensor parallel size: {tensor_parallel_size} GPUs")
    print(f"Max concurrent sequences: {max_num_seqs}")
    print(f"Max batched tokens: {max_num_batched_tokens}")
    print(f"Chunked prefill: {enable_chunked_prefill}")
    print(f"Prefix caching: {enable_prefix_caching}")
    print(f"Node: {os.uname().nodename}")
    
    # Get network information
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Local IP: {local_ip}")
        print(f"🌐 Complete Server URL: http://{local_ip}:{port}")
        print(f"📋 Test with: python test_vllm_connectivity.py http://{local_ip}:{port}")
    except Exception:
        print("Local IP: Unable to determine")
        print(f"🌐 Complete Server URL: http://localhost:{port} (or use actual IP)")
        print(f"📋 Test with: python test_vllm_connectivity.py http://localhost:{port}")
    
    print()
    
    # Check memory requirements
    estimated_memory = estimate_memory_requirements(model, dtype)
    print(f"Estimated memory requirement: {estimated_memory:.1f}GB")
    
    try:
        import torch
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Available GPU memory: {available_memory:.1f}GB")
            if estimated_memory > available_memory * 0.9:
                print("⚠️ Warning: Model may not fit in GPU memory")
        print()
    except Exception:
        pass
    
    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", host,
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--block-size", str(block_size),
    ]
    
    # Add conditional optimizations
    if enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    # Set environment variables for HuggingFace
    env = os.environ.copy()
    if "HF_TOKEN" in env:
        print("Using HuggingFace token for model access")
    
    print("Starting vLLM server...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the server
        process = subprocess.run(cmd, env=env)
        return process.returncode
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM server for LLM judge evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_vllm_server.py --model Qwen/Qwen3-32B

  # Custom configuration
  python run_vllm_server.py --model Qwen/Qwen3-32B --port 8001 --dtype float16
  
  # High throughput setup
  python run_vllm_server.py --model Qwen/Qwen3-32B --max-num-seqs 512 --max-num-batched-tokens 16384
  
Environment variables:
  MODEL_ID                 Model to use (default: Qwen/Qwen3-32B)
  PORT                     Server port (default: 8000)
  DTYPE                    Model precision (default: float16)
  MAX_MODEL_LEN            Maximum sequence length (default: 32768)
  GPU_MEMORY_UTILIZATION   GPU memory usage fraction (default: 0.9)
  HF_TOKEN                 HuggingFace access token
        """
    )
    
    parser.add_argument("--model", default=None, help="Model name (overrides MODEL_ID env var)")
    parser.add_argument("--port", type=int, default=None, help="Server port (overrides PORT env var)")
    parser.add_argument("--host", default=None, help="Server host (default: 0.0.0.0)")
    parser.add_argument("--dtype", default=None, help="Model precision (float16, float32, int8, int4)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="Max batched tokens")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Max number of sequences")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Number of GPUs for tensor parallelism (default: 8)")
    parser.add_argument("--block-size", type=int, default=None, help="Block size for memory management (default: 16)")
    parser.add_argument("--no-chunked-prefill", action="store_true", help="Disable chunked prefill")
    parser.add_argument("--no-prefix-caching", action="store_true", help="Disable prefix caching")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    parser.add_argument("--vllm-args", nargs="*", help="Additional arguments to pass to vLLM")
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
        return 0
    
    # Get configuration from environment or arguments
    config = get_server_config()
    
    # Override with command line arguments
    if args.model is not None:
        config["model"] = args.model
    if args.port is not None:
        config["port"] = args.port
    if args.host is not None:
        config["host"] = args.host
    if args.dtype is not None:
        config["dtype"] = args.dtype
    if args.max_model_len is not None:
        config["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_batched_tokens is not None:
        config["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        config["max_num_seqs"] = args.max_num_seqs
    if args.tensor_parallel_size is not None:
        config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.block_size is not None:
        config["block_size"] = args.block_size
    if args.no_chunked_prefill:
        config["enable_chunked_prefill"] = False
    if args.no_prefix_caching:
        config["enable_prefix_caching"] = False
    if args.no_trust_remote_code:
        config["trust_remote_code"] = False
    
    # Check dependencies
    check_dependencies()
    
    # Run server
    return run_vllm_server(
        additional_args=args.vllm_args,
        **config
    )

if __name__ == "__main__":
    sys.exit(main())
