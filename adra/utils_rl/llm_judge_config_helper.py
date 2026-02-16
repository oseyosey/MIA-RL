#!/usr/bin/env python3
"""
Configuration helper for LLM Judge optimization based on GPU type and sequence length.

This helper provides optimal settings for different hardware and workload configurations.
"""

from typing import Dict, Tuple, Any

def get_optimal_config(
    gpu_type: str = "h200", 
    num_gpus: int = 8,
    avg_sequence_length: int = 1000,
    workload_type: str = "balanced"
) -> Dict[str, Any]:
    """
    Get optimal configuration for LLM judge client based on hardware and workload.
    
    Args:
        gpu_type: Type of GPU ("h200", "a100", "v100")
        num_gpus: Number of GPUs in the server
        avg_sequence_length: Average token length of sequences
        workload_type: "conservative", "balanced", "aggressive"
        
    Returns:
        Dict with optimal max_workers and batch_size_per_worker
    """
    
    # Base configurations by GPU type
    gpu_configs = {
        "h200": {
            "memory_gb": 141,
            "bandwidth_multiplier": 1.0,
            "base_throughput": 2048
        },
        "a100": {
            "memory_gb": 80,
            "bandwidth_multiplier": 0.7,
            "base_throughput": 1024
        },
        "v100": {
            "memory_gb": 32,
            "bandwidth_multiplier": 0.4,
            "base_throughput": 512
        }
    }
    
    if gpu_type not in gpu_configs:
        raise ValueError(f"Unsupported GPU type: {gpu_type}. Use: {list(gpu_configs.keys())}")
    
    gpu_config = gpu_configs[gpu_type]
    
    # Scale by number of GPUs
    total_throughput = gpu_config["base_throughput"] * num_gpus
    
    # Adjust for sequence length
    if avg_sequence_length < 500:
        length_multiplier = 1.5  # Short sequences, can handle more
    elif avg_sequence_length < 1500:
        length_multiplier = 1.0  # Medium sequences, baseline
    else:
        length_multiplier = 0.6  # Long sequences, reduce load
    
    adjusted_throughput = int(total_throughput * length_multiplier)
    
    # Configure based on workload type
    workload_configs = {
        "conservative": {"worker_ratio": 0.125, "batch_multiplier": 4},    # 1/8 workers, 4x batch
        "balanced":     {"worker_ratio": 0.25,  "batch_multiplier": 2},    # 1/4 workers, 2x batch  
        "aggressive":   {"worker_ratio": 0.5,   "batch_multiplier": 1}     # 1/2 workers, 1x batch
    }
    
    if workload_type not in workload_configs:
        raise ValueError(f"Unsupported workload type: {workload_type}. Use: {list(workload_configs.keys())}")
    
    workload_config = workload_configs[workload_type]
    
    # Calculate optimal settings
    max_workers = max(16, int(adjusted_throughput * workload_config["worker_ratio"] / 32) * 32)
    batch_size_per_worker = min(128, max(8, int(adjusted_throughput / max_workers)))
    
    # Ensure we don't exceed server capacity
    total_requests = max_workers * batch_size_per_worker
    if total_requests > adjusted_throughput:
        # Scale down proportionally
        scale_factor = adjusted_throughput / total_requests
        max_workers = max(16, int(max_workers * scale_factor))
        batch_size_per_worker = min(128, int(adjusted_throughput / max_workers))
    
    return {
        "max_workers": max_workers,
        "batch_size_per_worker": batch_size_per_worker,
        "estimated_throughput": max_workers * batch_size_per_worker,
        "gpu_info": {
            "type": gpu_type,
            "count": num_gpus,
            "memory_total_gb": gpu_config["memory_gb"] * num_gpus
        },
        "sequence_info": {
            "avg_length": avg_sequence_length,
            "workload_type": workload_type
        }
    }

def print_config_recommendation(config: Dict[str, Any]) -> None:
    """Print a formatted configuration recommendation."""
    print("🎯 LLM Judge Optimization Recommendation")
    print("=" * 50)
    print(f"GPU Setup: {config['gpu_info']['count']}x {config['gpu_info']['type'].upper()}")
    print(f"Total GPU Memory: {config['gpu_info']['memory_total_gb']:,}GB")
    print(f"Average Sequence Length: {config['sequence_info']['avg_length']} tokens")
    print(f"Workload Type: {config['sequence_info']['workload_type'].title()}")
    print()
    print("📊 Optimal Configuration:")
    print(f"  max_workers = {config['max_workers']}")
    print(f"  batch_size_per_worker = {config['batch_size_per_worker']}")
    print(f"  Total concurrent requests: {config['estimated_throughput']}")
    print()
    print("🔧 To apply these settings:")
    print("  1. Update llm_judge_client.py:")
    print(f"     max_workers_default = {config['max_workers']}")
    print(f"     batch_size_per_worker_default = {config['batch_size_per_worker']}")
    print("  2. Or pass as parameters:")
    print(f"     client = LLMJudgeClient(batch_size_per_worker={config['batch_size_per_worker']})")

# Pre-defined optimal configurations
H200_CONFIGS = {
    "short_sequences": get_optimal_config("h200", 8, 500, "balanced"),      # <500 tokens
    "medium_sequences": get_optimal_config("h200", 8, 1000, "balanced"),    # ~1000 tokens  
    "long_sequences": get_optimal_config("h200", 8, 2000, "conservative")   # >1500 tokens
}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get optimal LLM judge configuration")
    parser.add_argument("--gpu-type", choices=["h200", "a100", "v100"], default="h200")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument("--workload", choices=["conservative", "balanced", "aggressive"], default="balanced")
    
    args = parser.parse_args()
    
    config = get_optimal_config(
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus, 
        avg_sequence_length=args.sequence_length,
        workload_type=args.workload
    )
    
    print_config_recommendation(config)
    
    print("\n🔍 Other Configurations:")
    print(f"H200 Short Sequences (500 tokens): workers={H200_CONFIGS['short_sequences']['max_workers']}, batch={H200_CONFIGS['short_sequences']['batch_size_per_worker']}")
    print(f"H200 Medium Sequences (1000 tokens): workers={H200_CONFIGS['medium_sequences']['max_workers']}, batch={H200_CONFIGS['medium_sequences']['batch_size_per_worker']}")  
    print(f"H200 Long Sequences (2000 tokens): workers={H200_CONFIGS['long_sequences']['max_workers']}, batch={H200_CONFIGS['long_sequences']['batch_size_per_worker']}")
