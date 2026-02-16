#!/usr/bin/env python3
"""TRL SFT fine-tuning using mixed contamination + general datasets.

This script performs SFT fine-tuning on a mixed dataset created by data_mixer.py.
The mixed dataset contains both contamination examples (e.g., AIME) and 
general examples (e.g., tulu-3-sft-mixture) in a specified ratio.

It can also train on raw datasets with problem/solution fields by using the
--raw_problem_solution_format flag.

Usage:
    # Using a HuggingFace mixed dataset
    python adra/scripts/produce_checkpoints/produce_checkpoint_trl_mixed_dataset.py \
        --hf_dataset_name username/mixed-sft-dataset \
        --model_name allenai/tulu-2-7b
    
    # Using a raw dataset with problem/solution fields
    python adra/scripts/produce_checkpoints/produce_checkpoint_trl_mixed_dataset.py \
        --hf_dataset_name osieosie/AIME-2021-2025-Cleaned \
        --model_name allenai/tulu-2-7b \
        --raw_problem_solution_format \
        --num_epochs 5 \
        --batch_size 4
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset, load_from_disk

# -----------------------------------------------------------------------------
# Ensure repository root is on PYTHONPATH
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adra.utils_rl import CheckpointConfig, produce_checkpoint  # noqa: E402
from adra.utils_rl.data_mixer import (  # noqa: E402
    load_mixed_dataset, 
    create_text_dataset, 
    preprocess_raw_problem_solution_dataset,
    filter_dataset_by_indices_file
)

# Dataset loading utilities are now imported from adra.utils_rl.data_mixer


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------

def train_mixed_dataset(
    dataset: Dataset,
    model_name: str = "allenai/tulu-2-7b",
    num_epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_length: int = 8192,
    effective_batch_size: int = 32,
    warmup_steps: int = 0,
    lr_scheduler_type: str = "constant",
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    checkpoint_name: Optional[str] = None,
    output_model_name: Optional[str] = None,
    output_base_dir: str = "model_checkpoints",
    gradient_checkpointing: bool = True,
    fp16: bool = True,
    bf16: Optional[bool] = None,
    save_total_limit: Optional[int] = 1,
) -> tuple[object, str]:
    """Train model on mixed dataset using TRL SFTTrainer.
    
    Args:
        dataset: Mixed dataset to train on
        model_name: Model name or path
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        effective_batch_size: Target effective batch size across all GPUs
        warmup_steps: Number of warmup steps
        lr_scheduler_type: LR scheduler type ("constant", "linear", "cosine", etc.)
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        checkpoint_name: Custom checkpoint name (fully overrides auto-generation)
        output_model_name: Model name prefix for output directory (default: use actual model_name)
        output_base_dir: Base directory for outputs
        gradient_checkpointing: Enable gradient checkpointing
        fp16: Use fp16 precision
        bf16: Use bf16 precision
        save_total_limit: Max number of checkpoints to keep (None = keep all, 1 = keep only best/last)
        
    Returns:
        Tuple of (model, checkpoint_directory)
    """
    
    # Calculate gradient accumulation steps
    world_size = int(os.environ.get("WORLD_SIZE", 1)) or 1
    gradient_accumulation_steps = max(1, effective_batch_size // (batch_size * world_size))
    
    actual_batch_size = batch_size * world_size * gradient_accumulation_steps
    if actual_batch_size != effective_batch_size:
        logging.warning(
            "Effective batch size %d cannot be met exactly; using %d",
            effective_batch_size, actual_batch_size
        )
    
    # Calculate training steps
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    logging.info("Training configuration:")
    logging.info("  Model: %s", model_name)
    logging.info("  Dataset size: %d examples", len(dataset))
    logging.info("  Epochs: %d", num_epochs)
    logging.info("  Per-device batch size: %d", batch_size)
    logging.info("  Gradient accumulation steps: %d", gradient_accumulation_steps)
    logging.info("  Effective batch size: %d", actual_batch_size)
    logging.info("  Learning rate: %s", learning_rate)
    logging.info("  LR scheduler: %s", lr_scheduler_type)
    logging.info("  Max sequence length: %d", max_seq_length)
    logging.info("  Total steps ≈ %d (warmup: %d)", total_steps, warmup_steps)
    logging.info("  LoRA: %s", "enabled" if use_lora else "disabled")
    
    # Auto-generate checkpoint name if not provided
    if checkpoint_name is None:
        # Extract dataset info from composition
        if "is_contamination" in dataset.column_names:
            contamination_examples = sum(dataset["is_contamination"])
            contamination_pct = contamination_examples / len(dataset) * 100
            checkpoint_name = f"mixed_sft_{len(dataset)}ex_{contamination_pct:.1f}pct_e{num_epochs}_lr{learning_rate}_sched{lr_scheduler_type}"
        else:
            # For raw datasets without contamination info
            checkpoint_name = f"sft_{len(dataset)}ex_e{num_epochs}_lr{learning_rate}_sched{lr_scheduler_type}"
        if use_lora:
            checkpoint_name += f"_lora_r{lora_r}"
    
    # Create text dataset for training
    text_dataset = create_text_dataset(dataset)
    
    # Configure checkpoint manager
    config = CheckpointConfig(
        model_name_or_path=model_name,
        max_seq_length=max_seq_length,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        save_total_limit=save_total_limit,
        output_base_dir=output_base_dir,
        output_model_name=output_model_name,
        checkpoint_name=checkpoint_name,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16,
        bf16=bf16,
        # TRL settings
        use_trl=True,
        apply_chat_template=False,  # dataset already templated
        add_generation_prompt=False,
        # TRL-specific parameters for better truncation handling
        packing=False,  # Disable packing to ensure proper truncation
        remove_unused_columns=False,  # Keep all columns for debugging
        # LoRA settings
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    
    # Train model
    logging.info("Starting SFT training...")
    model, ckpt_dir = produce_checkpoint(text_dataset, config)
    
    logging.info("Training completed successfully!")
    logging.info("Checkpoint saved to: %s", ckpt_dir)
    
    return model, ckpt_dir


# -----------------------------------------------------------------------------
# CLI interface
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train model on mixed contamination + general dataset")
    
    # Dataset args
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset_path", 
                              help="Local path to mixed dataset directory")
    dataset_group.add_argument("--hf_dataset_name",
                              help="HuggingFace dataset name (e.g., username/dataset-name)")
    parser.add_argument("--hf_token",
                       help="HuggingFace token for private datasets")
    parser.add_argument("--raw_problem_solution_format", action="store_true",
                       help="Enable if dataset has raw 'problem' and 'solution' fields instead of pre-formatted text/messages")
    parser.add_argument("--problem_field", default="problem",
                       help="Name of the problem field in raw dataset (default: problem)")
    parser.add_argument("--solution_field", default="solution",
                       help="Name of the solution field in raw dataset (default: solution)")
    parser.add_argument("--indices_file", type=str,
                       help="Path to JSON file with member_indices and nonmember_indices to filter dataset")
    
    # Model args
    parser.add_argument("--model_name", default="allenai/tulu-2-7b",
                       help="Model name or path")
    
    # Training args
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--effective_batch_size", type=int, default=32,
                       help="Target effective batch size across all GPUs")
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                       choices=["constant", "linear", "cosine", "cosine_with_restarts", 
                               "polynomial", "constant_with_warmup"],
                       help="Learning rate scheduler type (constant recommended for single-epoch memorization)")
    
    # LoRA args
    parser.add_argument("--use_lora", action="store_true",
                       help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # Precision args
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use fp16 precision")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bf16 precision (overrides fp16)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                       help="Disable gradient checkpointing")
    
    # Output args
    parser.add_argument("--checkpoint_name",
                       help="Custom checkpoint name (auto-generated if not provided)")
    parser.add_argument("--output_model_name",
                       help="Model name prefix for output directory (e.g., 'tulu-2-7b_contaminated'). If not provided, uses actual model_name")
    parser.add_argument("--output_base_dir", default="model_checkpoints",
                       help="Base directory for model outputs")
    parser.add_argument("--save_total_limit", type=int, default=1,
                       help="Maximum number of checkpoints to keep (default: 1, set to None/-1 to save all)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Load dataset
    logging.info("Loading dataset...")
    
    if args.raw_problem_solution_format:
        # Load raw dataset with problem/solution fields
        logging.info("Loading raw dataset with problem/solution format...")
        
        if args.dataset_path:
            raw_dataset = load_from_disk(args.dataset_path)
        elif args.hf_dataset_name:
            raw_dataset = load_dataset(args.hf_dataset_name, token=args.hf_token, split="train")
        else:
            raise ValueError("Must specify either dataset_path or hf_dataset_name")
        
        # Validate fields exist
        if len(raw_dataset) > 0:
            fields = list(raw_dataset[0].keys())
            logging.info("Dataset fields: %s", fields)
            
            if args.problem_field not in fields:
                raise ValueError(
                    f"Problem field '{args.problem_field}' not found in dataset. "
                    f"Available fields: {fields}"
                )
            if args.solution_field not in fields:
                raise ValueError(
                    f"Solution field '{args.solution_field}' not found in dataset. "
                    f"Available fields: {fields}"
                )
        
        # Filter by indices if provided
        if args.indices_file:
            logging.info("Filtering dataset by indices from: %s", args.indices_file)
            raw_dataset = filter_dataset_by_indices_file(raw_dataset, args.indices_file)
        
        # Preprocess raw dataset
        logging.info("Preprocessing raw dataset (applying chat template)...")
        dataset = preprocess_raw_problem_solution_dataset(
            raw_dataset,
            tokenizer_name=args.model_name,
            problem_field=args.problem_field,
            solution_field=args.solution_field
        )
        logging.info("Successfully preprocessed raw dataset")
    else:
        # Load pre-formatted mixed dataset
        logging.info("Loading pre-formatted mixed dataset...")
        dataset = load_mixed_dataset(
            args.dataset_path,
            args.hf_dataset_name,
            args.hf_token
        )
        logging.info("Successfully loaded mixed dataset")
    
    # Handle save_total_limit (convert -1 or 0 to None for unlimited)
    save_total_limit = args.save_total_limit if args.save_total_limit > 0 else None
    
    # Train model
    model, ckpt_dir = train_mixed_dataset(
        dataset=dataset,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        effective_batch_size=args.effective_batch_size,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        checkpoint_name=args.checkpoint_name,
        output_model_name=args.output_model_name,
        output_base_dir=args.output_base_dir,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        save_total_limit=save_total_limit,
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Dataset size: {len(dataset)} examples")
    if "is_contamination" in dataset.column_names:
        contamination_count = sum(dataset['is_contamination'])
        contamination_pct = contamination_count / len(dataset) * 100 if len(dataset) > 0 else 0
        print(f"Contamination: {contamination_count} examples ({contamination_pct:.2f}%)")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
