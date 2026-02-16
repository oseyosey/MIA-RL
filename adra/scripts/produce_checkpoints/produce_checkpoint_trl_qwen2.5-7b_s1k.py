#!/usr/bin/env python3
"""Example: TRL SFT fine-tuning of `Qwen/Qwen2.5-7B-Instruct` on the s1K-1.1 dataset.

This script trains the Qwen 2.5 7B Instruct model on the simplescaling/s1K-1.1 dataset,
using the 'question' field for user prompts and 'deepseek_attempt' field for assistant responses.

Run with:
    python adra/scripts/produce_checkpoint_trl_qwen2.5-7b_s1k.py
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer  # local import keeps deps minimal

# -----------------------------------------------------------------------------
# Ensure repository root (three levels up from this script) is on PYTHONPATH
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adra.utils_rl import CheckpointConfig, produce_checkpoint  # noqa: E402

# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

def load_s1k_subset_deepseek_r1(subset_size: int | None = None, subset_seed: int = 42) -> List[str]:
    """Load the s1K-1.1 dataset and return chat-templated strings.

    Args:
        subset_size: Number of examples to sample. If None, uses full dataset.
        subset_seed: Random seed for deterministic subsampling.

    Returns:
        List of chat-templated strings ready for training.
    """

    ds = load_dataset("simplescaling/s1K-1.1", split="train")

    # Deterministic subsampling (without replacement) if requested
    if subset_size is not None and subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices = rng.sample(range(len(ds)), subset_size)
        sampled_indices.sort()  # stable ordering
        ds = ds.select(sampled_indices)
        logging.info("Subsampled %d / %d examples with seed %d", len(ds), len(load_dataset("simplescaling/s1K-1.1", split="train")), subset_seed)
    else:
        logging.info("Using full dataset with %d examples (no subsampling applied).", len(ds))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    texts: List[str] = []
    for ex in ds:
        messages = [
            {"role": "user", "content": str(ex["question"]).strip()},
            {"role": "assistant", "content": str(ex["deepseek_attempt"]).strip()},
        ]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(templated)

    return texts

def load_s1k_subset_gemini(subset_size: int | None = None, subset_seed: int = 42) -> List[str]:
    """Load the s1K-1.1 dataset and return chat-templated strings.

    Args:
        subset_size: Number of examples to sample. If None, uses full dataset.
        subset_seed: Random seed for deterministic subsampling.

    Returns:
        List of chat-templated strings ready for training.
    """

    ds = load_dataset("simplescaling/s1K-1.1", split="train")

    # Deterministic subsampling (without replacement) if requested
    if subset_size is not None and subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices = rng.sample(range(len(ds)), subset_size)
        sampled_indices.sort()  # stable ordering
        ds = ds.select(sampled_indices)
        logging.info("Subsampled %d / %d examples with seed %d", len(ds), len(load_dataset("simplescaling/s1K-1.1", split="train")), subset_seed)
    else:
        logging.info("Using full dataset with %d examples (no subsampling applied).", len(ds))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    texts: List[str] = []
    for ex in ds:
        messages = [
            {"role": "user", "content": str(ex["question"]).strip()},
            {"role": "assistant", "content": str(ex["gemini_attempt"]).strip()},
        ]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(templated)

    return texts


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")

    # --------------------------------------------------------------
    # Load dataset and build chat-templated training strings
    # --------------------------------------------------------------
    subset_size = None  # Set to None for full dataset, or specify a number for subset
    subset_seed = 42

    #texts = load_s1k_subset_deepseek_r1(subset_size=subset_size, subset_seed=subset_seed)
    texts = load_s1k_subset_gemini(subset_size=subset_size, subset_seed=subset_seed)

    # ------------------------------------------------------------------
    # Hyper-parameters
    # ------------------------------------------------------------------
    num_epochs: int = 1
    effective_batch_size: int = 64  # global batch size across all GPUs
    batch_size: int = 8  # per-GPU micro-batch size (memory bound)

    # Derive gradient accumulation so that: global = world_size × micro × accum
    world_size = int(os.environ.get("WORLD_SIZE", 1)) or 1
    gradient_accumulation_steps = max(1, effective_batch_size // (batch_size * world_size))

    if (batch_size * world_size * gradient_accumulation_steps) != effective_batch_size:
        logging.warning(
            "Effective batch size %d cannot be met exactly; using %d",
            effective_batch_size,
            batch_size * world_size * gradient_accumulation_steps,
        )

    learning_rate: float = 5e-6
    max_seq_len: int = 8192

    # Compute warm-up steps as 0 for these relatively small runs
    steps_per_epoch = math.ceil(len(texts) / batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = 0  # small dataset ⇒ warm-up unnecessary

    logging.info("Dataset size: %d samples", len(texts))
    logging.info("Total optimisation steps ≈ %d (warm-up: %d)", total_steps, warmup_steps)

    cfg = CheckpointConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_len,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=1,
        output_base_dir="model_checkpoints",
        lr_scheduler_type="linear",
        fp16=True,
        checkpoint_name="qwen2.5-7b-s1k-sft-gemini-full-s42-e1-lr5e-6-bs64-schedlinear",
        # --- TRL / chat-template flags ---
        use_trl=True,
        apply_chat_template=False,  # we already applied manually above
        add_generation_prompt=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    model, ckpt_dir = produce_checkpoint(texts, cfg)

    print(f"\nCheckpoint saved to: {ckpt_dir}")
    print("Loaded model type:", type(model))


if __name__ == "__main__":
    main()
