#!/usr/bin/env python3
"""TRL SFT fine-tuning of `allenai/tulu-2-7b` on the AIME dataset.

This script fine-tunes tulu-2-7b on a subset of the AIME 2021-2025 dataset
(original or paraphrased), sampled deterministically with `subset_seed=42`.

Run with:
    python adra/scripts/produce_checkpoints/produce_checkpoint_trl_tulu2-7b_aime_m4.5.py
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

def load_aime_original_subset(subset_size: int = 120, subset_seed: int = 42) -> List[str]:
    """Load the AIME 2021-2025 original dataset and return chat-templated strings."""

    ds = load_dataset("osieosie/AIME-2021-2025-Cleaned", split="train")

    # Deterministic subsampling (without replacement) identical to the preprocessing script.
    if subset_size is not None and subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices = rng.sample(range(len(ds)), subset_size)
        sampled_indices.sort()  # stable ordering
        ds = ds.select(sampled_indices)
        logging.info("Subsampled %d / %d examples with seed %d", len(ds), len(load_dataset("osieosie/AIME-2021-2025-Cleaned", split="train")), subset_seed)
    else:
        logging.info("Using full dataset with %d examples (no subsampling applied).", len(ds))

    tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-7b", trust_remote_code=True)

    texts: List[str] = []
    for ex in ds:
        messages = [
            {"role": "user", "content": str(ex["problem"]).strip()},
            {"role": "assistant", "content": str(ex["solution"]).strip()},
        ]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(templated)

    return texts

def load_aime_paraphrased_subset(subset_size: int = 120, subset_seed: int = 42) -> List[str]:
    """Load the AIME 2021-2025 paraphrased dataset and return chat-templated strings."""

    ds = load_dataset("osieosie/AIME-2021-2025-Paraphrased-Gemini-2.5-Flash-v3.0", split="train")

    # Deterministic subsampling (without replacement) identical to the preprocessing script.
    if subset_size is not None and subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices = rng.sample(range(len(ds)), subset_size)
        sampled_indices.sort()  # stable ordering
        ds = ds.select(sampled_indices)
        logging.info("Subsampled %d / %d examples with seed %d", len(ds), len(load_dataset("osieosie/AIME-2021-2025-Paraphrased-Gemini-2.5-Flash-v3.0", split="train")), subset_seed)
    else:
        logging.info("Using full dataset with %d examples (no subsampling applied).", len(ds))

    tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-7b", trust_remote_code=True)

    texts: List[str] = []
    for ex in ds:
        messages = [
            {"role": "user", "content": str(ex["problem"]).strip()},
            {"role": "assistant", "content": str(ex["solution"]).strip()},
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
    subset_size = 120
    subset_seed = 42

    #* Choose between AIME original and paraphrased dataset *#
    # texts = load_aime_original_subset(subset_size=subset_size, subset_seed=subset_seed)
    texts = load_aime_paraphrased_subset(subset_size=subset_size, subset_seed=subset_seed)

    # ------------------------------------------------------------------
    # Hyper-parameters (identical to the AIME script for consistency)
    # ------------------------------------------------------------------
    num_epochs: int = 3
    effective_batch_size: int = 32  # global batch size across all GPUs
    batch_size: int = 16  # per-GPU micro-batch size (memory bound)

    # Derive gradient accumulation so that: global = world_size × micro × accum
    world_size = int(os.environ.get("WORLD_SIZE", 1)) or 1
    gradient_accumulation_steps = max(1, effective_batch_size // (batch_size * world_size))

    if (batch_size * world_size * gradient_accumulation_steps) != effective_batch_size:
        logging.warning(
            "Effective batch size %d cannot be met exactly; using %d",
            effective_batch_size,
            batch_size * world_size * gradient_accumulation_steps,
        )

    learning_rate: float = 2e-5
    max_seq_len: int = 8192

    # Compute warm-up steps as 0 for these relatively small runs
    steps_per_epoch = math.ceil(len(texts) / batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = 0  # small dataset ⇒ warm-up unnecessary

    logging.info("Dataset size: %d samples", len(texts))
    logging.info("Total optimisation steps ≈ %d (warm-up: %d)", total_steps, warmup_steps)

    cfg = CheckpointConfig(
        model_name_or_path="allenai/tulu-2-7b",
        max_seq_length=max_seq_len,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=1,
        output_base_dir="model_checkpoints",
        fp16=True,
        checkpoint_name="aime-paraphrased-sft-120-s42-m4.5-e3-lr2e-5",
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
