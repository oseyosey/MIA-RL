#!/usr/bin/env python3
"""Example: TRL SFT fine-tuning of `allenai/tulu-2-7b` on the AIME 2024 dataset.

Run with:
    python adra/scripts/produce_checkpoint_trl_tulu2-7b.py

This script loads the "Maxwell-Jia/AIME_2024" dataset from the Hugging Face
Hub, extracts the problem statements, and fine-tunes the
`allenai/tulu-2-7b` model for three epochs using TRL's `SFTTrainer` while
applying the model's chat template to each input. 5 percent of the total
training steps are used for learning-rate warm-up.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

from datasets import load_dataset

# Ensure repository root (three levels up from this script) is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adra.utils_rl import produce_checkpoint, CheckpointConfig  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --------------------------------------------------------------
    # Load dataset and build chat-templated training strings
    # --------------------------------------------------------------
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    # We want:   user = Problem,   assistant = Solution
    from transformers import AutoTokenizer  # local import to keep deps minimal

    tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-7b", trust_remote_code=True)

    texts = []
    for ex in ds:
        messages = [
            {"role": "user", "content": str(ex["Problem"]).strip()},
            {"role": "assistant", "content": str(ex["Solution"]).strip()},
        ]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # assistant content already present
        )
        texts.append(templated)

    # ------------------------------------------------------------------
    # Hyper-parameters
    # ------------------------------------------------------------------
    num_epochs: int = 3
    # Desired global (effective) batch size across all GPUs
    effective_batch_size: int = 8

    # Per-GPU micro-batch size (memory bound)
    batch_size: int = 4  # change if memory allows

    # Derive gradient accumulation so that: global = world_size × micro × accum
    import os
    world_size = int(os.environ.get("WORLD_SIZE", 1)) or 1
    gradient_accumulation_steps = max(1, effective_batch_size // (batch_size * world_size))

    if (batch_size * world_size * gradient_accumulation_steps) != effective_batch_size:
        logging.warning(
            "Effective batch size %d cannot be met exactly; using %d",
            effective_batch_size,
            batch_size * world_size * gradient_accumulation_steps,
        )

    learning_rate: float = 1e-5
    max_seq_len: int = 4096

    # Compute warm-up steps as 5 % of total optimisation steps
    steps_per_epoch = math.ceil(len(texts) / batch_size)
    total_steps = steps_per_epoch * num_epochs
    #warmup_steps = max(1, int(0.05 * total_steps)) # * for small examples, DONT NEED WARMUP.
    warmup_steps = 0

    logging.info("Dataset size: %d samples", len(texts))
    logging.info("Total optimisation steps ≈ %d (warm-up: %d)", total_steps, warmup_steps)

    cfg = CheckpointConfig(
        model_name_or_path="allenai/tulu-2-7b",
        max_seq_length=max_seq_len,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=3,
        output_base_dir="model_checkpoints",
        fp16=True,
        # bf16=True,  # tulu-2-7b benefits from bfloat16 on A100/H100; adjust as needed
        checkpoint_name="aime2024-sft-m1.7.0-e3-lr1e-5",
        # --- TRL / chat-template flags ---
        use_trl=True,
        # We already applied the chat template manually above → disable automatic templating
        apply_chat_template=False,
        add_generation_prompt=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    model, ckpt_dir = produce_checkpoint(texts, cfg)

    print(f"\nCheckpoint saved to: {ckpt_dir}")
    print("Loaded model type:", type(model))


if __name__ == "__main__":
    main() 