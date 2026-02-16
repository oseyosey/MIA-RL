#!/usr/bin/env python3
"""Example: TRL SFT fine-tuning with chat templating.

Run with:
    python adra/produce_checkpoint_trl_example.py

It fine-tunes the Qwen-2.5-3B model for one epoch on a couple of toy
conversational snippets, using the TRL `SFTTrainer` under the hood and the
model's chat template to format each input.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure repository root (three levels up from this script) is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adra.utils_rl import produce_checkpoint, CheckpointConfig  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Tiny toy conversation snippets – the chat template will wrap these as user
    # messages automatically.
    texts = [
        "ancient ruins hidden beneath dense jungle",
        "mysteries waiting for explorers to find and explore",
        "quiet breeze whispers through the leaves",
        "soft moonlight dances over crumbling stone"
    ]

    cfg = CheckpointConfig(
        # Qwen-2.5-3B Instruct weights on the HF Hub
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=64,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        save_total_limit=2,
        output_base_dir="model_checkpoints",
        bf16=True,
        warmup_steps=1,
        checkpoint_name="toy-v1.3.3",
        # --- new flags ---
        use_trl=True,
        apply_chat_template=True,
        add_generation_prompt=False,  # whether to append an assistant prompt
        # You may also enable LoRA for memory-efficient finetuning:
        # use_lora=True,
        # lora_r=8,
    )

    model, ckpt_dir = produce_checkpoint(texts, cfg)

    print(f"\nCheckpoint saved to: {ckpt_dir}")
    print("Loaded model type:", type(model))


if __name__ == "__main__":
    main() 