#!/usr/bin/env python3
"""Example script for checkpoint producer/manager.

Run with:
    python adra/produce_checkpoint_example.py

It fine-tunes *gpt2* on two toy sentences for one epoch and logs training
progress both to console and a `train.log` file located inside the created
checkpoint directory.
"""

import logging
import sys
from pathlib import Path

# Ensure repository root (three levels up) on PYTHONPATH for clean imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adra.utils_rl import produce_checkpoint, CheckpointConfig  # noqa: E402


def main() -> None:
    # Basic console logging; file logging is handled by CheckpointManager itself.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    texts = [
        "Hello world! This is a tiny fine-tuning run.",
        "Goodbye world! GPT-2 should now know about both greetings.",
    ]

    config = CheckpointConfig(
        model_name_or_path="gpt2",
        max_seq_length=32,
        num_train_epochs=5,
        save_total_limit=1,
        per_device_train_batch_size=1,
        output_base_dir="model_checkpoints",
    )

    model, ckpt_dir = produce_checkpoint(texts, config)

    print(f"\nCheckpoint successfully created at: {ckpt_dir}")
    print("Model loaded:", type(model))


if __name__ == "__main__":
    main() 