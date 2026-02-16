#!/usr/bin/env python
"""
merge_lora.py
--------------
Merge a LoRA adapter into the base model so that the resulting checkpoint no longer
requires PEFT/LoRA for inference.

The script relies only on HuggingFace `transformers` and `peft`.

Example
-------
python merge_lora.py \
    --base_model   /path/to/base_model \
    --lora_adapter /path/to/lora_adapter \
    --output_dir   /path/to/merged_model \
    --dtype        float16
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model weights.")
    parser.add_argument("--base_model", type=str, required=True, help="Path (or HF hub id) to the base model.")
    parser.add_argument("--lora_adapter", type=str, required=True, help="Path to the LoRA adapter folder containing adapter_config.json & adapter_model.safetensors.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write the merged model.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type to load the model weights with. 'auto' uses the default dtype of the checkpoint.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow execution of custom code from HF repo (use with care!).")
    parser.add_argument("--safe_serialization", action="store_true", help="Save weights in Safetensors format instead of PyTorch .bin.")
    return parser.parse_args()


def str_dtype_to_torch(dtype_str: str) -> Optional[torch.dtype]:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": None,
    }
    return mapping[dtype_str]


def main() -> None:
    args = parse_args()

    torch_dtype = str_dtype_to_torch(args.dtype)
    load_kwargs = {"trust_remote_code": args.trust_remote_code}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    print(f"[INFO] Loading base model from {args.base_model} …")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    print(f"[INFO] Loading LoRA adapter from {args.lora_adapter} …")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_adapter, is_trainable=False)

    print("[INFO] Merging LoRA weights into the base model (this may take a while) …")
    merged_model = lora_model.merge_and_unload()

    output_path = Path(args.output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving merged model to {output_path} …")
    merged_model.save_pretrained(output_path, safe_serialization=args.safe_serialization)

    # Copy / save tokenizer & generation config for completeness.
    print("[INFO] Copying tokenizer files …")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(output_path)

    for aux_file in ["generation_config.json", "generation_config.yaml", "special_tokens_map.json"]:
        src = Path(args.base_model) / aux_file
        if src.exists():
            shutil.copy(src, output_path / aux_file)

    print("[SUCCESS] LoRA adapter merged successfully!")


if __name__ == "__main__":
    main() 