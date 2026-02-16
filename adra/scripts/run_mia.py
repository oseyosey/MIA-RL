#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adra.mia.runner import RunConfig, run
from adra.mia.attacks import AttackConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run membership inference attacks on a dataset")

    # Models
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--reference-model", default=None, help="Optional reference HF model for reference-based attack")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--cache-dir", default=None)

    # Data
    p.add_argument("--dataset", required=True, help="HF dataset id or local .json/.jsonl path (expects 'id' and 'messages' when present)")
    p.add_argument("--split", default="test")
    p.add_argument("--text-key", default=None)
    p.add_argument("--problem-key", default="problem", help="Field name for the problem/question text")
    p.add_argument("--solution-key", default="solution", help="Field name for the solution/answer text")
    p.add_argument("--limit", type=int, default=None)

    # Attack
    p.add_argument("--attack", default="loss", choices=["loss", "ref", "reference", "zlib", "min_k", "min-k", "min_k++", "gradnorm"])
    p.add_argument("--min-k", dest="min_k", type=float, default=0.2)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--minkpp-k", dest="minkpp_k", type=float, default=0.2)
    p.add_argument("--gradnorm-p", dest="gradnorm_p", type=float, default=float("inf"))

    # Output
    p.add_argument("--output", required=True, help="Path to write per-sample scores (jsonl). Fields: {id, score}")
    p.add_argument("--add-generation-prompt", action="store_true", help="Append assistant prefix per chat template when formatting messages")
    p.add_argument("--verbose", action="store_true", help="Print first few chat-templated texts for inspection")
    p.add_argument("--max-length", type=int, default=None, help="Override maximum sequence length (tokens) for tokenization and sliding window")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = RunConfig(
        model_name_or_path=args.model,
        reference_model=args.reference_model,
        device=args.device,
        device_map=args.device_map,
        revision=args.revision,
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        split=args.split,
        text_key=args.text_key,
        problem_key=args.problem_key,
        solution_key=args.solution_key,
        limit=args.limit,
        attack=args.attack,
        output_path=args.output,
        add_generation_prompt=args.add_generation_prompt,
        verbose=args.verbose,
        max_length=args.max_length,
    )
    atk_cfg = AttackConfig(
        min_k_k=args.min_k,
        min_k_window=args.window,
        min_k_stride=args.stride,
        minkpp_k=args.minkpp_k,
        gradnorm_p=args.gradnorm_p,
    )
    out_path = run(run_cfg, atk_cfg)
    print(json.dumps({"output": out_path}))


if __name__ == "__main__":
    main()



