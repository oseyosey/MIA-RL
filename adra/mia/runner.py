from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .attacks import AttackConfig, build_attack
from .model_adapter import HFModelAdapter, HFReferenceModelAdapter, ModelLoadConfig
from .datasets import DatasetConfig, load_examples


@dataclass
class RunConfig:
    # Models
    model_name_or_path: str
    reference_model: Optional[str] = None
    device: str = "cuda:0"
    device_map: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None

    # Data
    dataset: str = ""
    split: str = "test"
    text_key: Optional[str] = None
    problem_key: str = "problem"
    solution_key: str = "solution"
    limit: Optional[int] = None
    add_generation_prompt: bool = False
    verbose: bool = False
    max_length: Optional[int] = None

    # Attack
    attack: str = "loss"

    # Output
    output_path: str = "mia_scores.jsonl"


def run(config: RunConfig, attack_cfg: Optional[AttackConfig] = None) -> str:
    attack_cfg = attack_cfg or AttackConfig()

    # Load models
    adapter = HFModelAdapter(
        ModelLoadConfig(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            device_map=config.device_map,
            revision=config.revision,
            cache_dir=config.cache_dir,
        )
    )

    # Apply max_length override if provided
    if config.max_length is not None:
        adapter.tokenizer.model_max_length = int(config.max_length)
        adapter.max_length = int(config.max_length)
        adapter.stride = max(1, adapter.max_length // 2)

    ref_adapter = None
    if config.reference_model:
        ref_adapter = HFReferenceModelAdapter(
            ModelLoadConfig(
                model_name_or_path=config.reference_model,
                device=config.device,
                device_map=config.device_map,
                revision=config.revision,
                cache_dir=config.cache_dir,
            )
        )

    attacker = build_attack(config.attack, adapter, attack_cfg, reference=ref_adapter)

    # Load data
    ds_cfg = DatasetConfig(
        name_or_path=config.dataset, 
        split=config.split, 
        text_key=config.text_key,
        problem_key=config.problem_key,
        solution_key=config.solution_key
    )
    examples = load_examples(
        ds_cfg,
        adapter.tokenizer,
        add_generation_prompt=config.add_generation_prompt,
        limit=config.limit,
    )

    # Optional preview of chat-templated texts
    if config.verbose:
        preview_n = min(1, len(examples))
        print(f"[MIA] Loaded {len(examples)} examples. Preview of chat-templated texts (first {preview_n}):")
        for i in range(preview_n):
            ex = examples[i]
            print(f"id={ex['id']}")
            print(ex["text"])  # full templated text
            print("-----")

    # Score
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(examples):
            score = attacker.score(ex["text"])  # already chat-templated when messages are present
            f.write(json.dumps({"idx": idx, "id": ex["id"], "score": float(score)}) + "\n") 
            # idx is the index of the example in the current dataset
            # id is the id of the example in the original dataset
            # score is the score of the example

    return str(out_path)


