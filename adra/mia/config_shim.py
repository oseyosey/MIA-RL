from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .runner import RunConfig
from .attacks import AttackConfig


def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_mimir_like(path: str, *, dataset: str, split: str = "test", text_key: str | None = None) -> Tuple[RunConfig, AttackConfig]:
    """
    Load a MIMIR-like JSON config and return (RunConfig, AttackConfig).
    Only maps a small subset of fields we actually use.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_cfg = RunConfig(
        model_name_or_path=cfg.get("base_model") or cfg.get("scoring_model_name") or "gpt2",
        device=_get(cfg, "env_config.device", "cuda:0"),
        device_map=_get(cfg, "env_config.device_map", None),
        cache_dir=_get(cfg, "env_config.cache_dir", None),
        dataset=dataset,
        split=split,
        text_key=text_key,
        # You can add: add_generation_prompt based on dataset if needed.
    )

    atk_cfg = AttackConfig(
        min_k_k=_get(cfg, "min_k.k", 0.2),
        min_k_window=_get(cfg, "min_k.window", 1),
        min_k_stride=_get(cfg, "min_k.stride", 1),
        minkpp_k=_get(cfg, "min_kpp.k", 0.2),
        gradnorm_p=_get(cfg, "gradnorm.p", float("inf")),
    )

    return run_cfg, atk_cfg


