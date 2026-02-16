from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any

import json
from datasets import load_dataset, Dataset


@dataclass
class DatasetConfig:
    # Either an HF dataset id or a local json/jsonl path
    name_or_path: str
    split: str = "train"
    text_key: Optional[str] = None
    problem_key: str = "problem"
    solution_key: str = "solution"


def _load_hf(name: str, split: str) -> Dataset:
    try:
        return load_dataset(name, split=split)
    except Exception as err:
        raise ValueError(f"Failed to load HF dataset '{name}:{split}': {err}")


def _load_local(path: str) -> Dataset:
    if path.endswith(".json"):
        return load_dataset("json", data_files=path, split="train")
    if path.endswith(".jsonl"):
        return load_dataset("json", data_files=path, split="train")
    raise ValueError("Only .json or .jsonl supported for local paths")


def load_texts(cfg: DatasetConfig, limit: Optional[int] = None) -> List[str]:
    if cfg.name_or_path.endswith((".json", ".jsonl")):
        ds = _load_local(cfg.name_or_path)
    else:
        ds = _load_hf(cfg.name_or_path, cfg.split)

    # Determine text key if not provided
    if cfg.text_key is None:
        # Heuristics for common datasets; else first string column
        for candidate in ("text", "document", "content", "prompt"):
            if candidate in ds.column_names:
                text_key = candidate
                break
        else:
            # pick first string column
            text_cols = [c for c in ds.column_names if isinstance(ds[c][0], str)]
            if not text_cols:
                raise ValueError("Could not infer text column; please pass --text-key")
            text_key = text_cols[0]
    else:
        text_key = cfg.text_key

    texts = ds[text_key]
    if limit is not None:
        texts = texts[:limit]
    # Ensure Python list[str]
    return [str(x) for x in texts]


# ------------------ Chat-templating support ------------------

FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
    "{% elif message['role'] == 'tool' %}Tool: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)


def ensure_chat_template(tokenizer, template_text: Optional[str] = None) -> None:
    """Set a minimal chat template if tokenizer does not provide one."""
    try:
        needs_template = getattr(tokenizer, "chat_template", None) in (None, "")
    except Exception:
        needs_template = True
    if needs_template:
        template = template_text if template_text else FALLBACK_CHAT_TEMPLATE
        try:
            tokenizer.chat_template = template
        except Exception:
            # Some tokenizers disallow setting; in that case, apply_chat_template may fail.
            pass


def _coerce_messages(val: Any) -> List[Dict[str, Any]]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    raise ValueError("messages field must be a list or a JSON-encoded list")


def load_examples(
    cfg: DatasetConfig,
    tokenizer,
    add_generation_prompt: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load examples returning a list of {"id": id_value, "text": chat_formatted_text}.
    If dataset provides a "messages" column, it is formatted via tokenizer.apply_chat_template.
    Otherwise, falls back to plain text column detection, with id from "id" if available,
    or index-based id as a string.
    """
    if cfg.name_or_path.endswith((".json", ".jsonl")):
        ds = _load_local(cfg.name_or_path)
    else:
        ds = _load_hf(cfg.name_or_path, cfg.split)

    n = len(ds)
    if limit is not None:
        n = min(n, limit)

    outputs: List[Dict[str, Any]] = []

    if "messages" in ds.column_names:
        ensure_chat_template(tokenizer)
        ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))
        msgs_col = ds["messages"]
        for i in range(n):
            raw_msgs = _coerce_messages(msgs_col[i]) # check for "messages" format. 
            text = tokenizer.apply_chat_template(
                raw_msgs, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            outputs.append({"id": ids[i], "text": text})
        return outputs

    # Check for problem/solution format
    if cfg.problem_key in ds.column_names and cfg.solution_key in ds.column_names:
        ensure_chat_template(tokenizer)
        ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))
        problem_col = ds[cfg.problem_key]
        solution_col = ds[cfg.solution_key]
        for i in range(n):
            messages = [
                {"role": "user", "content": str(problem_col[i]).strip()},
                {"role": "assistant", "content": str(solution_col[i]).strip()},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            outputs.append({"id": ids[i], "text": text})
        return outputs

    # Fallback to plain text column
    # Determine text key if not provided
    if cfg.text_key is None:
        for candidate in ("text", "document", "content", "prompt"):
            if candidate in ds.column_names:
                text_key = candidate
                break
        else:
            text_cols = [c for c in ds.column_names if isinstance(ds[c][0], str)]
            if not text_cols:
                raise ValueError("Could not infer text column; please pass --text-key")
            text_key = text_cols[0]
    else:
        text_key = cfg.text_key

    ids = ds["id"] if "id" in ds.column_names else list(range(len(ds)))
    texts = ds[text_key]
    for i in range(n):
        outputs.append({"id": ids[i], "text": str(texts[i])})
    return outputs


