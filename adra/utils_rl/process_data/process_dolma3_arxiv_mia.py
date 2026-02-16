"""
Process and combine member/non-member arXiv datasets for MIA.

This script:
1. Loads member and non-member HF datasets.
2. Samples up to N examples from each split (default: 1000).
3. Adds fields:
   - label (1=member, 0=non-member)
   - id (from meta.arxiv_id)
4. Truncates text to the first max_tokens tokens (default: 2048).
5. Saves locally and optionally pushes to HuggingFace Hub.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


def _parse_meta(meta: Any) -> Dict[str, Any]:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        meta = meta.strip()
        if not meta:
            return {}
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return {}


def _truncate_text(text: str, tokenizer, max_tokens: int) -> str:
    if not text:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    token_ids = token_ids[:max_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _prepare_split(
    dataset_path: str,
    split: str,
    max_examples: int,
    label: int,
    tokenizer,
    max_tokens: int,
    shuffle: bool,
    seed: int,
) -> Dataset:
    ds = load_dataset(dataset_path, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    def _process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        meta = _parse_meta(example.get("meta"))
        arxiv_id = meta.get("arxiv_id")
        if not arxiv_id:
            raise ValueError(f"Missing meta.arxiv_id in dataset {dataset_path}")
        text = str(example.get("text", ""))
        text = _truncate_text(text, tokenizer, max_tokens=max_tokens)
        return {
            "text": text,
            # Store meta as JSON string to avoid schema mismatch across datasets.
            "meta": json.dumps(meta, ensure_ascii=False),
            "label": label,
            "id": str(arxiv_id),
        }

    return ds.map(
        _process_example,
        remove_columns=ds.column_names,
        desc=f"Processing {dataset_path}",
    )


def process_dolma3_arxiv_mia(
    member_dataset_path: str = "osieosie/dolma3-arxiv-3k-seed42",
    nonmember_dataset_path: str = "osieosie/arxiv-ocr-2025-1k",
    split: str = "train",
    max_examples: int = 1000,
    max_tokens: int = 2048,
    tokenizer_name: str = "allenai/tulu-2-7b",
    shuffle: bool = True,
    seed: int = 42,
    output_dir: Optional[str] = None,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
) -> Dataset:
    print("=== Processing dolma3 arXiv MIA dataset ===")
    print(f"Member dataset: {member_dataset_path} (label=1)")
    print(f"Non-member dataset: {nonmember_dataset_path} (label=0)")
    print(f"Split: {split}")
    print(f"Max examples per dataset: {max_examples}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Max tokens: {max_tokens}")
    print(f"Shuffle: {shuffle} (seed={seed})")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    members = _prepare_split(
        dataset_path=member_dataset_path,
        split=split,
        max_examples=max_examples,
        label=1,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        shuffle=shuffle,
        seed=seed,
    )
    nonmembers = _prepare_split(
        dataset_path=nonmember_dataset_path,
        split=split,
        max_examples=max_examples,
        label=0,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        shuffle=shuffle,
        seed=seed,
    )

    combined = concatenate_datasets([members, nonmembers])
    if shuffle:
        combined = combined.shuffle(seed=seed)
    print(f"Combined dataset size: {len(combined)}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving dataset to: {output_path}")
        combined.save_to_disk(str(output_path))
        print("Saved successfully.")

    if push_to_hub:
        hub_repo_id = repo_id or "osieosie/dolma3-arxiv-mia-1k"
        print(f"Pushing dataset to HuggingFace Hub: {hub_repo_id}")
        combined.push_to_hub(hub_repo_id, private=private)
        print("Pushed to HF Hub successfully.")

    if len(combined) > 0:
        ex = combined[0]
        print("\n=== Sample from processed dataset ===")
        print(f"text preview: {ex.get('text', '')[:300]}")
        print(f"meta: {ex.get('meta', {})}")
        print(f"label: {ex.get('label')}")
        print(f"id: {ex.get('id')}")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine member/non-member arXiv datasets for MIA with token truncation"
    )
    parser.add_argument(
        "--member-dataset-path",
        type=str,
        default="osieosie/dolma3-arxiv-3k-seed42",
        help="Member dataset path on HuggingFace Hub",
    )
    parser.add_argument(
        "--nonmember-dataset-path",
        type=str,
        default="osieosie/arxiv-ocr-2025-1k",
        help="Non-member dataset path on HuggingFace Hub",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Maximum number of examples per dataset (default: 1000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to keep from text (default: 2048)",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="allenai/tulu-2-7b",
        help="Tokenizer name for token truncation (default: allenai/tulu-2-7b)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle each dataset before sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to save the processed dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the processed dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repository ID for HuggingFace Hub (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repository private",
    )

    args = parser.parse_args()

    process_dolma3_arxiv_mia(
        member_dataset_path=args.member_dataset_path,
        nonmember_dataset_path=args.nonmember_dataset_path,
        split=args.split,
        max_examples=args.max_examples,
        max_tokens=args.max_tokens,
        tokenizer_name=args.tokenizer_name,
        shuffle=args.shuffle,
        seed=args.seed,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
    )


if __name__ == "__main__":
    main()

