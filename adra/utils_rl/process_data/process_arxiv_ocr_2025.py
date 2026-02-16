"""
Process and sample the SlowGuess/Arxiv_2025_OCR dataset.

This script:
1. Lists all full.md files in the dataset repo (lexicographic order).
2. Extracts up to N papers (default: 1000).
3. Cleans the markdown text:
   - Removes content before the first Introduction section header
     (supports numeric/roman variants).
   - Drops simple comments and macro/definition lines.
   - Truncates before a bibliography/references section.
4. Outputs a dataset with fields: text, meta.
5. Saves locally and optionally pushes to HuggingFace Hub.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm


INTRO_HEADING_RE = re.compile(
    r"(?im)^#{1,6}\s*(?:\d+(?:\.\d+)?|[ivxlcdm]+)\s*\.?\s*introduction\b.*$"
)
REFERENCES_HEADING_RE = re.compile(
    r"(?im)^#{1,6}\s*(references|bibliography|reference)\b.*$"
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
LATEX_COMMENT_RE = re.compile(r"(?m)^\s*%.*$")
LATEX_MACRO_RE = re.compile(
    r"(?im)^\s*\\(newcommand|renewcommand|def|DeclareMathOperator|newtheorem)\b.*$"
)


def _clean_text(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_COMMENT_RE.sub("", text)
    text = LATEX_COMMENT_RE.sub("", text)
    text = LATEX_MACRO_RE.sub("", text)

    intro_match = INTRO_HEADING_RE.search(text)
    if not intro_match:
        return None

    text = text[intro_match.start() :]
    refs_match = REFERENCES_HEADING_RE.search(text)
    if refs_match:
        text = text[: refs_match.start()]

    cleaned = text.strip()
    return cleaned or None


def _extract_arxiv_id(path: str) -> Optional[str]:
    parts = path.split("/")
    if len(parts) < 2:
        return None
    return parts[-2]


def process_arxiv_ocr_2025(
    dataset_path: str = "SlowGuess/Arxiv_2025_OCR",
    output_dir: Optional[str] = None,
    max_examples: int = 1000,
    shuffle: bool = False,
    seed: int = 42,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
) -> Dataset:
    disable_progress_bars()
    print(f"Scanning repo files from {dataset_path}...")
    repo_files = sorted(list_repo_files(dataset_path, repo_type="dataset"))
    full_md_paths = [path for path in repo_files if path.endswith("/full.md")]
    print(f"Found {len(full_md_paths)} full.md files.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(full_md_paths)

    records = []
    with tqdm(total=max_examples, desc="Collected examples") as progress:
        for path in full_md_paths:
            if max_examples and len(records) >= max_examples:
                break

            arxiv_id = _extract_arxiv_id(path)
            if not arxiv_id:
                continue

            local_path = hf_hub_download(
                repo_id=dataset_path,
                repo_type="dataset",
                filename=path,
            )
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            cleaned = _clean_text(raw_text)
            if not cleaned:
                continue

            meta = {"arxiv_id": arxiv_id, "year": 2025}
            records.append({"text": cleaned, "meta": json.dumps(meta)})
            progress.update(1)

    dataset = Dataset.from_list(records)
    print(f"Final dataset size: {len(dataset)}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving dataset to: {output_path}")
        dataset.save_to_disk(str(output_path))
        print("Saved successfully.")

    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        print(f"Pushing dataset to HuggingFace Hub: {repo_id}")
        dataset.push_to_hub(repo_id, private=private)
        print("Pushed to HF Hub successfully.")

    if len(dataset) > 0:
        ex = dataset[0]
        print("\n=== Sample from processed dataset ===")
        print(f"text preview: {ex.get('text', '')[:300]}")
        print(f"meta: {ex.get('meta', '')}")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process SlowGuess/Arxiv_2025_OCR and extract 2025 arXiv OCR papers"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="SlowGuess/Arxiv_2025_OCR",
        help="Path to the dataset on HuggingFace Hub",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to save the processed dataset",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Maximum number of papers to include (default: 1000)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle file list before sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
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

    process_arxiv_ocr_2025(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
    )


if __name__ == "__main__":
    main()

