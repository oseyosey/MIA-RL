from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _iter_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def _read_scores_from_jsonl(path: Path) -> List[float]:
    scores: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "score" in obj:
                try:
                    scores.append(float(obj["score"]))
                except Exception:
                    continue
    return scores


def _compute_stats(values: List[float]) -> Tuple[int, float, float, float, float]:
    if len(values) == 0:
        return 0, math.nan, math.nan, math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    count = int(arr.size)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if count > 1 else 0.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    return count, mean, std, vmin, vmax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inspect a directory of JSONL files (MIA outputs) and print score statistics per file."
        )
    )
    p.add_argument("--dir", required=True, type=Path, help="Directory to scan")
    p.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern to match files (default: *.jsonl)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.dir
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    files = sorted(_iter_files(root, args.pattern, args.recursive))
    if len(files) == 0:
        print("No files matched.")
        return

    print("file,count,mean,std,min,max")
    for fp in files:
        scores = _read_scores_from_jsonl(fp)
        count, mean, std, vmin, vmax = _compute_stats(scores)
        print(f"{fp},{count},{mean},{std},{vmin},{vmax}")


if __name__ == "__main__":
    main()


