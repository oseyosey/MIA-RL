from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_id_to_score(rows: List[dict]) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    for i, row in enumerate(rows):
        if "id" not in row or "score" not in row:
            raise KeyError(f"Row {i} missing 'id' or 'score' field: {row}")
        rid = row["id"]
        if not isinstance(rid, int):
            raise TypeError(f"Row {i}: 'id' must be int, got {type(rid)}")
        mapping[rid] = float(row["score"])  # last wins if duplicates
    return mapping


def _diff_scores(primary: List[dict], reference_map: Dict[int, float]) -> List[dict]:
    out: List[dict] = []
    for idx, row in enumerate(primary):
        rid = row.get("id")
        if rid is None or not isinstance(rid, int):
            raise TypeError(f"Row {idx}: missing or non-int 'id': {row}")
        s = float(row.get("score", 0.0))
        ref = reference_map.get(rid, 0.0)
        diff = s - ref
        out.append({
            "id": rid,
            "idx": row.get("idx", idx),
            "score": diff,
        })
    return out


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute score differences between two MIA-format JSONL files: "
            "diff = primary.score - reference.score, matched by 'id'."
        )
    )
    p.add_argument("--primary", required=True, type=Path, help="Path to primary JSONL (minuend)")
    p.add_argument("--reference", required=True, type=Path, help="Path to reference JSONL (subtrahend)")
    p.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Optional explicit output path. If not provided, writes alongside the primary file "
            "as '<primary_stem>_minus_<reference_stem>.jsonl'."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.primary.exists():
        raise FileNotFoundError(f"Primary file not found: {args.primary}")
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")

    primary_rows = _read_jsonl(args.primary)
    reference_rows = _read_jsonl(args.reference)

    reference_map = _build_id_to_score(reference_rows)
    diff_rows = _diff_scores(primary_rows, reference_map)

    if args.output is None:
        out_name = f"{args.primary.stem}_ref.jsonl"
        output_path = args.primary.parent / out_name
    else:
        output_path = args.output

    _write_jsonl(output_path, diff_rows)
    print(f"Wrote diff JSONL to: {output_path}")


if __name__ == "__main__":
    main()


