#!/usr/bin/env python3
"""
Format MIA AUROC scores from CSV into a table-ready format.

Usage:
    python -m adra.utils.format_mia_aurocs <csv_path> [--output OUTPUT]

Examples:
    python -m adra.utils.format_mia_aurocs mia_aurocs.csv
    python -m adra.utils.format_mia_aurocs /path/to/mia_aurocs.csv --output formatted_results.txt
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict


# Desired table order → CSV key base
ROW_MAPPING = [
    ("lexical_jaccard_avg", "adra_lexical_jaccard_sim_avg"),
    ("lexical_jaccard_best", "adra_lexical_jaccard_sim_best"),
    ("lexical_lcs_avg", "adra_lexical_lcs_avg"),
    ("lexical_lcs_best", "adra_lexical_lcs_best"),
    ("lexical_lcs_ratio_avg", "adra_lexical_lcs_ratio_avg"),
    ("lexical_lcs_ratio_best", "adra_lexical_lcs_ratio_best"),
    ("lexical_lcs_ratio_cand_avg", "adra_lexical_lcs_ratio_cand_avg"),
    ("lexical_lcs_ratio_cand_best", "adra_lexical_lcs_ratio_cand_best"),
    ("lexical_coverage_avg", "adra_lexical_ngram_coverage_avg"),
    ("lexical_coverage_best", "adra_lexical_ngram_coverage_best"),
    ("lexical_coverage_ref_avg", "adra_lexical_ngram_coverage_ref_avg"),
    ("lexical_coverage_ref_best", "adra_lexical_ngram_coverage_ref_best"),
    ("lexical_token_overlap_avg", "adra_lexical_token_overlap_cand_avg"),
    ("lexical_token_overlap_best", "adra_lexical_token_overlap_cand_best"),
    ("lexical_token_overlap_ref_avg", "adra_lexical_token_overlap_ref_avg"),
    ("lexical_token_overlap_ref_best", "adra_lexical_token_overlap_ref_best"),
    ("embed_cosine_avg", "adra_q3_8b_embedding_cosine_sim_avg"),
    ("embed_cosine_best", "adra_q3_8b_embedding_cosine_sim_best"),
]


def load_aurocs(csv_path: Path) -> Dict[str, float]:
    """
    Load AUROC values from CSV file.
    
    Args:
        csv_path: Path to CSV file containing attack and auroc columns
        
    Returns:
        Dictionary mapping attack names to AUROC values
    """
    values = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values[row["attack"]] = float(row["auroc"])
    return values


def format_results(values: Dict[str, float], skip_missing: bool = True) -> str:
    """
    Format AUROC values into paste-ready rows.
    
    Args:
        values: Dictionary mapping attack names to AUROC values
        skip_missing: If True, skip missing pairs; if False, raise error
        
    Returns:
        Formatted string with one line per attack in the format: value/original_value
    """
    lines = []
    missing_pairs = []
    
    for display_name, key in ROW_MAPPING:
        v = values.get(key)
        v_orig = values.get(f"{key}_original")

        if v is None or v_orig is None:
            if skip_missing:
                missing_pairs.append(key)
                continue
            else:
                raise KeyError(f"Missing pair for {key} (looking for '{key}' and '{key}_original')")

        lines.append(f"{v:.2f}/{v_orig:.2f}")
    
    if missing_pairs and skip_missing:
        print(f"⚠️  Skipped {len(missing_pairs)} missing pairs: {', '.join(missing_pairs)}", file=sys.stderr)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Format MIA AUROC scores from CSV into table-ready format'
    )
    parser.add_argument(
        'csv_path',
        type=Path,
        help='Path to CSV file containing AUROC scores'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Error on missing pairs instead of skipping them'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.csv_path.exists():
        parser.error(f"File does not exist: {args.csv_path}")
    
    if not args.csv_path.is_file():
        parser.error(f"Not a file: {args.csv_path}")
    
    try:
        # Load AUROC values
        values = load_aurocs(args.csv_path)
        
        # Format results
        formatted_output = format_results(values, skip_missing=not args.strict)
        
        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output + '\n')
            print(f"✅ Formatted results written to {args.output}")
        else:
            print(formatted_output)
            
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to process CSV: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
