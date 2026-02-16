#!/usr/bin/env python3
"""
Extract AUROC scores from all MIA attack metrics files in a directory.

Usage:
    python -m adra.utils.extract_mia_aurocs <metrics_dir> [--output OUTPUT] [--format FORMAT]

Examples:
    python -m adra.utils.extract_mia_aurocs /path/to/metrics/folder
    python -m adra.utils.extract_mia_aurocs /path/to/metrics/folder --output results.csv --format csv
    python -m adra.utils.extract_mia_aurocs /path/to/metrics/folder --output results.jsonl --format jsonl
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict


def extract_aurocs(metrics_dir: Path) -> List[Dict[str, float]]:
    """
    Extract AUROC scores from all *_metrics.json files in the directory.
    
    Args:
        metrics_dir: Path to directory containing metrics JSON files
        
    Returns:
        List of dictionaries with 'attack' and 'auroc' keys
    """
    results = []
    
    # Find all metrics JSON files
    metrics_files = sorted(metrics_dir.glob("*_metrics.json"))
    
    if not metrics_files:
        print(f"Warning: No *_metrics.json files found in {metrics_dir}")
        return results
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Extract attack name (remove _metrics.json suffix)
            attack_name = metrics_file.stem.replace('_metrics', '')
            
            # Get AUROC (prefer 'auroc' over 'roc_auc' if both exist)
            auroc = data.get('auroc') or data.get('roc_auc')
            
            if auroc is None:
                print(f"Warning: No AUROC found in {metrics_file.name}")
                continue
            
            # Convert to percentage and round to 2 decimal places
            auroc_percent = round(float(auroc) * 100, 2)
            
            results.append({
                'attack': attack_name,
                'auroc': auroc_percent
            })
            
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {metrics_file.name}: {e}")
        except Exception as e:
            print(f"Error: Failed to process {metrics_file.name}: {e}")
    
    return results


def write_csv(results: List[Dict[str, float]], output_path: Path, top5_mean: float, top10_mean: float):
    """Write results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['attack', 'auroc'])
        writer.writeheader()
        writer.writerows(results)
        # Add summary rows
        writer.writerow({'attack': 'top5_mean', 'auroc': top5_mean})
        writer.writerow({'attack': 'top10_mean', 'auroc': top10_mean})


def write_jsonl(results: List[Dict[str, float]], output_path: Path, top5_mean: float, top10_mean: float):
    """Write results to JSONL file."""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
        # Add summary entries
        f.write(json.dumps({'attack': 'top5_mean', 'auroc': top5_mean}) + '\n')
        f.write(json.dumps({'attack': 'top10_mean', 'auroc': top10_mean}) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Extract AUROC scores from MIA attack metrics files'
    )
    parser.add_argument(
        'metrics_dir',
        type=Path,
        help='Directory containing *_metrics.json files'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output file path (default: <metrics_dir>/mia_aurocs.csv)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'jsonl'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.metrics_dir.exists():
        parser.error(f"Directory does not exist: {args.metrics_dir}")
    
    if not args.metrics_dir.is_dir():
        parser.error(f"Not a directory: {args.metrics_dir}")
    
    # Extract AUROC scores
    results = extract_aurocs(args.metrics_dir)
    
    if not results:
        print("No results to write.")
        return
    
    # Calculate top 5 and top 10 mean accuracy
    aurocs = [r['auroc'] for r in results]
    sorted_aurocs = sorted(aurocs, reverse=True)
    
    # Calculate means
    top5_mean = round(sum(sorted_aurocs[:5]) / min(5, len(sorted_aurocs)), 2) if sorted_aurocs else 0.0
    top10_mean = round(sum(sorted_aurocs[:10]) / min(10, len(sorted_aurocs)), 2) if sorted_aurocs else 0.0
    
    # Determine output path
    if args.output is None:
        output_path = args.metrics_dir / f"mia_aurocs.{args.format}"
    else:
        output_path = args.output
    
    # Write results
    if args.format == 'csv':
        write_csv(results, output_path, top5_mean, top10_mean)
    else:
        write_jsonl(results, output_path, top5_mean, top10_mean)
    
    print(f"✅ Extracted {len(results)} AUROC scores to {output_path}")
    
    # Print summary
    if results:
        print(f"   AUROC range: {min(aurocs):.2f}% - {max(aurocs):.2f}%")
        print(f"   Mean AUROC: {sum(aurocs) / len(aurocs):.2f}%")
        print(f"   Top 5 mean AUROC: {top5_mean:.2f}%")
        print(f"   Top 10 mean AUROC: {top10_mean:.2f}%")


if __name__ == '__main__':
    main()

