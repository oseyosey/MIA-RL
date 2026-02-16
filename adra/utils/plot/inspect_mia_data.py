#!/usr/bin/env python3
"""
Inspect and visualize MIA (Membership Inference Attack) data from processed parquet files.

This script:
- Plots the distribution of MIA weights for members vs non-members
- Extracts sample examples from both groups and saves them in formatted JSONL
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_parquet_data(parquet_path: str) -> pd.DataFrame:
    """Load parquet file into a pandas DataFrame.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        DataFrame with the loaded data
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    return df


def extract_mia_weights(df: pd.DataFrame) -> tuple[List[float], List[float], str]:
    """Extract MIA weights for members and non-members.
    
    Args:
        df: DataFrame containing the parquet data
        
    Returns:
        Tuple of (member_weights, nonmember_weights, weight_tag)
    """
    member_weights = []
    nonmember_weights = []
    weight_tag = None
    
    for idx, row in df.iterrows():
        extra_info = row.get('extra_info', {})
        if not isinstance(extra_info, dict):
            continue
        
        is_member = extra_info.get('is_member', False)
        mia_weight = extra_info.get('mia_weight')
        
        # Extract weight tag if available
        if weight_tag is None:
            weight_tag = extra_info.get('mia_weight_tag', 'unknown')
        
        if mia_weight is not None:
            if is_member:
                member_weights.append(float(mia_weight))
            else:
                nonmember_weights.append(float(mia_weight))
    
    return member_weights, nonmember_weights, weight_tag


def plot_mia_weight_distribution(
    member_weights: List[float],
    nonmember_weights: List[float],
    weight_tag: str,
    output_path: str,
    figsize: tuple = (10, 6)
):
    """Plot the distribution of MIA weights for members and non-members.
    
    Args:
        member_weights: List of MIA weights for members
        nonmember_weights: List of MIA weights for non-members
        weight_tag: Tag identifying the type of MIA weights
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Create histogram with overlapping distributions
    plt.hist(
        member_weights,
        bins=50,
        alpha=0.6,
        label=f'Members (n={len(member_weights)})',
        color='#2ecc71',  # Green for members
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.hist(
        nonmember_weights,
        bins=50,
        alpha=0.6,
        label=f'Non-members (n={len(nonmember_weights)})',
        color='#e74c3c',  # Red for non-members
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.xlabel('MIA Weight', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'MIA Weight Distribution ({weight_tag})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    member_mean = sum(member_weights) / len(member_weights) if member_weights else 0
    nonmember_mean = sum(nonmember_weights) / len(nonmember_weights) if nonmember_weights else 0
    
    stats_text = (
        f'Member mean: {member_mean:.4f}\n'
        f'Non-member mean: {nonmember_mean:.4f}\n'
        f'Difference: {abs(member_mean - nonmember_mean):.4f}'
    )
    plt.text(
        0.02, 0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def convert_pandas_to_native(obj: Any) -> Any:
    """Recursively convert pandas/numpy objects to native Python types.
    
    Args:
        obj: Object that might be a pandas Series, numpy array, etc.
        
    Returns:
        Native Python object (dict, list, str, int, float, etc.)
    """
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_pandas_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_pandas_to_native(item) for item in obj]
    else:
        return obj


def format_example_for_jsonl(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format an example for JSONL output with better structure.
    
    Args:
        example: Raw example from the parquet file
        
    Returns:
        Formatted dictionary with clean structure
    """
    formatted = {}
    
    # Extract basic fields
    if 'prompt' in example and isinstance(example['prompt'], list):
        if len(example['prompt']) > 0:
            formatted['problem'] = example['prompt'][0].get('content', '')
    
    if 'reward_model' in example:
        reward_model = example['reward_model']
        if isinstance(reward_model, dict):
            formatted['ground_truth_solution'] = reward_model.get('ground_truth', '')
    
    # Extract extra_info fields
    extra_info = example.get('extra_info', {})
    if isinstance(extra_info, dict):
        formatted['is_member'] = extra_info.get('is_member', False)
        formatted['mia_weight'] = extra_info.get('mia_weight')
        formatted['mia_weight_tag'] = extra_info.get('mia_weight_tag', 'unknown')
        
        # Add target_gt if present
        if 'target_gt' in extra_info:
            target_gt = extra_info['target_gt']
            if isinstance(target_gt, list):
                formatted['target_gt'] = target_gt
            elif target_gt is not None and target_gt != '':
                # Convert to list if it's a single value
                formatted['target_gt'] = [target_gt]
            else:
                formatted['target_gt'] = []
        
        # Add assistant prefix if present
        if 'assistant_prefix' in extra_info:
            formatted['assistant_prefix'] = extra_info['assistant_prefix']
        
        # Add other relevant fields
        for field in ['split', 'index', 'metric', 'metric_profile']:
            if field in extra_info:
                formatted[field] = extra_info[field]
    
    # Add data source and ability
    formatted['data_source'] = example.get('data_source', '')
    formatted['ability'] = example.get('ability', '')
    
    return formatted


def extract_sample_examples(
    df: pd.DataFrame,
    n_members: int = 2,
    n_nonmembers: int = 2
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract sample examples from members and non-members.
    
    Args:
        df: DataFrame containing the parquet data
        n_members: Number of member examples to extract
        n_nonmembers: Number of non-member examples to extract
        
    Returns:
        Tuple of (member_examples, nonmember_examples)
    """
    member_examples = []
    nonmember_examples = []
    
    for idx, row in df.iterrows():
        extra_info = row.get('extra_info', {})
        if not isinstance(extra_info, dict):
            continue
        
        is_member = extra_info.get('is_member', False)
        
        if is_member and len(member_examples) < n_members:
            example_dict = row.to_dict()
            # Convert pandas objects to native Python types
            example_dict = convert_pandas_to_native(example_dict)
            formatted = format_example_for_jsonl(example_dict)
            member_examples.append(formatted)
        elif not is_member and len(nonmember_examples) < n_nonmembers:
            example_dict = row.to_dict()
            # Convert pandas objects to native Python types
            example_dict = convert_pandas_to_native(example_dict)
            formatted = format_example_for_jsonl(example_dict)
            nonmember_examples.append(formatted)
        
        if len(member_examples) >= n_members and len(nonmember_examples) >= n_nonmembers:
            break
    
    return member_examples, nonmember_examples


def save_examples_to_jsonl(
    examples: List[Dict[str, Any]],
    output_path: str,
    indent: int = 2
):
    """Save examples to JSONL file with nice formatting.
    
    Args:
        examples: List of example dictionaries
        output_path: Path to save the JSONL file
        indent: Indentation level for JSON formatting
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            # Write each example as a nicely formatted JSON object
            json_str = json.dumps(example, indent=indent, ensure_ascii=False)
            f.write(json_str + '\n\n')
    
    print(f"✓ Saved {len(examples)} examples to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and visualize MIA data from processed parquet files"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="Path to the input parquet file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and examples (default: same as parquet directory)"
    )
    parser.add_argument(
        "--n_member_samples",
        type=int,
        default=2,
        help="Number of member examples to extract (default: 2)"
    )
    parser.add_argument(
        "--n_nonmember_samples",
        type=int,
        default=2,
        help="Number of non-member examples to extract (default: 2)"
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="mia_weight_distribution.png",
        help="Name for the output plot file (default: mia_weight_distribution.png)"
    )
    
    args = parser.parse_args()
    
    # Load parquet data
    print(f"Loading parquet file: {args.parquet_path}")
    df = load_parquet_data(args.parquet_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Extract MIA weights
    print("\nExtracting MIA weights...")
    member_weights, nonmember_weights, weight_tag = extract_mia_weights(df)
    print(f"✓ Found {len(member_weights)} member weights and {len(nonmember_weights)} non-member weights")
    print(f"✓ Weight tag: {weight_tag}")
    
    if not member_weights and not nonmember_weights:
        print("⚠️  WARNING: No MIA weights found in the data!")
        return
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.parquet_path)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot distribution
    print(f"\nPlotting MIA weight distribution...")
    plot_path = os.path.join(output_dir, args.plot_name)
    plot_mia_weight_distribution(
        member_weights,
        nonmember_weights,
        weight_tag,
        plot_path
    )
    
    # Extract sample examples
    print(f"\nExtracting sample examples...")
    member_examples, nonmember_examples = extract_sample_examples(
        df,
        n_members=args.n_member_samples,
        n_nonmembers=args.n_nonmember_samples
    )
    
    # Save examples to JSONL
    if member_examples:
        members_path = os.path.join(output_dir, "sample_members.jsonl")
        save_examples_to_jsonl(member_examples, members_path)
    
    if nonmember_examples:
        nonmembers_path = os.path.join(output_dir, "sample_nonmembers.jsonl")
        save_examples_to_jsonl(nonmember_examples, nonmembers_path)
    
    print(f"\n✅ All done! Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

