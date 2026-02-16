"""
Process and clean the AI-MO/olympiads-ref dataset.

This script:
1. Loads the dataset from HuggingFace
2. Filters out image links in format ![](xxx) from solution field
3. Filters examples based on token count bounds:
   - problem: 50-1000 tokens
   - solution: 500-4000 tokens (after URL filtering)
4. Saves to local directory
5. Optionally uploads to HuggingFace Hub

Usage:
    python process_olympiads.py --output-dir /path/to/output
    
    # With custom token bounds
    python process_olympiads.py --output-dir /path/to/output \\
        --problem-min-tokens 50 --problem-max-tokens 1000 \\
        --solution-min-tokens 500 --solution-max-tokens 4000
    
    # Push to HuggingFace Hub
    python process_olympiads.py --output-dir /path/to/output \\
        --push-to-hub --repo-id "username/dataset-name"
"""

import argparse
import re
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def filter_image_links(text: str) -> str:
    """
    Filter out image links in format ![](xxx) from text.
    
    Args:
        text: The text to filter
        
    Returns:
        Text with image links removed
    """
    if not text:
        return text
    
    # Pattern to match ![](xxx) where xxx can be any characters including URLs
    # This matches:
    # - ![](https://...)
    # - ![](relative/path)
    # - ![](any text)
    pattern = r'!\[\]\([^)]+\)'
    
    # Remove all matches
    filtered_text = re.sub(pattern, '', text)
    
    # Clean up any extra whitespace that might result from removal
    # Replace multiple newlines with single newline, multiple spaces with single space
    filtered_text = re.sub(r'\n\s*\n+', '\n\n', filtered_text)
    filtered_text = re.sub(r' +', ' ', filtered_text)
    
    return filtered_text.strip()


def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in text using the tokenizer.
    
    Args:
        text: The text to tokenize
        tokenizer: The tokenizer to use
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    # Tokenize without special tokens for accurate counting
    tokens = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return len(tokens['input_ids'])


def process_olympiads_dataset(
    dataset_path: str = "AI-MO/olympiads-ref",
    dataset_split: str = "train",
    output_dir: Optional[str] = None,
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
    problem_min_tokens: int = 50,
    problem_max_tokens: int = 1000,
    solution_min_tokens: int = 500,
    solution_max_tokens: int = 4000,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
) -> Dataset:
    """
    Process and clean the olympiads dataset.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace Hub
        dataset_split: Dataset split to load
        output_dir: Local directory to save the processed dataset
        tokenizer_name: Name of the tokenizer to use for token counting
        problem_min_tokens: Minimum number of tokens for problem field
        problem_max_tokens: Maximum number of tokens for problem field
        solution_min_tokens: Minimum number of tokens for solution field
        solution_max_tokens: Maximum number of tokens for solution field
        push_to_hub: Whether to push the processed dataset to HuggingFace Hub
        repo_id: Repository ID for HuggingFace Hub (required if push_to_hub=True)
        private: Whether the HuggingFace repository should be private
        
    Returns:
        The processed dataset
    """
    print(f"Loading dataset: {dataset_path} (split: {dataset_split})")
    try:
        dataset = load_dataset(dataset_path, split=dataset_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying to load with trust_remote_code=True...")
        dataset = load_dataset(dataset_path, split=dataset_split, trust_remote_code=True)
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Process each example
    processed_examples = []
    stats = {
        'total': 0,
        'filtered_image_links': 0,
        'filtered_problem_too_short': 0,
        'filtered_problem_too_long': 0,
        'filtered_solution_too_short': 0,
        'filtered_solution_too_long': 0,
        'kept': 0,
    }
    
    print("\nProcessing examples...")
    for i, example in enumerate(dataset):
        stats['total'] += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} examples...")
        
        processed_example = dict(example)
        
        # Get problem and solution fields
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        
        # Filter image links from solution (before token counting)
        original_solution = solution
        solution = filter_image_links(solution)
        
        if original_solution != solution:
            stats['filtered_image_links'] += 1
            processed_example["solution"] = solution
        
        # Count tokens
        problem_tokens = count_tokens(problem, tokenizer)
        solution_tokens = count_tokens(solution, tokenizer)
        
        # Check bounds - BOTH problem AND solution must be within bounds
        # Skip if problem is out of bounds
        if problem_tokens < problem_min_tokens:
            stats['filtered_problem_too_short'] += 1
            continue
        
        if problem_tokens > problem_max_tokens:
            stats['filtered_problem_too_long'] += 1
            continue
        
        # Skip if solution is out of bounds
        if solution_tokens < solution_min_tokens:
            stats['filtered_solution_too_short'] += 1
            continue
        
        if solution_tokens > solution_max_tokens:
            stats['filtered_solution_too_long'] += 1
            continue
        
        # Both problem and solution are within bounds - keep this example
        # Add token counts to example for reference
        processed_example["problem_tokens"] = problem_tokens
        processed_example["solution_tokens"] = solution_tokens
        
        processed_examples.append(processed_example)
        stats['kept'] += 1
    
    print(f"\nProcessing complete:")
    print(f"  Total examples processed: {stats['total']}")
    print(f"  Examples with image links filtered: {stats['filtered_image_links']}")
    print(f"  Filtered - problem too short (<{problem_min_tokens} tokens): {stats['filtered_problem_too_short']}")
    print(f"  Filtered - problem too long (>{problem_max_tokens} tokens): {stats['filtered_problem_too_long']}")
    print(f"  Filtered - solution too short (<{solution_min_tokens} tokens): {stats['filtered_solution_too_short']}")
    print(f"  Filtered - solution too long (>{solution_max_tokens} tokens): {stats['filtered_solution_too_long']}")
    print(f"  Examples kept: {stats['kept']} ({100*stats['kept']/stats['total']:.2f}%)")
    
    # Create new dataset
    processed_dataset = Dataset.from_list(processed_examples)
    
    # Save to disk if output_dir is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving dataset to: {output_path}")
        processed_dataset.save_to_disk(str(output_path))
        print("Saved successfully.")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        
        print(f"\nPushing dataset to HuggingFace Hub: {repo_id}")
        try:
            processed_dataset.push_to_hub(repo_id, private=private)
            print("Pushed to HF Hub successfully.")
        except Exception as e:
            print(f"HF Hub push failed: {e}")
            print("Ensure HUGGINGFACE_HUB_TOKEN is set and you have write permissions.")
    
    # Print a sample
    if len(processed_dataset) > 0:
        print("\n=== Sample from processed dataset ===")
        ex = processed_dataset[0]
        for key in ex.keys():
            if key in ['problem', 'solution']:
                value = ex.get(key, 'N/A')
                print(f"{key}: {value[:200]}..." if len(str(value)) > 200 else f"{key}: {value}")
            else:
                print(f"{key}: {ex.get(key, 'N/A')}")
    
    return processed_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Process and clean the AI-MO/olympiads-ref dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="AI-MO/olympiads-ref",
        help="Path to the dataset on HuggingFace Hub (default: AI-MO/olympiads-ref)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to load (default: train)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to save the processed dataset",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer to use for token counting (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--problem-min-tokens",
        type=int,
        default=50,
        help="Minimum number of tokens for problem field (default: 50)",
    )
    parser.add_argument(
        "--problem-max-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens for problem field (default: 1000)",
    )
    parser.add_argument(
        "--solution-min-tokens",
        type=int,
        default=500,
        help="Minimum number of tokens for solution field (default: 500)",
    )
    parser.add_argument(
        "--solution-max-tokens",
        type=int,
        default=4000,
        help="Maximum number of tokens for solution field (default: 4000)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the processed dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repository ID for HuggingFace Hub (required if --push-to-hub is used)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repository private",
    )
    
    args = parser.parse_args()
    
    process_olympiads_dataset(
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        problem_min_tokens=args.problem_min_tokens,
        problem_max_tokens=args.problem_max_tokens,
        solution_min_tokens=args.solution_min_tokens,
        solution_max_tokens=args.solution_max_tokens,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
    )


if __name__ == "__main__":
    main()

