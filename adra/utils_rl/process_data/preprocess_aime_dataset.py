"""
Preprocess AIME 2021-2025 dataset to extract only the first solution.

The original dataset contains multiple solutions from different authors,
marked with "~author_name". This script extracts only the first solution
(before the first "~" marker) to create a cleaner dataset for training.

Usage:
    python -m adra.utils_rl.preprocess_aime_dataset
    
    # With custom output path
    python -m adra.utils_rl.preprocess_aime_dataset --output-dir /path/to/output
    
    # Push to HuggingFace Hub
    python -m adra.utils_rl.preprocess_aime_dataset --push-to-hub --repo-id "username/dataset-name"
"""

import argparse
import re
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset


def extract_first_solution(solution: str) -> str:
    """
    Extract the first solution from a solution string that may contain multiple solutions.
    
    Solutions are typically separated by author attributions that start with "~"
    (e.g., "~chem1kall", "~Steven Chen", etc.).
    
    Args:
        solution: The solution text potentially containing multiple solutions
        
    Returns:
        The first solution only, with trailing whitespace removed
    """
    if not solution:
        return solution
    
    # Find the first occurrence of a line starting with "~" (author attribution)
    # Pattern: newline followed by "~" at the start of a line
    pattern = r'\n~[^\n]+'
    match = re.search(pattern, solution)
    
    if match:
        # Extract everything before the first author attribution
        first_solution = solution[:match.start()].strip()
        return first_solution
    
    # If no author attribution found, return the original solution
    return solution.strip()


def preprocess_aime_dataset(
    dataset_path: str = "hamishivi/aime-2021-2025",
    dataset_split: str = "train",
    output_dir: Optional[str] = None,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
) -> Dataset:
    """
    Preprocess the AIME dataset to extract only the first solution.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace Hub
        dataset_split: Dataset split to load
        output_dir: Local directory to save the processed dataset
        push_to_hub: Whether to push the processed dataset to HuggingFace Hub
        repo_id: Repository ID for HuggingFace Hub (required if push_to_hub=True)
        
    Returns:
        The preprocessed dataset
    """
    print(f"Loading dataset: {dataset_path} (split: {dataset_split})")
    dataset = load_dataset(dataset_path, split=dataset_split)
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Process each example
    processed_examples = []
    solutions_modified = 0
    
    for i, example in enumerate(dataset):
        processed_example = dict(example)
        
        # Extract first solution
        original_solution = example.get("solution", "")
        first_solution = extract_first_solution(original_solution)
        
        # Track if solution was modified
        if original_solution != first_solution:
            solutions_modified += 1
            if i < 3:  # Show first few examples
                print(f"\n=== Example {i} ===")
                print(f"Original length: {len(original_solution)} chars")
                print(f"Processed length: {len(first_solution)} chars")
                print(f"First 200 chars of original: {original_solution[:200]}...")
                print(f"First 200 chars of processed: {first_solution[:200]}...")
        
        processed_example["solution"] = first_solution
        processed_examples.append(processed_example)
    
    print(f"\nProcessing complete:")
    print(f"  Total examples: {len(processed_examples)}")
    print(f"  Solutions modified: {solutions_modified}")
    print(f"  Solutions unchanged: {len(processed_examples) - solutions_modified}")
    
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
            processed_dataset.push_to_hub(repo_id, private=False)
            print("Pushed to HF Hub successfully.")
        except Exception as e:
            print(f"HF Hub push failed: {e}")
            print("Ensure HUGGINGFACE_HUB_TOKEN is set and you have write permissions.")
    
    # Print a sample
    if len(processed_dataset) > 0:
        print("\n=== Sample from processed dataset ===")
        ex = processed_dataset[0]
        print(f"ID: {ex.get('id', 'N/A')}")
        print(f"Problem: {ex.get('problem', 'N/A')[:150]}...")
        print(f"Solution: {ex.get('solution', 'N/A')[:300]}...")
        print(f"Answer: {ex.get('answer', 'N/A')}")
        print(f"URL: {ex.get('url', 'N/A')}")
    
    return processed_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess AIME dataset to extract only the first solution"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="hamishivi/aime-2021-2025",
        help="Path to the dataset on HuggingFace Hub (default: hamishivi/aime-2021-2025)",
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
        default="/gpfs/scrubbed/osey/Dataset_Distillation/data/AIME-2021-2025-cleaned",
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
        default="AIME-2021-2025-Cleaned",
        help="Repository ID for HuggingFace Hub (default: AIME-2021-2025-Cleaned)",
    )
    args = parser.parse_args()
    
    preprocess_aime_dataset(
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
