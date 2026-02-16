"""
Process and clean the ai2-adapt-dev/tulu_v3.9_wildchat_100k dataset.

This script applies the same filtering criteria as process_wildchats.py but without
the exclusion filtering step (since this is the excluded dataset itself).

Usage:
    python process_tulu_wildchat.py --output-dir /path/to/output
    
    # With custom token bounds
    python process_tulu_wildchat.py --output-dir /path/to/output \
        --user-min-tokens 100 --user-max-tokens 1000 \
        --assistant-min-tokens 1000 --assistant-max-tokens 3000
    
    # Push to HuggingFace Hub
    python process_tulu_wildchat.py --output-dir /path/to/output \
        --push-to-hub --repo-id "username/dataset-name"
"""

import argparse
from process_wildchats import process_tulu_wildchat_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Process and clean the ai2-adapt-dev/tulu_v3.9_wildchat_100k dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="ai2-adapt-dev/tulu_v3.9_wildchat_100k",
        help="Path to the Tulu WildChat dataset on HuggingFace Hub (default: ai2-adapt-dev/tulu_v3.9_wildchat_100k)",
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
        default="allenai/Llama-3.1-Tulu-3-8B",
        help="Tokenizer to use (default: allenai/Llama-3.1-Tulu-3-8B)",
    )
    parser.add_argument(
        "--user-min-tokens",
        type=int,
        default=100,
        help="Minimum number of tokens for user message (default: 100)",
    )
    parser.add_argument(
        "--user-max-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens for user message (default: 1000)",
    )
    parser.add_argument(
        "--assistant-min-tokens",
        type=int,
        default=1000,
        help="Minimum number of tokens for assistant message (default: 1000)",
    )
    parser.add_argument(
        "--assistant-max-tokens",
        type=int,
        default=3000,
        help="Maximum number of tokens for assistant message (default: 3000)",
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
    parser.add_argument(
        "--filter-language",
        type=str,
        default=None,
        help="Filter by language (e.g., 'English'). If not provided, no language filtering is applied.",
    )
    
    args = parser.parse_args()
    
    process_tulu_wildchat_dataset(
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        user_min_tokens=args.user_min_tokens,
        user_max_tokens=args.user_max_tokens,
        assistant_min_tokens=args.assistant_min_tokens,
        assistant_max_tokens=args.assistant_max_tokens,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
        filter_language=args.filter_language,
    )


if __name__ == "__main__":
    main()

