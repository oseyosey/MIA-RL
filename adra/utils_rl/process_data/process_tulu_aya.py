"""
Process and clean the Tulu Aya dataset from allenai/tulu-3-sft-mixture.

This script filters for the ai2-adapt-dev/tulu_v3.9_aya_100k subset and applies
filtering criteria similar to process_wildchats.py:
1. Filters for source == "ai2-adapt-dev/tulu_v3.9_aya_100k"
2. Filters for single-turn conversations (len(messages) == 2)
3. Filters examples based on token count bounds:
   - user message: 50-500 tokens (default)
   - assistant message: 50-500 tokens (default)
4. Keeps existing 'messages' field structure
5. Saves to local directory
6. Optionally uploads to HuggingFace Hub

Usage:
    python process_tulu_aya.py --output-dir /path/to/output
    
    # With custom token bounds
    python process_tulu_aya.py --output-dir /path/to/output \
        --user-min-tokens 50 --user-max-tokens 500 \
        --assistant-min-tokens 50 --assistant-max-tokens 500
    
    # Push to HuggingFace Hub
    python process_tulu_aya.py --output-dir /path/to/output \
        --push-to-hub --repo-id "username/dataset-name"
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def extract_messages_content(messages: list) -> Tuple[str, str]:
    """
    Extract user and assistant content from messages list.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' fields
        
    Returns:
        Tuple of (user_content, assistant_content)
    """
    user_content = ""
    assistant_content = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            user_content = content
        elif role == "assistant":
            assistant_content = content
    
    return user_content, assistant_content


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


def process_tulu_aya_dataset(
    dataset_path: str = "allenai/tulu-3-sft-mixture",
    dataset_split: str = "train",
    source_filter: str = "ai2-adapt-dev/tulu_v3.9_aya_100k",
    output_dir: Optional[str] = None,
    tokenizer_name: str = "allenai/Llama-3.1-Tulu-3-8B",
    user_min_tokens: int = 50,
    user_max_tokens: int = 500,
    assistant_min_tokens: int = 50,
    assistant_max_tokens: int = 500,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
) -> Dataset:
    """
    Process and clean the Tulu Aya dataset from tulu-3-sft-mixture.
    
    Args:
        dataset_path: Path to the tulu-3-sft-mixture dataset on HuggingFace Hub
        dataset_split: Dataset split to load
        source_filter: Source value to filter for (default: ai2-adapt-dev/tulu_v3.9_aya_100k)
        output_dir: Local directory to save the processed dataset
        tokenizer_name: Name of the tokenizer to use
        user_min_tokens: Minimum number of tokens for user message
        user_max_tokens: Maximum number of tokens for user message
        assistant_min_tokens: Minimum number of tokens for assistant message
        assistant_max_tokens: Maximum number of tokens for assistant message
        push_to_hub: Whether to push the processed dataset to HuggingFace Hub
        repo_id: Repository ID for HuggingFace Hub (required if push_to_hub=True)
        private: Whether the HuggingFace repository should be private
        
    Returns:
        The processed dataset
    """
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Load main dataset
    print(f"\nLoading dataset: {dataset_path} (split: {dataset_split})")
    try:
        dataset = load_dataset(dataset_path, split=dataset_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying to load with trust_remote_code=True...")
        dataset = load_dataset(dataset_path, split=dataset_split, trust_remote_code=True)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtering for source: {source_filter}")
    
    # Process each example
    processed_examples = []
    stats = {
        'total': 0,
        'filtered_wrong_source': 0,
        'filtered_not_single_turn': 0,
        'filtered_empty_content': 0,
        'filtered_user_too_short': 0,
        'filtered_user_too_long': 0,
        'filtered_assistant_too_short': 0,
        'filtered_assistant_too_long': 0,
        'kept': 0,
    }
    
    print("\nProcessing examples...")
    for i, example in enumerate(dataset):
        stats['total'] += 1
        
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} examples...")
        
        # Filter 1: Check if source matches
        source = example.get("source", "")
        if source != source_filter:
            stats['filtered_wrong_source'] += 1
            continue
        
        # Filter 2: Check if single turn (len(messages) == 2)
        messages = example.get("messages", [])
        if len(messages) != 2:
            stats['filtered_not_single_turn'] += 1
            continue
        
        # Extract user and assistant content from messages
        user_content, assistant_content = extract_messages_content(messages)
        
        # Skip if either user or assistant content is empty
        if not user_content or not assistant_content:
            stats['filtered_empty_content'] += 1
            continue
        
        # Count tokens for user and assistant messages
        user_tokens = count_tokens(user_content, tokenizer)
        assistant_tokens = count_tokens(assistant_content, tokenizer)
        
        # Filter 3: Check user token bounds
        if user_tokens < user_min_tokens:
            stats['filtered_user_too_short'] += 1
            continue
        
        if user_tokens > user_max_tokens:
            stats['filtered_user_too_long'] += 1
            continue
        
        # Filter 4: Check assistant token bounds
        if assistant_tokens < assistant_min_tokens:
            stats['filtered_assistant_too_short'] += 1
            continue
        
        if assistant_tokens > assistant_max_tokens:
            stats['filtered_assistant_too_long'] += 1
            continue
        
        # All filters passed - process this example
        processed_example = dict(example)
        
        # Add token counts
        processed_example["user_tokens"] = user_tokens
        processed_example["assistant_tokens"] = assistant_tokens
        
        # Messages field already exists in correct format, no need to modify
        
        processed_examples.append(processed_example)
        stats['kept'] += 1
    
    print(f"\nProcessing complete:")
    print(f"  Total examples processed: {stats['total']}")
    print(f"  Filtered - wrong source: {stats['filtered_wrong_source']}")
    print(f"  Filtered - not single turn: {stats['filtered_not_single_turn']}")
    print(f"  Filtered - empty content: {stats['filtered_empty_content']}")
    print(f"  Filtered - user too short (<{user_min_tokens} tokens): {stats['filtered_user_too_short']}")
    print(f"  Filtered - user too long (>{user_max_tokens} tokens): {stats['filtered_user_too_long']}")
    print(f"  Filtered - assistant too short (<{assistant_min_tokens} tokens): {stats['filtered_assistant_too_short']}")
    print(f"  Filtered - assistant too long (>{assistant_max_tokens} tokens): {stats['filtered_assistant_too_long']}")
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
        print(f"id: {ex.get('id', 'N/A')}")
        print(f"source: {ex.get('source', 'N/A')}")
        print(f"user_tokens: {ex.get('user_tokens', 'N/A')}")
        print(f"assistant_tokens: {ex.get('assistant_tokens', 'N/A')}")
        
        # Show messages field
        messages = ex.get('messages', [])
        print(f"\nmessages (list structure):")
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            truncated = content[:200] + "..." if len(content) > 200 else content
            print(f"  {role}: {truncated}")
    
    return processed_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Process and clean the Tulu Aya dataset from allenai/tulu-3-sft-mixture"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="allenai/tulu-3-sft-mixture",
        help="Path to the tulu-3-sft-mixture dataset on HuggingFace Hub (default: allenai/tulu-3-sft-mixture)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to load (default: train)",
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default="ai2-adapt-dev/tulu_v3.9_aya_100k",
        help="Source value to filter for (default: ai2-adapt-dev/tulu_v3.9_aya_100k)",
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
        default=50,
        help="Minimum number of tokens for user message (default: 50)",
    )
    parser.add_argument(
        "--user-max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens for user message (default: 500)",
    )
    parser.add_argument(
        "--assistant-min-tokens",
        type=int,
        default=50,
        help="Minimum number of tokens for assistant message (default: 50)",
    )
    parser.add_argument(
        "--assistant-max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens for assistant message (default: 500)",
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
    
    process_tulu_aya_dataset(
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        source_filter=args.source_filter,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        user_min_tokens=args.user_min_tokens,
        user_max_tokens=args.user_max_tokens,
        assistant_min_tokens=args.assistant_min_tokens,
        assistant_max_tokens=args.assistant_max_tokens,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
    )


if __name__ == "__main__":
    main()

