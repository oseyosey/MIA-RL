"""
Process and clean the allenai/WildChat-1M dataset.

This script:
1. Loads the WildChat-1M dataset from HuggingFace
2. Filters out conversations present in ai2-adapt-dev/tulu_v3.9_wildchat_100k
3. Filters for single-turn (turn=1) conversations
4. Optionally filters by language (if --filter-language is specified)
5. Filters based on token count bounds:
   - user message: 100-1000 tokens
   - assistant message: 1000-3000 tokens
6. Creates 'messages' field as a list structure [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
7. Saves to local directory
8. Optionally uploads to HuggingFace Hub

Usage:
    python process_wildchats.py --output-dir /path/to/output
    
    # With custom token bounds
    python process_wildchats.py --output-dir /path/to/output \
        --user-min-tokens 100 --user-max-tokens 1000 \
        --assistant-min-tokens 1000 --assistant-max-tokens 3000
    
    # Push to HuggingFace Hub
    python process_wildchats.py --output-dir /path/to/output \
        --push-to-hub --repo-id "username/dataset-name"
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def extract_conversation_content(conversation: list) -> Tuple[str, str]:
    """
    Extract user and assistant content from conversation list.
    
    Args:
        conversation: List of conversation turns with 'role' and 'content' fields
        
    Returns:
        Tuple of (user_content, assistant_content)
    """
    user_content = ""
    assistant_content = ""
    
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        
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


def format_messages_list(conversation: list) -> list:
    """
    Format conversation into a list of message dictionaries.
    
    Args:
        conversation: List of conversation turns with 'role' and 'content' fields
        
    Returns:
        List of message dictionaries in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    # Convert WildChat conversation format to standard messages list format
    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in conversation
        if turn.get("role") in ["user", "assistant"] and turn.get("content")
    ]
    
    return messages


def process_wildchats_dataset(
    dataset_path: str = "allenai/WildChat-1M",
    exclude_dataset_path: str = "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    dataset_split: str = "train",
    output_dir: Optional[str] = None,
    tokenizer_name: str = "allenai/Llama-3.1-Tulu-3-8B",
    user_min_tokens: int = 100,
    user_max_tokens: int = 1000,
    assistant_min_tokens: int = 1000,
    assistant_max_tokens: int = 3000,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
    exclude_model_substring: Optional[str] = None,
    filter_language: Optional[str] = None,
) -> Dataset:
    """
    Process and clean the WildChat dataset.
    
    Args:
        dataset_path: Path to the WildChat dataset on HuggingFace Hub
        exclude_dataset_path: Path to the exclusion subset on HuggingFace Hub
        dataset_split: Dataset split to load
        output_dir: Local directory to save the processed dataset
        tokenizer_name: Name of the tokenizer to use
        user_min_tokens: Minimum number of tokens for user message
        user_max_tokens: Maximum number of tokens for user message
        assistant_min_tokens: Minimum number of tokens for assistant message
        assistant_max_tokens: Maximum number of tokens for assistant message
        push_to_hub: Whether to push the processed dataset to HuggingFace Hub
        repo_id: Repository ID for HuggingFace Hub (required if push_to_hub=True)
        private: Whether the HuggingFace repository should be private
        exclude_model_substring: If provided, exclude examples whose `model` field
            contains this substring (e.g., "gpt-3.5")
        filter_language: If provided, only keep examples with this language (e.g., "English").
            If None, no language filtering is applied (default: None)
        
    Returns:
        The processed dataset
    """
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Load exclusion dataset to build set of conversation hashes
    print(f"\nLoading exclusion dataset: {exclude_dataset_path}")
    try:
        exclude_dataset = load_dataset(exclude_dataset_path, split=dataset_split)
    except Exception as e:
        print(f"Error loading exclusion dataset: {e}")
        print("Trying to load with trust_remote_code=True...")
        exclude_dataset = load_dataset(exclude_dataset_path, split=dataset_split, trust_remote_code=True)
    
    print(f"Exclusion dataset size: {len(exclude_dataset)}")
    
    # Build exclusion set from conversation_hash field
    print("Building exclusion set from conversation_hash field...")
    exclusion_set = set()
    for example in exclude_dataset:
        conv_hash = example.get("conversation_hash")
        if conv_hash:
            exclusion_set.add(conv_hash)
    
    print(f"Exclusion set size: {len(exclusion_set)} unique conversation hashes")
    
    # Load main dataset
    print(f"\nLoading main dataset: {dataset_path} (split: {dataset_split})")
    try:
        dataset = load_dataset(dataset_path, split=dataset_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying to load with trust_remote_code=True...")
        dataset = load_dataset(dataset_path, split=dataset_split, trust_remote_code=True)
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Process each example
    processed_examples = []
    stats = {
        'total': 0,
        'filtered_exclusion_set': 0,
        'filtered_not_single_turn': 0,
        'filtered_not_english': 0,
        'filtered_user_too_short': 0,
        'filtered_user_too_long': 0,
        'filtered_assistant_too_short': 0,
        'filtered_assistant_too_long': 0,
        'filtered_model_excluded': 0,
        'kept': 0,
    }
    
    print("\nProcessing examples...")
    for i, example in enumerate(dataset):
        stats['total'] += 1
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} examples...")
        
        # Filter 1: Check if conversation_hash is in exclusion set
        conv_hash = example.get("conversation_hash")
        if conv_hash in exclusion_set:
            stats['filtered_exclusion_set'] += 1
            continue
        
        # Filter 2: Check if single turn (turn == 1)
        turn = example.get("turn")
        if turn != 1:
            stats['filtered_not_single_turn'] += 1
            continue
        
        # Filter 3: Check language if filter is specified
        if filter_language:
            language = example.get("language", "")
            if language != filter_language:
                stats['filtered_not_english'] += 1
                continue
        
        # Filter 4: Exclude specific model substring if requested
        if exclude_model_substring:
            model_name = example.get("model", "") or ""
            if exclude_model_substring in model_name:
                stats['filtered_model_excluded'] += 1
                continue
        
        # Extract conversation content
        conversation = example.get("conversation", [])
        if not conversation:
            stats['filtered_user_too_short'] += 1
            continue
        
        user_content, assistant_content = extract_conversation_content(conversation)
        
        # Skip if either user or assistant content is empty
        if not user_content or not assistant_content:
            stats['filtered_user_too_short'] += 1
            continue
        
        # Count tokens for user and assistant messages
        user_tokens = count_tokens(user_content, tokenizer)
        assistant_tokens = count_tokens(assistant_content, tokenizer)
        
        # Filter 5: Check user token bounds
        if user_tokens < user_min_tokens:
            stats['filtered_user_too_short'] += 1
            continue
        
        if user_tokens > user_max_tokens:
            stats['filtered_user_too_long'] += 1
            continue
        
        # Filter 6: Check assistant token bounds
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
        
        # Format conversation into messages list structure
        messages = format_messages_list(conversation)
        processed_example["messages"] = messages
        
        processed_examples.append(processed_example)
        stats['kept'] += 1
    
    print(f"\nProcessing complete:")
    print(f"  Total examples processed: {stats['total']}")
    print(f"  Filtered - in exclusion set: {stats['filtered_exclusion_set']}")
    print(f"  Filtered - not single turn: {stats['filtered_not_single_turn']}")
    print(f"  Filtered - not English: {stats['filtered_not_english']}")
    if exclude_model_substring:
        print(f"  Filtered - model contained '{exclude_model_substring}': {stats['filtered_model_excluded']}")
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
        print(f"conversation_hash: {ex.get('conversation_hash', 'N/A')}")
        print(f"model: {ex.get('model', 'N/A')}")
        print(f"turn: {ex.get('turn', 'N/A')}")
        print(f"language: {ex.get('language', 'N/A')}")
        print(f"user_tokens: {ex.get('user_tokens', 'N/A')}")
        print(f"assistant_tokens: {ex.get('assistant_tokens', 'N/A')}")
        
        # Show truncated conversation
        conversation = ex.get('conversation', [])
        if conversation:
            for turn in conversation:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                truncated = content[:150] + "..." if len(content) > 150 else content
                print(f"{role}: {truncated}")
        
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
        description="Process and clean the allenai/WildChat-1M dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="allenai/WildChat-1M",
        help="Path to the WildChat dataset on HuggingFace Hub (default: allenai/WildChat-1M)",
    )
    parser.add_argument(
        "--exclude-dataset-path",
        type=str,
        default="ai2-adapt-dev/tulu_v3.9_wildchat_100k",
        help="Path to the exclusion subset on HuggingFace Hub (default: ai2-adapt-dev/tulu_v3.9_wildchat_100k)",
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
        "--exclude-model-substring",
        type=str,
        default=None,
        help="If set, exclude examples whose `model` field contains this substring (e.g., 'gpt-3.5')",
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
    
    process_wildchats_dataset(
        dataset_path=args.dataset_path,
        exclude_dataset_path=args.exclude_dataset_path,
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
        exclude_model_substring=args.exclude_model_substring,
        filter_language=args.filter_language,
    )


def process_tulu_wildchat_dataset(
    dataset_path: str = "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    dataset_split: str = "train",
    output_dir: Optional[str] = None,
    tokenizer_name: str = "allenai/Llama-3.1-Tulu-3-8B",
    user_min_tokens: int = 100,
    user_max_tokens: int = 1000,
    assistant_min_tokens: int = 1000,
    assistant_max_tokens: int = 3000,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
    filter_language: Optional[str] = None,
) -> Dataset:
    """
    Process and clean the Tulu WildChat dataset (same filtering as WildChat-1M but without exclusion).
    
    Args:
        dataset_path: Path to the Tulu WildChat dataset on HuggingFace Hub
        dataset_split: Dataset split to load
        output_dir: Local directory to save the processed dataset
        tokenizer_name: Name of the tokenizer to use
        user_min_tokens: Minimum number of tokens for user message
        user_max_tokens: Maximum number of tokens for user message
        assistant_min_tokens: Minimum number of tokens for assistant message
        assistant_max_tokens: Maximum number of tokens for assistant message
        push_to_hub: Whether to push the processed dataset to HuggingFace Hub
        repo_id: Repository ID for HuggingFace Hub (required if push_to_hub=True)
        private: Whether the HuggingFace repository should be private
        filter_language: If provided, only keep examples with this language (e.g., "English").
            If None, no language filtering is applied (default: None)
        
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
    
    # Process each example
    processed_examples = []
    stats = {
        'total': 0,
        'filtered_not_single_turn': 0,
        'filtered_not_english': 0,
        'filtered_user_too_short': 0,
        'filtered_user_too_long': 0,
        'filtered_assistant_too_short': 0,
        'filtered_assistant_too_long': 0,
        'kept': 0,
    }
    
    print("\nProcessing examples...")
    for i, example in enumerate(dataset):
        stats['total'] += 1
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} examples...")
        
        # Filter 1: Check if single turn (turn == 1)
        turn = example.get("turn")
        if turn != 1:
            stats['filtered_not_single_turn'] += 1
            continue
        
        # Filter 2: Check language if filter is specified
        if filter_language:
            language = example.get("language", "")
            if language != filter_language:
                stats['filtered_not_english'] += 1
                continue
        
        # Extract conversation content
        conversation = example.get("conversation", [])
        if not conversation:
            stats['filtered_user_too_short'] += 1
            continue
        
        user_content, assistant_content = extract_conversation_content(conversation)
        
        # Skip if either user or assistant content is empty
        if not user_content or not assistant_content:
            stats['filtered_user_too_short'] += 1
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
        
        # Format conversation into messages list structure
        messages = format_messages_list(conversation)
        processed_example["messages"] = messages
        
        processed_examples.append(processed_example)
        stats['kept'] += 1
    
    print(f"\nProcessing complete:")
    print(f"  Total examples processed: {stats['total']}")
    print(f"  Filtered - not single turn: {stats['filtered_not_single_turn']}")
    print(f"  Filtered - not English: {stats['filtered_not_english']}")
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
        print(f"conversation_hash: {ex.get('conversation_hash', 'N/A')}")
        print(f"model: {ex.get('model', 'N/A')}")
        print(f"turn: {ex.get('turn', 'N/A')}")
        print(f"language: {ex.get('language', 'N/A')}")
        print(f"user_tokens: {ex.get('user_tokens', 'N/A')}")
        print(f"assistant_tokens: {ex.get('assistant_tokens', 'N/A')}")
        
        # Show truncated conversation
        conversation = ex.get('conversation', [])
        if conversation:
            for turn in conversation:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                truncated = content[:150] + "..." if len(content) > 150 else content
                print(f"{role}: {truncated}")
        
        # Show messages field
        messages = ex.get('messages', [])
        print(f"\nmessages (list structure):")
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            truncated = content[:200] + "..." if len(content) > 200 else content
            print(f"  {role}: {truncated}")
    
    return processed_dataset


if __name__ == "__main__":
    main()

