"""
Paraphrase script for osieosie/dolma3-arxiv dataset.

This script:
- Loads the dolma3-arxiv-mia-1k dataset
- Subsamples examples either:
  a) Deterministically using --subsample-size and --subsample-seed, OR
  b) Using pre-specified indices from a JSON file via --indices-file
- Paraphrases specified fields (`text` by default) using Gemini via LiteLLM
- Copies originals to `text_original`
- Replaces paraphrased fields with paraphrased outputs
- Saves the resulting dataset to /gpfs/scrubbed/osey/Dataset_Distillation/data/<indices_file_stem>-paraphrased-v3.0
- Saves sampled indices to JSON with member/nonmember split for reproducibility
- Supports checkpoint continuation to fill missing entries

Usage:
  # Fresh run - paraphrase text field with random sampling
  python -m adra.dataset_paraphrase.paraphrase_dolma3_arxiv \
      --subsample-size 64 \
      --subsample-seed 2

  # Fresh run - use pre-specified indices from file
  python -m adra.dataset_paraphrase.paraphrase_dolma3_arxiv \
      --indices-file data/dolma3-arxiv_rl/.../dolma3-arxiv_64_indices.json

  # Paraphrase only specific field
  python -m adra.dataset_paraphrase.paraphrase_dolma3_arxiv \
      --subsample-size 64 \
      --subsample-seed 2 \
      --fields text

  # With batching for improved efficiency
  python -m adra.dataset_paraphrase.paraphrase_dolma3_arxiv \
      --indices-file /gpfs/scrubbed/osey/Dataset_Distillation/data/dolma3-arxiv_rl/dolma3-arxiv-mia-1k_64_rl_lexical_trio_v3_unique_ratio_penalty_1.50_augment_random_7_seed2_prefix_0.25_assist_0.25/dolma3-arxiv_64_indices.json \
      --enable-batching \
      --batch-size 16

  # Continue from local checkpoint (fill missing entries)
  python -m adra.dataset_paraphrase.paraphrase_dolma3_arxiv \
      --indices-file data/dolma3-arxiv_rl/.../dolma3-arxiv_64_indices.json \
      --continue-from-checkpoint

  # Reprocess extraction from existing raw responses (no new API calls)
  python -m adra.dataset_paraphrase.paraphrase_dolma3_arxiv \
      --indices-file data/dolma3-arxiv_rl/.../dolma3-arxiv_64_indices.json \
      --reprocess-extraction

  # Continue from HuggingFace Hub dataset
  python -m adra.dataset_paraphrase.paraphrase_dolma3_arxiv \
      --indices-file data/dolma3-arxiv_rl/.../dolma3-arxiv_64_indices.json \
      --continue-from-checkpoint \
      --checkpoint-path "username/dataset-repo"

Env:
  export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Ensure package import when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset, load_dataset

from adra.dataset_paraphrase import DatasetParaphraser, ParaphraseConfig

PROMPT_TEMPLATE = (
    """
    You will be given an ORIGINAL block that may contain natural-language, math, code, or other texts.\n\n
    Your job is to rephrase the natural-language content while mostly PRESERVING NON-NATURAL-LANGUAGE CONTENT.\n
    1. Please ensure the paraphrased output have similar length to the original text (±15%).\n
    2. When encountering mathematical formulas, if necessary, you can replace variable names. For example, you can replace 'x' with 'y' or 'a'.\n
    3. No extra commentary: Do not add explanations, apologies, or any other text besides the required output.\n
    4. Output directly the paraphrased output after \"rewrite_output:\".\n\n

    ORIGINAL:\n\"{input}\"\n
    """
)


def is_null_or_empty(value) -> bool:
    """Check if a value is null, None, or effectively empty."""
    if value is None:
        return True
    if isinstance(value, str) and len(value.strip()) == 0:
        return True
    return False


def find_missing_entries(dataset: Dataset, fields: List[str]) -> List[int]:
    """Find indices of entries with missing/null values in specified fields."""
    missing_indices = []
    for i in range(len(dataset)):
        example = dataset[i]
        if any(is_null_or_empty(example.get(field)) for field in fields):
            missing_indices.append(i)
    return missing_indices


def fill_missing_entries(
    dataset: Dataset, missing_indices: List[int], config: ParaphraseConfig
) -> Dataset:
    """Fill missing entries in the dataset using the paraphraser."""
    if not missing_indices:
        print("No missing entries found.")
        return dataset

    print(f"Found {len(missing_indices)} entries with missing data. Filling...")

    # Create a subset dataset with only missing entries
    missing_dataset = dataset.select(missing_indices)

    # Create a modified config that reads from original fields
    fill_config = ParaphraseConfig(
        model=config.model,
        api_key=config.api_key,
        dataset_path=config.dataset_path,
        dataset_split=config.dataset_split,
        max_examples=len(missing_indices),
        output_path=None,
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
        model_kwargs=config.model_kwargs,
        batch_size=config.batch_size,
        enable_batching=config.enable_batching,
    )

    # Add field configs that read from original fields
    for field_config in config.field_configs:
        original_field = f"{field_config.output_field}_original"
        fill_config.add_field_config(
            input_field=original_field,  # Read from original
            output_field=field_config.output_field,  # Write to processed
            prompt=field_config.prompt,
            max_tokens=field_config.max_tokens,
            temperature=field_config.temperature,
            output_label=field_config.output_label,
            strip_quotes=field_config.strip_quotes,
        )

    # Create paraphraser with the corrected config
    paraphraser = DatasetParaphraser(fill_config)
    paraphraser.dataset = missing_dataset  # Set the dataset directly

    # Process only the missing entries
    filled_dataset = paraphraser.paraphrase()

    # Update the original dataset with filled entries
    updated_examples = []
    filled_idx = 0

    for i in range(len(dataset)):
        if i in missing_indices:
            # Use the filled version
            updated_example = dict(dataset[i])
            filled_example = filled_dataset[filled_idx]

            # Update the fields that were filled
            for field_config in config.field_configs:
                field_name = field_config.output_field
                raw_field_name = f"{field_name}_raw"

                updated_example[field_name] = filled_example.get(field_name)
                updated_example[raw_field_name] = filled_example.get(raw_field_name)

            updated_examples.append(updated_example)
            filled_idx += 1
        else:
            # Keep original
            updated_examples.append(dict(dataset[i]))

    return Dataset.from_list(updated_examples)


def reprocess_extractions(dataset: Dataset, config: ParaphraseConfig) -> Dataset:
    """
    Reprocess extractions from existing raw responses without making new API calls.

    Args:
        dataset: Dataset with existing raw response fields
        config: Configuration with field configs for extraction

    Returns:
        Dataset with reprocessed extracted fields
    """
    from adra.dataset_paraphrase.paraphrase import DatasetParaphraser

    print("=== REPROCESSING EXTRACTIONS FROM RAW RESPONSES ===")

    # Create a temporary paraphraser just for the extraction logic
    temp_paraphraser = DatasetParaphraser(config)

    # Process each example
    updated_examples = []
    reprocessed_count = 0

    for i, example in enumerate(tqdm(dataset, desc="Reprocessing extractions")):
        updated_example = dict(example)

        # Reprocess each field configuration
        for field_config in config.field_configs:
            raw_field_name = f"{field_config.output_field}_raw"

            # Check if we have raw response data
            if raw_field_name in example and example[raw_field_name] is not None:
                raw_text = example[raw_field_name]

                # Re-extract using the updated logic
                old_extracted = example.get(field_config.output_field)
                new_extracted = temp_paraphraser._extract_output(
                    raw_text, field_config
                )

                # Update the field
                updated_example[field_config.output_field] = new_extracted

                # Log if extraction changed significantly
                if old_extracted != new_extracted:
                    reprocessed_count += 1
                    if i < 3:  # Show first few changes
                        logger.info(
                            "Example %d, field %s: extraction updated",
                            i,
                            field_config.output_field,
                        )
                        logger.debug(
                            "  Old: %s...",
                            old_extracted[:100] if old_extracted else "None",
                        )
                        logger.debug(
                            "  New: %s...",
                            new_extracted[:100] if new_extracted else "None",
                        )
            else:
                logger.warning(
                    "No raw response found for field %s in example %d",
                    field_config.output_field,
                    i,
                )

        updated_examples.append(updated_example)

    print(
        f"Reprocessing complete. Updated extractions for {reprocessed_count} field instances."
    )
    return Dataset.from_list(updated_examples)


def load_indices_from_file(indices_file: Path) -> tuple[List[int], Optional[int], Optional[int]]:
    """
    Load sampled indices from a JSON file.

    Args:
        indices_file: Path to JSON file containing indices

    Returns:
        Tuple of (sampled_indices, member_seed, total_size)
    """
    if not indices_file.exists():
        raise FileNotFoundError(f"Indices file not found: {indices_file}")

    with open(indices_file, "r") as f:
        indices_data = json.load(f)

    # Extract member and nonmember indices
    member_indices = indices_data.get("member_indices", [])
    nonmember_indices = indices_data.get("nonmember_indices", [])

    # Combine and sort indices
    sampled_indices = sorted(member_indices + nonmember_indices)

    # Extract seed and size if available
    member_seed = indices_data.get("member_seed")
    total_size = len(sampled_indices)

    logging.info(
        "Loaded %d indices from file: %d members, %d non-members",
        total_size,
        len(member_indices),
        len(nonmember_indices),
    )

    return sampled_indices, member_seed, total_size


def subsample_dataset(dataset: Dataset, subset_size: int, subset_seed: int):
    """
    Deterministically subsample dataset using same logic as data_mixer.py.

    Args:
        dataset: The dataset to subsample
        subset_size: Number of examples to sample
        subset_seed: Random seed for deterministic sampling

    Returns:
        Tuple of (subsampled_dataset, sampled_indices)
    """
    if subset_size > len(dataset):
        raise ValueError(
            f"subsample_size ({subset_size}) exceeds dataset size ({len(dataset)})"
        )

    rng = random.Random(subset_seed)
    sampled_indices = rng.sample(range(len(dataset)), subset_size)
    sampled_indices.sort()  # stable ordering

    logging.info(
        "Subsampled %d / %d examples with seed %d",
        subset_size,
        len(dataset),
        subset_seed,
    )

    return dataset.select(sampled_indices), sampled_indices


def subsample_dataset_from_indices(dataset: Dataset, sampled_indices: List[int]):
    """
    Subsample dataset using pre-specified indices.

    Args:
        dataset: The dataset to subsample
        sampled_indices: List of indices to select

    Returns:
        Tuple of (subsampled_dataset, sampled_indices)
    """
    # Validate indices
    max_idx = max(sampled_indices) if sampled_indices else -1
    if max_idx >= len(dataset):
        raise ValueError(f"Index {max_idx} exceeds dataset size ({len(dataset)})")

    logging.info(
        "Subsampled %d / %d examples using provided indices",
        len(sampled_indices),
        len(dataset),
    )

    return dataset.select(sampled_indices), sampled_indices


def save_sampled_indices(
    output_dir: Path,
    subsampled_dataset: Dataset,
    sampled_indices: List[int],
    subset_size: int,
    subset_seed: int,
    dataset_path: str,
    split: str,
) -> None:
    """
    Save sampled indices to JSON file with member/nonmember split for reproducibility.

    Args:
        output_dir: Output directory path
        subsampled_dataset: The subsampled dataset (to extract labels)
        sampled_indices: List of sampled indices (from original dataset)
        subset_size: Number of examples sampled
        subset_seed: Random seed used for sampling
        dataset_path: HuggingFace dataset path
        split: Dataset split used
    """
    # Split indices by label (0=non-member, 1=member)
    member_indices = []
    nonmember_indices = []

    for i, idx in enumerate(sampled_indices):
        label = subsampled_dataset[i]["label"]
        if label == 1:
            member_indices.append(idx)
        else:
            nonmember_indices.append(idx)

    # Create detailed indices info
    indices_path = output_dir / f"dolma3-arxiv_{subset_size}_indices.json"

    indices_info = {
        "member_indices": member_indices,
        "nonmember_indices": nonmember_indices,
        "member_seed": subset_seed,
        "member_size": len(member_indices),
        "nonmember_size": len(nonmember_indices),
        "dataset_info": {
            "dataset_path": dataset_path,
            "dataset_length": subset_size,
            "split": split,
            "total_dataset_size": 2000,
            "total_members": len(member_indices),
            "total_nonmembers": len(nonmember_indices),
        },
    }

    with open(indices_path, "w") as f:
        json.dump(indices_info, f, indent=2)

    logging.info("Sampled indices saved to: %s", indices_path)
    print(f"Sampled indices saved to: {indices_path}")
    print(f"  Members: {len(member_indices)}, Non-members: {len(nonmember_indices)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paraphrase osieosie/dolma3-arxiv-mia-1k dataset with required "
            "subsampling and checkpoint continuation support"
        )
    )

    # Subsampling arguments (either indices-file OR subsample-size+seed)
    subsampling_group = parser.add_mutually_exclusive_group(required=True)
    subsampling_group.add_argument(
        "--indices-file",
        type=str,
        help=(
            "Path to JSON file containing member_indices and nonmember_indices "
            "(alternative to --subsample-size/--subsample-seed)"
        ),
    )
    subsampling_group.add_argument(
        "--subsample-size",
        type=int,
        help=(
            "Number of examples to subsample from dataset "
            "(required if not using --indices-file)"
        ),
    )

    parser.add_argument(
        "--subsample-seed",
        type=int,
        help="Random seed for deterministic subsampling (required if not using --indices-file)",
    )

    # Field selection
    parser.add_argument(
        "--fields", nargs="+", default=["text"], help="Fields to paraphrase (default: text)"
    )

    # Checkpoint and processing options
    parser.add_argument(
        "--continue-from-checkpoint",
        action="store_true",
        help="Continue from existing checkpoint, filling only missing entries",
    )
    parser.add_argument(
        "--reprocess-extraction",
        action="store_true",
        help="Reprocess extraction from existing raw responses without making new API calls",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help=(
            "Path to checkpoint dataset - can be local path or HuggingFace Hub repo ID "
            "(default: auto-detect from output dir)"
        ),
    )

    # Batching options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for LLM API calls (default: 1)",
    )
    parser.add_argument(
        "--enable-batching",
        action="store_true",
        help="Enable LiteLLM batch completion for improved efficiency",
    )

    # Manual paraphrase overrides
    parser.add_argument(
        "--manual-paraphrase-file",
        type=str,
        help=(
            "JSON file with manual paraphrases for entries blocked by content filters. "
            'Format: [{"id": X, "text": "paraphrased text"}]'
        ),
    )

    args = parser.parse_args()

    # Load manual paraphrase overrides if provided
    manual_paraphrases = {}
    if args.manual_paraphrase_file:
        manual_file_path = Path(args.manual_paraphrase_file)
        if not manual_file_path.exists():
            print(f"Warning: Manual paraphrase file not found: {args.manual_paraphrase_file}")
        else:
            with open(manual_file_path, "r") as f:
                manual_list = json.load(f)
            # Create lookup dict by id
            for entry in manual_list:
                key = entry["id"]
                manual_paraphrases[key] = entry
            print(f"Loaded {len(manual_paraphrases)} manual paraphrase overrides")

    # Determine subset_size and subset_seed based on input method
    if args.indices_file:
        # Load indices from file
        indices_path = Path(args.indices_file)
        (
            sampled_indices_from_file,
            member_seed_from_file,
            total_size_from_file,
        ) = load_indices_from_file(indices_path)

        # Use values from file, or fallback to defaults
        subset_size = total_size_from_file
        subset_seed = member_seed_from_file if member_seed_from_file is not None else 0

        # Generate output directory name from indices file name if possible
        indices_file_stem = indices_path.stem  # e.g., "dolma3-arxiv_64_indices"
        output_dir = (
            "/gpfs/scrubbed/osey/Dataset_Distillation/data/"
            f"{indices_file_stem}-paraphrased-v3.0"
        )
        hf_repo_id = f"{indices_file_stem}-Paraphrased-Gemini-2.5-Flash-v3.0"
    else:
        # Use provided subsample-size and seed
        if args.subsample_size is None or args.subsample_seed is None:
            raise ValueError(
                "Both --subsample-size and --subsample-seed are required when not using --indices-file"
            )
        subset_size = args.subsample_size
        subset_seed = args.subsample_seed
        output_dir = (
            "/gpfs/scrubbed/osey/Dataset_Distillation/data/"
            f"dolma3-arxiv-mia-1k-1024-{subset_size}-seed{subset_seed}-paraphrased-v3.0"
        )
        hf_repo_id = (
            f"dolma3-arxiv-mia-1k-1024-{subset_size}-seed{subset_seed}-Paraphrased-Gemini-2.5-Flash-v3.0"
        )
        sampled_indices_from_file = None

    # Configuration for dolma3-arxiv-mia-1k dataset
    dataset_path = "osieosie/dolma3-arxiv-mia-1k-1024"
    dataset_split = "train"

    api_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is required.")

    config = ParaphraseConfig(
        model="gemini/gemini-2.5-flash",
        api_key=api_key,
        dataset_path=dataset_path,
        dataset_split=dataset_split,
        # Process all examples (default None)
        max_examples=None,
        # We'll save manually after adding original fields
        output_path=None,
        # Batching configuration
        batch_size=args.batch_size,
        enable_batching=args.enable_batching,
        # Manual paraphrase overrides
        manual_paraphrases=manual_paraphrases,
    )

    # Add field configs based on user selection
    for field in args.fields:
        config.add_field_config(
            input_field=field,
            output_field=field,
            prompt=PROMPT_TEMPLATE,
            max_tokens=32768,
            temperature=0.7,
            output_label="rewrite_output",
            strip_quotes=True,
        )

    # Track sampled indices and dataset for saving
    sampled_indices = None
    subsampled_dataset = None

    if args.reprocess_extraction:
        # Reprocess extraction mode - no new API calls
        print("=== REPROCESS EXTRACTION MODE ===")

        # Load existing dataset
        checkpoint_path = args.checkpoint_path or output_dir

        # Determine if checkpoint_path is a HuggingFace Hub repo or local path
        is_hf_repo = not (
            checkpoint_path.startswith("/")
            or checkpoint_path.startswith("./")
            or Path(checkpoint_path).exists()
        )

        if is_hf_repo:
            print(f"Loading dataset from HuggingFace Hub: {checkpoint_path}")
            try:
                processed = load_dataset(checkpoint_path, split="train")
            except Exception as e:
                print(f"Failed to load from HuggingFace Hub: {e}")
                print("Make sure the repository exists and is accessible.")
                return
        else:
            # Local path
            if not Path(checkpoint_path).exists():
                print(f"Local dataset path {checkpoint_path} does not exist. Cannot reprocess.")
                return

            print(f"Loading dataset from local path: {checkpoint_path}")
            try:
                processed = Dataset.load_from_disk(checkpoint_path)
            except ValueError as e:
                if "Keys mismatch" in str(e):
                    print(f"Dataset schema mismatch detected: {e}")
                    print("The dataset appears to be corrupted or from a different format.")
                    return
                else:
                    raise e

        # Check if dataset has raw response fields
        expected_raw_fields = [
            f"{fc.output_field}_raw" for fc in config.field_configs
        ]
        missing_raw_fields = [
            field for field in expected_raw_fields if field not in processed.column_names
        ]

        if missing_raw_fields:
            print(f"Error: Dataset is missing raw response fields: {missing_raw_fields}")
            print("Cannot reprocess extraction without raw responses.")
            print("This dataset may have been created without raw response recording.")
            return

        print(f"Found raw response fields: {expected_raw_fields}")
        print(f"Reprocessing extractions for {len(processed)} examples...")

        # Reprocess extractions
        processed = reprocess_extractions(processed, config)

    elif args.continue_from_checkpoint:
        # Checkpoint continuation mode
        print("=== CHECKPOINT CONTINUATION MODE ===")

        # Try to load existing processed dataset
        checkpoint_path = args.checkpoint_path or output_dir

        # Determine if checkpoint_path is a HuggingFace Hub repo or local path
        is_hf_repo = not (
            checkpoint_path.startswith("/")
            or checkpoint_path.startswith("./")
            or Path(checkpoint_path).exists()
        )

        if is_hf_repo:
            print(f"Loading checkpoint from HuggingFace Hub: {checkpoint_path}")
            try:
                processed = load_dataset(checkpoint_path, split="train")
            except Exception as e:
                print(f"Failed to load from HuggingFace Hub: {e}")
                print("Make sure the repository exists and is accessible.")
                return
        else:
            # Local path
            if not Path(checkpoint_path).exists():
                print(f"Local checkpoint path {checkpoint_path} does not exist. Cannot continue.")
                return

            print(f"Loading checkpoint from local path: {checkpoint_path}")
            try:
                processed = Dataset.load_from_disk(checkpoint_path)
            except ValueError as e:
                if "Keys mismatch" in str(e):
                    print(f"Dataset schema mismatch detected: {e}")
                    print("This might be due to cached dataset conflicts.")
                    print(
                        "Try deleting the checkpoint directory and running fresh, or check if the dataset is corrupted."
                    )
                    return
                else:
                    raise e

        # Find missing entries based on fields being processed
        missing_indices = find_missing_entries(processed, args.fields)

        if not missing_indices:
            print("No missing entries found in checkpoint. Dataset is complete!")
            return

        print(f"Found {len(missing_indices)} entries with missing data:")
        # Show some examples of missing entries
        for i, idx in enumerate(missing_indices[:5]):
            example = processed[idx]
            missing_fields = [
                field for field in args.fields if is_null_or_empty(example.get(field))
            ]
            print(f"  Index {idx}: missing fields: {missing_fields}")

        if len(missing_indices) > 5:
            print(f"  ... and {len(missing_indices) - 5} more")

        # Fill missing entries
        processed = fill_missing_entries(processed, missing_indices, config)

    else:
        # Fresh run mode
        print("=== FRESH RUN MODE ===")

        # Load full dataset first
        full_dataset = load_dataset(dataset_path, split=dataset_split)
        print(f"Loaded {len(full_dataset)} total examples from {dataset_path}")

        # Subsample dataset BEFORE paraphrasing
        if args.indices_file:
            # Use indices from file
            print(f"Using indices from file: {args.indices_file}")
            originals, sampled_indices = subsample_dataset_from_indices(
                full_dataset, sampled_indices_from_file
            )
            print(f"Subsampled {len(originals)} examples using provided indices")
        else:
            # Use random sampling
            originals, sampled_indices = subsample_dataset(
                full_dataset, subset_size, subset_seed
            )
            print(f"Subsampled {len(originals)} examples with seed {subset_seed}")

        # Store subsampled dataset for indices saving
        subsampled_dataset = originals

        # Run paraphrasing on subsampled dataset
        paraphraser = DatasetParaphraser(config)
        paraphraser.dataset = originals  # Set the subsampled dataset
        processed = paraphraser.paraphrase()  # no save yet

        # Add original fields for the fields being processed
        for field in args.fields:
            original_field = f"{field}_original"
            processed = processed.add_column(original_field, originals[field])  # type: ignore

        # Add other dolma3-arxiv fields from original dataset if they don't exist
        for field in ["meta", "id", "label"]:
            if field in originals.column_names and field not in processed.column_names:
                processed = processed.add_column(field, originals[field])  # type: ignore

    # Save final dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}")
    processed.save_to_disk(output_dir)
    print("Saved successfully.")

    # Save sampled indices if this was a fresh run
    if sampled_indices is not None and subsampled_dataset is not None:
        save_sampled_indices(
            Path(output_dir),
            subsampled_dataset,
            sampled_indices,
            subset_size,
            subset_seed,
            dataset_path,
            dataset_split,
        )

    # Push to Hugging Face Hub
    try:
        print(f"\nPushing dataset to Hugging Face Hub: {hf_repo_id}")
        processed.push_to_hub(hf_repo_id, private=False)
        print("Pushed to HF Hub successfully.")
    except Exception as e:
        print(f"HF Hub push failed: {e}")
        print("Ensure HUGGINGFACE_HUB_TOKEN is set and you have write permissions.")

    # Print a small sample
    if len(processed) > 0:
        ex = processed[0]
        print("\nSample:")

        # Print metadata fields
        for field in ["id", "label", "meta"]:
            if field in ex:
                print(f"{field}: {ex.get(field, 'N/A')}")

        # Print fields that were paraphrased
        for field in args.fields:
            original_field = f"{field}_original"
            raw_field = f"{field}_raw"

            if original_field in ex:
                print(f"{original_field}: {ex.get(original_field, 'N/A')[:160]}...")
            print(f"{field}: {ex.get(field, 'N/A')[:160]}...")
            if raw_field in ex:
                print(f"{raw_field}: {ex.get(raw_field, 'N/A')[:250]}...")

        # Check for any null extractions
        null_count = {}
        for field in args.fields:
            null_count[field] = sum(
                1 for i in range(len(processed)) if not processed[i].get(field)
            )

        if any(count > 0 for count in null_count.values()):
            print("\nWarning: Found null extractions:")
            for field, count in null_count.items():
                if count > 0:
                    print(f"  {field}: {count} null entries")
            print("Check the raw fields to debug extraction issues.")


if __name__ == "__main__":
    main()

