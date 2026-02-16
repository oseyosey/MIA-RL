"""
Paraphrase script for osieosie/AIME-2021-2025-Cleaned dataset.

This script:
- Loads the AIME 2021-2025 dataset
- Paraphrases both `problem` and `solution` fields using Gemini via LiteLLM
- Copies originals to `problem_original` and `solution_original`
- Replaces `problem` and `solution` with paraphrased outputs
- Saves the resulting dataset to /gpfs/scrubbed/osey/Dataset_Distillation/data/AIME-2021-2025-paraphrased
- Supports checkpoint continuation to fill missing entries

Usage:
  # Fresh run
  python -m adra.dataset_paraphrase.paraphrase_aime_test
  
  # With batching for improved efficiency
  python -m adra.dataset_paraphrase.paraphrase_aime_test --enable-batching --batch-size 5
  
  # Continue from local checkpoint (fill missing entries)
  python -m adra.dataset_paraphrase.paraphrase_aime_test --continue-from-checkpoint
  
  # Reprocess extraction from existing raw responses (no new API calls)
  python -m adra.dataset_paraphrase.paraphrase_aime_test --reprocess-extraction
  
  # Continue from HuggingFace Hub dataset
  python -m adra.dataset_paraphrase.paraphrase_aime_test --continue-from-checkpoint --checkpoint-path "username/dataset-repo"

Env:
  export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Ensure package import when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset, Dataset

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


def fill_missing_entries(dataset: Dataset, missing_indices: List[int], config: ParaphraseConfig) -> Dataset:
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
    )
    
    # Add field configs that read from original fields
    for field_config in config.field_configs:
        if field_config.output_field == "problem":
            fill_config.add_field_config(
                input_field="problem_original",  # Read from original
                output_field="problem",          # Write to processed
                prompt=field_config.prompt,
                max_tokens=field_config.max_tokens,
                temperature=field_config.temperature,
                output_label=field_config.output_label,
                strip_quotes=field_config.strip_quotes,
            )
        elif field_config.output_field == "solution":
            fill_config.add_field_config(
                input_field="solution_original",  # Read from original
                output_field="solution",           # Write to processed
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
                new_extracted = temp_paraphraser._extract_output(raw_text, field_config)
                
                # Update the field
                updated_example[field_config.output_field] = new_extracted
                
                # Log if extraction changed significantly
                if old_extracted != new_extracted:
                    reprocessed_count += 1
                    if i < 3:  # Show first few changes
                        logger.info(f"Example {i}, field {field_config.output_field}: extraction updated")
                        logger.debug(f"  Old: {old_extracted[:100] if old_extracted else 'None'}...")
                        logger.debug(f"  New: {new_extracted[:100] if new_extracted else 'None'}...")
            else:
                logger.warning(f"No raw response found for field {field_config.output_field} in example {i}")
        
        updated_examples.append(updated_example)
    
    print(f"Reprocessing complete. Updated extractions for {reprocessed_count} field instances.")
    return Dataset.from_list(updated_examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase AIME 2021-2025 dataset with checkpoint continuation support")
    parser.add_argument("--continue-from-checkpoint", action="store_true", 
                        help="Continue from existing checkpoint, filling only missing entries")
    parser.add_argument("--reprocess-extraction", action="store_true",
                        help="Reprocess extraction from existing raw responses without making new API calls")
    parser.add_argument("--checkpoint-path", type=str,
                        help="Path to checkpoint dataset - can be local path or HuggingFace Hub repo ID (default: auto-detect from output dir)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for LLM API calls (default: 1)")
    parser.add_argument("--enable-batching", action="store_true",
                        help="Enable LiteLLM batch completion for improved efficiency")
    args = parser.parse_args()
    
    # Configuration for AIME 2021-2025 dataset
    dataset_path = "osieosie/AIME-2021-2025-Cleaned"
    dataset_split = "train"
    output_dir = "/gpfs/scrubbed/osey/Dataset_Distillation/data/AIME-2021-2025-paraphrased-v3.0"
    hf_repo_id = "AIME-2021-2025-Paraphrased-Gemini-2.5-Flash-v3.0"

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
    )

    # Paraphrase `problem` (overwrite original field)
    config.add_field_config(
        input_field="problem",
        output_field="problem",
        prompt=PROMPT_TEMPLATE,
        max_tokens=16384,
        temperature=0.7,
        output_label="rewrite_output",
        strip_quotes=True,
    )

    # Paraphrase `solution` (overwrite original field)
    config.add_field_config(
        input_field="solution",
        output_field="solution",
        prompt=PROMPT_TEMPLATE,
        max_tokens=16384, # this is usually to avoid null entries.
        temperature=0.7,
        output_label="rewrite_output",
        strip_quotes=True,
    )

    if args.reprocess_extraction:
        # Reprocess extraction mode - no new API calls
        print("=== REPROCESS EXTRACTION MODE ===")
        
        # Load existing dataset
        checkpoint_path = args.checkpoint_path or output_dir
        
        # Determine if checkpoint_path is a HuggingFace Hub repo or local path
        is_hf_repo = not (checkpoint_path.startswith('/') or checkpoint_path.startswith('./') or Path(checkpoint_path).exists())
        
        if is_hf_repo:
            print(f"Loading dataset from HuggingFace Hub: {checkpoint_path}")
            try:
                processed = load_dataset(checkpoint_path, split='train')
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
        expected_raw_fields = [f"{fc.output_field}_raw" for fc in config.field_configs]
        missing_raw_fields = [field for field in expected_raw_fields if field not in processed.column_names]
        
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
        is_hf_repo = not (checkpoint_path.startswith('/') or checkpoint_path.startswith('./') or Path(checkpoint_path).exists())
        
        if is_hf_repo:
            print(f"Loading checkpoint from HuggingFace Hub: {checkpoint_path}")
            try:
                processed = load_dataset(checkpoint_path, split='train')  # Most HF datasets use 'train' split
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
                    print("Try deleting the checkpoint directory and running fresh, or check if the dataset is corrupted.")
                    return
                else:
                    raise e
        
        # Find missing entries
        missing_indices = find_missing_entries(processed, ["problem", "solution"])
        
        if not missing_indices:
            print("No missing entries found in checkpoint. Dataset is complete!")
            return
        
        print(f"Found {len(missing_indices)} entries with missing data:")
        # Show some examples of missing entries
        for i, idx in enumerate(missing_indices[:5]):
            example = processed[idx]
            problem_missing = is_null_or_empty(example.get("problem"))
            solution_missing = is_null_or_empty(example.get("solution"))
            print(f"  Index {idx}: problem={'MISSING' if problem_missing else 'OK'}, solution={'MISSING' if solution_missing else 'OK'}")
        
        if len(missing_indices) > 5:
            print(f"  ... and {len(missing_indices) - 5} more")
        
        # Fill missing entries
        processed = fill_missing_entries(processed, missing_indices, config)
        
    else:
        # Fresh run mode
        print("=== FRESH RUN MODE ===")
        
        # Load originals to preserve before paraphrasing
        originals = load_dataset(dataset_path, split=dataset_split)
        
        print(f"Loaded {len(originals)} examples from {dataset_path}")

        # Run paraphrasing
        paraphraser = DatasetParaphraser(config)
        processed = paraphraser.run()  # no save yet

        # Add original fields
        # Ensure the originals length matches the processed dataset length
        if len(originals) != len(processed):
            originals = originals.select(range(len(processed)))
        processed = processed.add_column("problem_original", originals["problem"])  # type: ignore
        processed = processed.add_column("solution_original", originals["solution"])  # type: ignore
        
        # Add other AIME-specific fields from original dataset if they don't exist
        for field in ["id", "answer", "url"]:
            if field in originals.column_names and field not in processed.column_names:
                processed = processed.add_column(field, originals[field])  # type: ignore

    # Save final dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}")
    processed.save_to_disk(output_dir)
    print("Saved successfully.")

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
        print(f"id: {ex.get('id', 'N/A')}")
        print(f"problem_original: {ex.get('problem_original', 'N/A')[:160]}...")
        print(f"problem: {ex.get('problem', 'N/A')[:160]}...")
        print(f"solution_original: {ex.get('solution_original', 'N/A')[:160]}...")
        print(f"solution: {ex.get('solution', 'N/A')[:160]}...")
        print(f"answer: {ex.get('answer', 'N/A')}")
        print(f"url: {ex.get('url', 'N/A')}")
        print(f"problem_raw: {ex.get('problem_raw', 'N/A')[:250]}...")
        print(f"solution_raw: {ex.get('solution_raw', 'N/A')[:250]}...")
        
        # Show reasoning content if available
        if 'problem_reasoning' in ex and ex.get('problem_reasoning'):
            print(f"problem_reasoning: {ex.get('problem_reasoning', 'N/A')[:200]}...")
        if 'solution_reasoning' in ex and ex.get('solution_reasoning'):
            print(f"solution_reasoning: {ex.get('solution_reasoning', 'N/A')[:200]}...")
        
        # Check for any null extractions
        null_problems = sum(1 for i in range(len(processed)) if not processed[i].get('problem'))
        null_solutions = sum(1 for i in range(len(processed)) if not processed[i].get('solution'))
        
        if null_problems > 0 or null_solutions > 0:
            print(f"\nWarning: Found {null_problems} null problems and {null_solutions} null solutions")
            print("Check the raw fields to debug extraction issues.")


if __name__ == "__main__":
    main()


