#!/usr/bin/env python3
"""Data mixing utilities for combining contamination and general datasets.

This module provides functionality to:
1. Process contamination datasets (e.g. AIME) with subset control
2. Process general mixing datasets (e.g., tulu-3-sft-mixture) 
3. Combine them with specified contamination percentage
4. Upload mixed datasets to HuggingFace Hub

Example usage:
    
    python adra/scripts/produce_checkpoints/data_mixer.py \
        --contamination_dataset aime \
        --contamination_dataset_name osieosie/AIME-2021-2025-Paraphrased-Gemini-2.5-Flash-v3.0 \
        --contamination_size 64 \
        --contamination_seed 1 \
        --contamination_percentage 1.0 \
        --general_dataset tulu3_sft \
        --general_seed 1 \
        --upload_to_hf
    
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import HfApi

# -----------------------------------------------------------------------------
# Ensure repository root is on PYTHONPATH
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Year filtering utilities
# -----------------------------------------------------------------------------

def extract_year_from_url(url: str) -> int:
    """Extract year from AIME problem URL.
    
    Example: https://artofproblemsolving.com/wiki/index.php/2025_AIME_I_Problems/Problem_11
    Returns: 2025
    
    Args:
        url: URL string from the dataset
        
    Returns:
        Year as integer, or None if year cannot be extracted
    """
    import re
    match = re.search(r'/(\d{4})_AIME', url)
    if match:
        return int(match.group(1))
    return None


def filter_dataset_by_years(dataset, years: List[int], verbose: bool = False):
    """Filter dataset to only include examples from specified years.
    
    Args:
        dataset: The dataset to filter (must have 'url' field)
        years: List of years to include
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (filtered_dataset, filtered_indices)
    """
    filtered_indices = []
    for idx in range(len(dataset)):
        if "url" in dataset[idx]:
            year = extract_year_from_url(dataset[idx]["url"])
            if year in years:
                filtered_indices.append(idx)
    
    if verbose:
        logging.info("Filtered %d examples from years %s out of %d total", 
                    len(filtered_indices), years, len(dataset))
    
    if len(filtered_indices) == 0:
        raise ValueError(f"No examples found for years {years}. Check that the dataset has 'url' field with year information.")
    
    return dataset.select(filtered_indices), filtered_indices


# -----------------------------------------------------------------------------
# Contamination dataset processors
# -----------------------------------------------------------------------------

def process_aime_contamination(
    subset_size: int = 500, 
    subset_seed: int = 42,
    tokenizer_name: str = "allenai/tulu-2-7b",
    dataset_name: str = "osieosie/AIME-2021-2025-Paraphrased-Gemini-2.5-Flash-v3.0",
    split: str = "train",
    filter_by_year: bool = False,
    year_filter: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Process AIME contamination dataset.
    
    Args:
        subset_size: Number of examples to sample from dataset
        subset_seed: Random seed for deterministic sampling
        tokenizer_name: Tokenizer to use for chat templating
        dataset_name: HuggingFace dataset name to load
        split: Dataset split to use ("train" or "test")
        filter_by_year: Whether to filter by year
        year_filter: List of years to include (only used if filter_by_year=True)
        
    Returns:
        List of processed examples with 'text' and metadata fields
    """
    logging.info("Loading AIME contamination dataset: %s (split: %s)", dataset_name, split)
    
    ds = load_dataset(dataset_name, split=split)
    full_dataset_size = len(ds)
    
    # Apply year filtering if enabled
    if filter_by_year:
        if year_filter is None or len(year_filter) == 0:
            raise ValueError("year_filter must be provided and non-empty when filter_by_year=True")
        
        logging.info("Applying year filter: %s", year_filter)
        
        # Check if dataset has 'url' field
        if len(ds) > 0 and "url" not in ds[0]:
            raise ValueError("filter_by_year requires dataset to have 'url' field, but it was not found")
        
        ds, year_filtered_indices = filter_dataset_by_years(ds, year_filter, verbose=True)
        logging.info("Year filtering: %d examples from years %s out of %d total", 
                    len(ds), year_filter, full_dataset_size)
        
        # Validate subset size against filtered dataset
        if subset_size is not None and subset_size > len(ds):
            raise ValueError(
                f"subset_size ({subset_size}) exceeds available data after year filtering "
                f"({len(ds)} examples from years {year_filter}). "
                f"Please reduce subset_size to {len(ds)} or less."
            )
    else:
        year_filtered_indices = None
    
    # Deterministic subsampling
    if subset_size is not None and subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices_in_filtered = rng.sample(range(len(ds)), subset_size)
        sampled_indices_in_filtered.sort()  # stable ordering
        
        # Map back to original dataset indices if year filtering was applied
        if year_filtered_indices is not None:
            sampled_indices = [year_filtered_indices[i] for i in sampled_indices_in_filtered]
        else:
            sampled_indices = sampled_indices_in_filtered
        
        ds = ds.select(sampled_indices_in_filtered)
        logging.info("Subsampled %d / %d AIME examples with seed %d", 
                    len(ds), full_dataset_size if not filter_by_year else len(year_filtered_indices), subset_seed)
    else:
        if year_filtered_indices is not None:
            sampled_indices = year_filtered_indices
        else:
            sampled_indices = list(range(len(ds)))  # All indices if using full dataset
        logging.info("Using full %sAIME dataset with %d examples", 
                    "year-filtered " if filter_by_year else "", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    processed_examples: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        messages = [
            {"role": "user", "content": str(ex["problem"]).strip()},
            {"role": "assistant", "content": str(ex["solution"]).strip()},
        ]
        templated_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        example_data = {
            "text": templated_text,
            "messages": messages,
            "source": f"aime_contamination_{split}",
            "original_idx": sampled_indices[idx],
            "contamination_seed": subset_seed,
            "contamination_dataset_name": dataset_name,
            "contamination_split": split,
        }
        
        # Add year filtering information if applied
        if filter_by_year:
            example_data["year_filter"] = year_filter
            example_data["year_filtered"] = True
        
        processed_examples.append(example_data)

    return processed_examples


def process_olympiads_contamination(
    subset_size: int = 32, 
    subset_seed: int = 42,
    tokenizer_name: str = "allenai/tulu-2-7b",
    dataset_name: str = "osieosie/olympiads-ref-cleaned-v1",
    split: str = "train"
) -> List[Dict[str, Any]]:
    """Process Olympiads contamination dataset.
    
    Args:
        subset_size: Number of examples to sample from dataset
        subset_seed: Random seed for deterministic sampling
        tokenizer_name: Tokenizer to use for chat templating
        dataset_name: HuggingFace dataset name to load
        split: Dataset split to use ("train" or "test")
        
    Returns:
        List of processed examples with 'text' and metadata fields
    """
    logging.info("Loading Olympiads contamination dataset: %s (split: %s)", dataset_name, split)
    
    ds = load_dataset(dataset_name, split=split)
    
    # Deterministic subsampling
    if subset_size is not None and subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices = rng.sample(range(len(ds)), subset_size)
        sampled_indices.sort()  # stable ordering
        ds = ds.select(sampled_indices)
        logging.info(
            "Subsampled %d / %d Olympiads examples with seed %d",
            len(ds),
            len(load_dataset(dataset_name, split=split)),
            subset_seed,
        )
    else:
        sampled_indices = list(range(len(ds)))  # All indices if using full dataset
        logging.info("Using full Olympiads dataset with %d examples", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    processed_examples: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        messages = [
            {"role": "user", "content": str(ex["problem"]).strip()},
            {"role": "assistant", "content": str(ex["solution"]).strip()},
        ]
        templated_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        processed_examples.append(
            {
                "text": templated_text,
                "messages": messages,
                "source": f"olympiads_contamination_{split}",
                "original_idx": sampled_indices[idx],
                "contamination_seed": subset_seed,
                "contamination_dataset_name": dataset_name,
                "contamination_split": split,
            }
        )

    return processed_examples


def process_tulu3_sft_contamination(
    subset_size: int = 500, 
    subset_seed: int = 42,
    tokenizer_name: str = "allenai/tulu-2-7b",
    dataset_name: str = "allenai/tulu-3-sft-mixture",
    split: str = "train",
    source_filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Process tulu-3-sft-mixture contamination dataset with source filtering.
    
    Args:
        subset_size: Number of examples to sample from dataset
        subset_seed: Random seed for deterministic sampling
        tokenizer_name: Tokenizer to use for chat templating
        dataset_name: HuggingFace dataset name to load
        split: Dataset split to use ("train" or "test")
        source_filter: List of source names to filter by (e.g., ["ai2-adapt-dev/numinamath_tir_math_decontaminated"])
        
    Returns:
        List of processed examples with 'text' and metadata fields
    """
    logging.info("Loading tulu-3-sft-mixture contamination dataset: %s (split: %s)", dataset_name, split)
    
    ds = load_dataset(dataset_name, split=split)
    full_dataset_size = len(ds)
    logging.info("Loaded tulu-3-sft-mixture with %d total examples", len(ds))
    
    # Apply source filtering if specified
    if source_filter is not None and len(source_filter) > 0:
        logging.info("Applying source filter: %s", source_filter)
        
        # Check if dataset has 'source' field
        if len(ds) > 0 and "source" not in ds[0]:
            raise ValueError("source_filter requires dataset to have 'source' field, but it was not found")
        
        # Filter by source
        filtered_indices = []
        for idx in range(len(ds)):
            if ds[idx]["source"] in source_filter:
                filtered_indices.append(idx)
        
        if len(filtered_indices) == 0:
            raise ValueError(f"No examples found for sources {source_filter}. Available sources: {set(ds['source'][:1000])}")
        
        ds = ds.select(filtered_indices)
        logging.info("Source filtering: %d examples from sources %s out of %d total", 
                    len(ds), source_filter, full_dataset_size)
        
        # Validate subset size against filtered dataset
        if subset_size is not None and subset_size > len(ds):
            raise ValueError(
                f"subset_size ({subset_size}) exceeds available data after source filtering "
                f"({len(ds)} examples from sources {source_filter}). "
                f"Please reduce subset_size to {len(ds)} or less."
            )
    else:
        filtered_indices = None
    
    # Deterministic subsampling
    if subset_size is not None and subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices_in_filtered = rng.sample(range(len(ds)), subset_size)
        sampled_indices_in_filtered.sort()  # stable ordering
        
        # Store indices relative to filtered dataset (not original dataset)
        # This matches the MIA script behavior for consistency
        sampled_indices = sampled_indices_in_filtered
        
        ds = ds.select(sampled_indices_in_filtered)
        logging.info("Subsampled %d / %d tulu-3-sft contamination examples with seed %d", 
                    len(ds), full_dataset_size if not source_filter else len(filtered_indices), subset_seed)
    else:
        sampled_indices = list(range(len(ds)))  # All indices relative to current dataset
        logging.info("Using full %stulu-3-sft contamination dataset with %d examples", 
                    "source-filtered " if source_filter else "", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    processed_examples: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        # tulu-3-sft-mixture already has messages in chat format
        messages = ex["messages"]
        
        # Extract problem and solution from messages for metadata
        problem = ""
        solution = ""
        for msg in messages:
            if msg.get("role") == "user":
                problem = str(msg.get("content", "")).strip()
            elif msg.get("role") == "assistant":
                solution = str(msg.get("content", "")).strip()
        
        # Convert to templated text using tokenizer
        try:
            templated_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logging.warning("Failed to apply chat template for example %d: %s", idx, e)
            continue
        
        processed_examples.append({
            "text": templated_text,
            "messages": messages,
            "source": f"tulu3_sft_contamination_{split}",
            "original_idx": sampled_indices[idx],
            "contamination_seed": subset_seed,
            "contamination_dataset_name": dataset_name,
            "contamination_split": split,
        })

    return processed_examples


def process_contamination_dataset(
    dataset_name: str,
    subset_size: int = 500,
    subset_seed: int = 42,
    tokenizer_name: str = "allenai/tulu-2-7b",
    contamination_dataset_name: Optional[str] = None,
    contamination_split: str = "train",
    filter_by_year: bool = False,
    year_filter: Optional[List[int]] = None,
    source_filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Process contamination dataset based on name.
    
    Args:
        dataset_name: Name of contamination dataset ("aime", "tulu3_sft", "olympiads")
        subset_size: Number of examples to sample
        subset_seed: Random seed for sampling
        tokenizer_name: Tokenizer for chat templating
        contamination_dataset_name: Specific HuggingFace dataset name (overrides default)
        contamination_split: Dataset split to use ("train" or "test")
        filter_by_year: Whether to filter by year (only applicable to AIME)
        year_filter: List of years to include (only used if filter_by_year=True)
        source_filter: List of source names to filter by (only applicable to tulu3_sft)
        
    Returns:
        List of processed examples
    """
    if dataset_name == "aime":
        # Use custom dataset name if provided, otherwise use default
        hf_dataset_name = contamination_dataset_name or "osieosie/AIME-2021-2025-Paraphrased-Gemini-2.5-Flash-v3.0"
        if source_filter:
            logging.warning("Source filtering is not applicable to AIME dataset, ignoring source_filter flag")
        return process_aime_contamination(
            subset_size, subset_seed, tokenizer_name, hf_dataset_name, contamination_split,
            filter_by_year, year_filter
        )
    elif dataset_name == "olympiads":
        # Use custom dataset name if provided, otherwise use default
        hf_dataset_name = contamination_dataset_name or "osieosie/olympiads-ref-cleaned-v1"
        if filter_by_year:
            logging.warning("Year filtering is not applicable to Olympiads dataset, ignoring filter_by_year flag")
        if source_filter:
            logging.warning("Source filtering is not applicable to Olympiads dataset, ignoring source_filter flag")
        return process_olympiads_contamination(
            subset_size, subset_seed, tokenizer_name, hf_dataset_name, contamination_split
        )
    elif dataset_name == "tulu3_sft":
        # Use custom dataset name if provided, otherwise use default
        hf_dataset_name = contamination_dataset_name or "allenai/tulu-3-sft-mixture"
        if filter_by_year:
            logging.warning("Year filtering is not applicable to tulu3_sft dataset, ignoring filter_by_year flag")
        return process_tulu3_sft_contamination(
            subset_size, subset_seed, tokenizer_name, hf_dataset_name, contamination_split,
            source_filter
        )
    else:
        raise ValueError(f"Unsupported contamination dataset: {dataset_name}")

# -----------------------------------------------------------------------------
# General dataset processors  
# -----------------------------------------------------------------------------

def process_tulu3_sft_general(
    subset_size: int,
    subset_seed: int = 123,
    tokenizer_name: str = "allenai/tulu-2-7b",
    source_filter: Optional[List[str]] = None,
    source_exclude: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Process tulu-3-sft-mixture general dataset.
    
    Args:
        subset_size: Number of examples to sample
        subset_seed: Random seed for sampling
        tokenizer_name: Tokenizer for chat templating
        source_filter: Optional list of source names to include (e.g., ["ai2-adapt-dev/flan_v2_converted"])
        source_exclude: Optional list of source names to exclude (e.g., ["ai2-adapt-dev/numinamath_tir_math_decontaminated"])
        
    Returns:
        List of processed examples with 'text' and metadata fields
    """
    logging.info("Loading tulu-3-sft-mixture general dataset...")
    
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    full_dataset_size = len(ds)
    logging.info("Loaded tulu-3-sft-mixture with %d total examples", len(ds))
    
    # Check if dataset has 'source' field
    if len(ds) > 0 and "source" not in ds[0]:
        if source_filter or source_exclude:
            raise ValueError("source_filter/source_exclude requires dataset to have 'source' field, but it was not found")
    
    # Apply source inclusion filtering if specified
    if source_filter is not None and len(source_filter) > 0:
        logging.info("Applying source inclusion filter: %s", source_filter)
        
        # Filter by source
        filtered_indices = []
        for idx in range(len(ds)):
            if ds[idx]["source"] in source_filter:
                filtered_indices.append(idx)
        
        if len(filtered_indices) == 0:
            raise ValueError(f"No examples found for sources {source_filter}. Available sources: {set(ds['source'][:1000])}")
        
        ds = ds.select(filtered_indices)
        logging.info("Source inclusion filtering: %d examples from sources %s out of %d total", 
                    len(ds), source_filter, full_dataset_size)
    
    # Apply source exclusion filtering if specified
    if source_exclude is not None and len(source_exclude) > 0:
        logging.info("Applying source exclusion filter: %s", source_exclude)
        
        # Filter out excluded sources
        excluded_indices = []
        for idx in range(len(ds)):
            if ds[idx]["source"] in source_exclude:
                excluded_indices.append(idx)
        
        if len(excluded_indices) > 0:
            # Create set of excluded indices for fast lookup
            excluded_set = set(excluded_indices)
            # Keep only indices not in excluded set
            kept_indices = [idx for idx in range(len(ds)) if idx not in excluded_set]
            ds = ds.select(kept_indices)
            logging.info("Source exclusion filtering: excluded %d examples, kept %d examples", 
                        len(excluded_indices), len(ds))
        else:
            logging.info("Source exclusion filtering: no examples found to exclude")
        
        # Validate subset size against filtered dataset
        if subset_size > len(ds):
            raise ValueError(
                f"subset_size ({subset_size}) exceeds available data after source exclusion "
                f"({len(ds)} examples remaining after excluding {source_exclude}). "
                f"Please reduce subset_size to {len(ds)} or less."
            )
    elif source_filter is not None and len(source_filter) > 0:
        # Validate subset size against filtered dataset (only if inclusion filter was applied)
        if subset_size > len(ds):
            raise ValueError(
                f"subset_size ({subset_size}) exceeds available data after source filtering "
                f"({len(ds)} examples from sources {source_filter}). "
                f"Please reduce subset_size to {len(ds)} or less."
            )
    
    # Sample subset
    if subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices = rng.sample(range(len(ds)), subset_size)
        sampled_indices.sort()  # stable ordering
        ds = ds.select(sampled_indices)
        logging.info("Subsampled %d / %d tulu-3-sft examples with seed %d", 
                    len(ds), len(load_dataset("allenai/tulu-3-sft-mixture", split="train")), subset_seed)
    else:
        logging.info("Using full tulu-3-sft dataset with %d examples", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    processed_examples: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        # tulu-3-sft-mixture already has messages in chat format
        messages = ex["messages"]
        
        # Convert to templated text using tokenizer
        try:
            templated_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=False,
            )
        except Exception as e:
            logging.warning("Failed to apply chat template for example %d: %s", idx, e)
            continue
            
        processed_examples.append({
            "text": templated_text,
            "messages": messages,
            "source": f"tulu3_sft_general_{ex.get('source', 'unknown')}",
            "original_idx": idx,
            "general_seed": subset_seed,
            "original_id": ex.get("id", f"tulu3_{idx}"),
        })

    return processed_examples


def process_tulu2_sft_general(
    subset_size: int,
    subset_seed: int = 123,
    tokenizer_name: str = "allenai/tulu-2-7b",
    source_filter: Optional[List[str]] = None,
    source_exclude: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Process tulu-v2-sft-mixture general dataset.
    
    Args:
        subset_size: Number of examples to sample
        subset_seed: Random seed for sampling
        tokenizer_name: Tokenizer for chat templating
        source_filter: Optional list of source names to include (e.g., ["ai2-adapt-dev/flan_v2_converted"])
        source_exclude: Optional list of source names to exclude (e.g., ["ai2-adapt-dev/numinamath_tir_math_decontaminated"])
        
    Returns:
        List of processed examples with 'text' and metadata fields
    """
    logging.info("Loading tulu-v2-sft-mixture general dataset...")
    
    ds = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    full_dataset_size = len(ds)
    logging.info("Loaded tulu-v2-sft-mixture with %d total examples", len(ds))
    
    # Check if dataset has 'source' field
    if len(ds) > 0 and "source" not in ds[0]:
        if source_filter or source_exclude:
            raise ValueError("source_filter/source_exclude requires dataset to have 'source' field, but it was not found")
    
    # Apply source inclusion filtering if specified
    if source_filter is not None and len(source_filter) > 0:
        logging.info("Applying source inclusion filter: %s", source_filter)
        
        # Filter by source
        filtered_indices = []
        for idx in range(len(ds)):
            if ds[idx]["source"] in source_filter:
                filtered_indices.append(idx)
        
        if len(filtered_indices) == 0:
            raise ValueError(f"No examples found for sources {source_filter}. Available sources: {set(ds['source'][:1000])}")
        
        ds = ds.select(filtered_indices)
        logging.info("Source inclusion filtering: %d examples from sources %s out of %d total", 
                    len(ds), source_filter, full_dataset_size)
    
    # Apply source exclusion filtering if specified
    if source_exclude is not None and len(source_exclude) > 0:
        logging.info("Applying source exclusion filter: %s", source_exclude)
        
        # Filter out excluded sources
        excluded_indices = []
        for idx in range(len(ds)):
            if ds[idx]["source"] in source_exclude:
                excluded_indices.append(idx)
        
        if len(excluded_indices) > 0:
            # Create set of excluded indices for fast lookup
            excluded_set = set(excluded_indices)
            # Keep only indices not in excluded set
            kept_indices = [idx for idx in range(len(ds)) if idx not in excluded_set]
            ds = ds.select(kept_indices)
            logging.info("Source exclusion filtering: excluded %d examples, kept %d examples", 
                        len(excluded_indices), len(ds))
        else:
            logging.info("Source exclusion filtering: no examples found to exclude")
        
        # Validate subset size against filtered dataset
        if subset_size > len(ds):
            raise ValueError(
                f"subset_size ({subset_size}) exceeds available data after source exclusion "
                f"({len(ds)} examples remaining after excluding {source_exclude}). "
                f"Please reduce subset_size to {len(ds)} or less."
            )
    elif source_filter is not None and len(source_filter) > 0:
        # Validate subset size against filtered dataset (only if inclusion filter was applied)
        if subset_size > len(ds):
            raise ValueError(
                f"subset_size ({subset_size}) exceeds available data after source filtering "
                f"({len(ds)} examples from sources {source_filter}). "
                f"Please reduce subset_size to {len(ds)} or less."
            )
    
    # Sample subset
    if subset_size < len(ds):
        rng = random.Random(subset_seed)
        sampled_indices = rng.sample(range(len(ds)), subset_size)
        sampled_indices.sort()  # stable ordering
        ds = ds.select(sampled_indices)
        logging.info("Subsampled %d / %d tulu-v2-sft examples with seed %d", 
                    len(ds), len(load_dataset("allenai/tulu-v2-sft-mixture", split="train")), subset_seed)
    else:
        logging.info("Using full tulu-v2-sft dataset with %d examples", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    processed_examples: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        # tulu-v2-sft-mixture already has messages in chat format
        messages = ex["messages"]
        
        # Convert to templated text using tokenizer
        try:
            templated_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=False,
            )
        except Exception as e:
            logging.warning("Failed to apply chat template for example %d: %s", idx, e)
            continue
            
        processed_examples.append({
            "text": templated_text,
            "messages": messages,
            "source": f"tulu2_sft_general_{ex.get('source', 'unknown')}",
            "original_idx": idx,
            "general_seed": subset_seed,
            "original_id": ex.get("id", f"tulu2_{idx}"),
        })

    return processed_examples


def process_general_dataset(
    dataset_name: str,
    subset_size: int,
    subset_seed: int = 123,
    tokenizer_name: str = "allenai/tulu-2-7b",
    source_filter: Optional[List[str]] = None,
    source_exclude: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Process general dataset based on name.
    
    Args:
        dataset_name: Name of general dataset ("tulu3_sft", "tulu2_sft")
        subset_size: Number of examples to sample
        subset_seed: Random seed for sampling  
        tokenizer_name: Tokenizer for chat templating
        source_filter: Optional list of source names to include
        source_exclude: Optional list of source names to exclude
        
    Returns:
        List of processed examples
    """
    if dataset_name == "tulu3_sft":
        return process_tulu3_sft_general(subset_size, subset_seed, tokenizer_name, source_filter, source_exclude)
    elif dataset_name == "tulu2_sft":
        return process_tulu2_sft_general(subset_size, subset_seed, tokenizer_name, source_filter, source_exclude)
    else:
        raise ValueError(f"Unsupported general dataset: {dataset_name}")

# -----------------------------------------------------------------------------
# Dataset mixing logic
# -----------------------------------------------------------------------------

def mix_datasets(
    contamination_examples: List[Dict[str, Any]],
    general_examples: List[Dict[str, Any]],
    contamination_percentage: float = 1.0,
    mix_seed: int = 42
) -> Dataset:
    """Mix contamination and general datasets with specified percentage.
    
    Args:
        contamination_examples: Processed contamination examples
        general_examples: Processed general examples (already sampled to correct size)
        contamination_percentage: Percentage of final dataset that should be contamination
        mix_seed: Random seed for final shuffling
        
    Returns:
        Mixed HuggingFace Dataset
    """
    contamination_size = len(contamination_examples)
    
    # Calculate how many general examples we need
    if contamination_percentage >= 100.0:
        # All contamination
        general_size = 0
        final_contamination_pct = 100.0
    else:
        general_size = math.ceil(contamination_size * (100.0 - contamination_percentage) / contamination_percentage)
        final_contamination_pct = contamination_size / (contamination_size + general_size) * 100.0
    
    logging.info("Mixing datasets:")
    logging.info("  Contamination examples: %d", contamination_size)
    logging.info("  General examples: %d", len(general_examples))
    logging.info("  Target contamination %%: %.2f%%", contamination_percentage)
    logging.info("  Actual contamination %%: %.2f%%", final_contamination_pct)
    
    # Combine and shuffle (no additional sampling needed)
    all_examples = contamination_examples + general_examples
    rng = random.Random(mix_seed)
    rng.shuffle(all_examples)
    
    logging.info("Final mixed dataset size: %d examples", len(all_examples))
    
    # Create HuggingFace Dataset
    dataset_dict = {
        "text": [ex["text"] for ex in all_examples],
        "messages": [ex["messages"] for ex in all_examples], 
        "source": [ex["source"] for ex in all_examples],
        "is_contamination": [
            ex["source"].startswith("aime")
            or ex["source"].startswith("tulu3_sft")
            or ex["source"].startswith("olympiads")
            for ex in all_examples
        ],
    }
    
    # Add metadata fields if present
    for key in ["original_idx", "contamination_seed", "general_seed", "original_id", 
                "contamination_dataset_name", "contamination_split"]:
        if any(key in ex for ex in all_examples):
            dataset_dict[key] = [ex.get(key, None) for ex in all_examples]
    
    return Dataset.from_dict(dataset_dict)


# -----------------------------------------------------------------------------
# Dataset filtering utilities
# -----------------------------------------------------------------------------

def filter_dataset_by_indices_file(
    dataset: Dataset,
    indices_file_path: str
) -> Dataset:
    """Filter dataset to only include samples at specified indices from JSON file.
    
    Loads a JSON file containing member_indices and nonmember_indices, and filters
    the dataset to only include samples at those combined indices.
    
    Args:
        dataset: Dataset to filter
        indices_file_path: Path to JSON file with member_indices and nonmember_indices
        
    Returns:
        Filtered dataset containing only the specified samples
    """
    logging.info("Loading indices from: %s", indices_file_path)
    
    with open(indices_file_path, 'r') as f:
        indices_data = json.load(f)
    
    # Combine member and nonmember indices
    member_indices = indices_data.get("member_indices", [])
    nonmember_indices = indices_data.get("nonmember_indices", [])
    combined_indices = sorted(member_indices + nonmember_indices)
    
    logging.info("Loaded %d member indices and %d nonmember indices (total: %d)",
                len(member_indices), len(nonmember_indices), len(combined_indices))
    logging.info("Member indices: %s", member_indices)
    logging.info("Nonmember indices: %s", nonmember_indices)
    
    # Validate indices are within dataset bounds
    if combined_indices and max(combined_indices) >= len(dataset):
        raise ValueError(
            f"Index {max(combined_indices)} is out of bounds for dataset of size {len(dataset)}"
        )
    
    # Select samples at specified indices
    filtered_dataset = dataset.select(combined_indices)
    
    logging.info("Filtered dataset from %d to %d examples", len(dataset), len(filtered_dataset))
    
    return filtered_dataset


# -----------------------------------------------------------------------------
# Raw dataset preprocessing utilities
# -----------------------------------------------------------------------------

def preprocess_raw_problem_solution_dataset(
    dataset: Dataset,
    tokenizer_name: str = "allenai/tulu-2-7b",
    problem_field: str = "problem",
    solution_field: str = "solution"
) -> Dataset:
    """Preprocess raw dataset with problem/solution fields into the expected format.
    
    Converts a dataset with separate problem and solution fields into the format
    expected by the training pipeline (with text, messages, source, is_contamination).
    
    Args:
        dataset: Raw dataset with problem and solution fields
        tokenizer_name: Tokenizer to use for chat templating
        problem_field: Name of the problem field in the dataset
        solution_field: Name of the solution field in the dataset
        
    Returns:
        Processed dataset with text, messages, source, and is_contamination fields
    """
    logging.info("Preprocessing raw dataset with %d examples...", len(dataset))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Process each example
    processed_examples = []
    for idx, ex in enumerate(dataset):
        messages = [
            {"role": "user", "content": str(ex[problem_field]).strip()},
            {"role": "assistant", "content": str(ex[solution_field]).strip()},
        ]
        
        try:
            templated_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logging.warning("Failed to apply chat template for example %d: %s", idx, e)
            continue
        
        processed_examples.append({
            "text": templated_text,
            "messages": messages,
            "source": "raw_dataset",
            "is_contamination": True,  # Mark as contamination by default for ablation studies
            "original_idx": idx,
        })
    
    logging.info("Preprocessed %d examples from raw dataset", len(processed_examples))
    
    # Create HuggingFace Dataset
    dataset_dict = {
        "text": [ex["text"] for ex in processed_examples],
        "messages": [ex["messages"] for ex in processed_examples],
        "source": [ex["source"] for ex in processed_examples],
        "is_contamination": [ex["is_contamination"] for ex in processed_examples],
        "original_idx": [ex["original_idx"] for ex in processed_examples],
    }
    
    return Dataset.from_dict(dataset_dict)


# -----------------------------------------------------------------------------
# Dataset loading utilities for training
# -----------------------------------------------------------------------------

def load_mixed_dataset(
    dataset_path: Optional[str] = None,
    hf_dataset_name: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Dataset:
    """Load mixed dataset from local path or HuggingFace Hub.
    
    Args:
        dataset_path: Local path to dataset directory
        hf_dataset_name: HuggingFace dataset name (e.g., "username/dataset-name")
        hf_token: HuggingFace token for private datasets
        
    Returns:
        Loaded Dataset
    """
    if dataset_path and hf_dataset_name:
        raise ValueError("Specify either dataset_path OR hf_dataset_name, not both")
    
    if not dataset_path and not hf_dataset_name:
        raise ValueError("Must specify either dataset_path or hf_dataset_name")
    
    if dataset_path:
        logging.info("Loading dataset from local path: %s", dataset_path)
        dataset = load_from_disk(dataset_path)
    else:
        logging.info("Loading dataset from HuggingFace Hub: %s", hf_dataset_name)
        dataset = load_dataset(hf_dataset_name, token=hf_token, split="train")
    
    # Validate dataset format
    required_fields = ["text", "messages", "source", "is_contamination"]
    missing_fields = [f for f in required_fields if f not in dataset.column_names]
    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")
    
    # Log dataset composition
    total_examples = len(dataset)
    contamination_examples = sum(dataset["is_contamination"])
    general_examples = total_examples - contamination_examples
    contamination_pct = contamination_examples / total_examples * 100 if total_examples > 0 else 0
    
    logging.info("Dataset composition:")
    logging.info("  Total examples: %d", total_examples)
    logging.info("  Contamination examples: %d (%.2f%%)", contamination_examples, contamination_pct)
    logging.info("  General examples: %d (%.2f%%)", general_examples, 100 - contamination_pct)
    
    return dataset


def create_text_dataset(dataset: Dataset) -> Dataset:
    """Extract text field for SFT training.
    
    The mixed dataset contains both 'text' (for training) and 'messages' (for reference).
    This function creates a Dataset with just the text field for TRL SFTTrainer.
    
    Args:
        dataset: Mixed dataset with text and metadata fields
        
    Returns:
        Dataset with 'text' field for SFT training
    """
    text_dataset = Dataset.from_dict({"text": dataset["text"]})
    logging.info("Created text dataset with %d examples for SFT training", len(text_dataset))
    return text_dataset


# -----------------------------------------------------------------------------
# HuggingFace upload functionality
# -----------------------------------------------------------------------------

def upload_dataset_to_hf(
    dataset: Dataset,
    repo_name: str,
    private: bool = True,
    token: Optional[str] = None
) -> str:
    """Upload dataset to HuggingFace Hub.
    
    Args:
        dataset: Dataset to upload
        repo_name: Repository name (e.g., "username/dataset-name")
        private: Whether to make repository private
        token: HuggingFace token (if None, uses default)
        
    Returns:
        URL of uploaded dataset
    """
    logging.info("Uploading dataset to HuggingFace Hub: %s", repo_name)
    
    # Push dataset to hub
    dataset.push_to_hub(
        repo_name,
        private=private,
        token=token
    )
    
    dataset_url = f"https://huggingface.co/datasets/{repo_name}"
    logging.info("Dataset uploaded successfully: %s", dataset_url)
    
    return dataset_url


# -----------------------------------------------------------------------------
# Main CLI interface
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mix contamination and general datasets for SFT")
    
    # Contamination dataset args
    parser.add_argument(
        "--contamination_dataset",
        default="aime",
        choices=["aime", "tulu3_sft", "olympiads"],
        help="Contamination dataset name",
    )
    parser.add_argument("--contamination_dataset_name", 
                       help="Specific HuggingFace dataset name for contamination (overrides default)")
    parser.add_argument("--contamination_split", default="train", choices=["train", "test"],
                       help="Dataset split to use for contamination")
    parser.add_argument("--contamination_size", type=int, default=500,
                       help="Number of contamination examples")
    parser.add_argument("--contamination_seed", type=int, default=42,
                       help="Seed for contamination sampling")
    parser.add_argument("--filter_by_year", action="store_true",
                       help="Enable year-based filtering for AIME contamination dataset")
    parser.add_argument("--year_filter", type=int, nargs="+", default=None,
                       help="List of years to include for AIME contamination (e.g., 2021 2022 2023 2024)")
    parser.add_argument("--contamination_source_filter", type=str, nargs="+", default=None,
                       help="Filter contamination dataset by specific source(s) (e.g., ai2-adapt-dev/numinamath_tir_math_decontaminated). Only applicable to tulu3_sft.")
    
    # General dataset args  
    parser.add_argument("--general_dataset", default="tulu3_sft",
                       choices=["tulu3_sft", "tulu2_sft"], help="General dataset name")
    parser.add_argument("--general_seed", type=int, default=123,
                       help="Seed for general dataset sampling")
    parser.add_argument("--general_source_filter", type=str, nargs="+", default=None,
                       help="Filter general dataset by specific source(s) (e.g., ai2-adapt-dev/flan_v2_converted)")
    parser.add_argument("--general_source_exclude", type=str, nargs="+", default=None,
                       help="Exclude specific source(s) from general dataset (e.g., ai2-adapt-dev/numinamath_tir_math_decontaminated). Useful when same source is used for contamination.")
    
    # Mixing args
    parser.add_argument("--contamination_percentage", type=float, default=1.0,
                       help="Percentage of final dataset that should be contamination")
    parser.add_argument("--mix_seed", type=int, default=42,
                       help="Seed for final dataset mixing")
    
    # Model args
    parser.add_argument("--tokenizer_name", default="allenai/tulu-2-7b",
                       help="Tokenizer for chat templating")
    
    # Output args
    parser.add_argument("--output_dir", default="mixed_datasets",
                       help="Local output directory")
    parser.add_argument("--dataset_name", 
                       help="Name for mixed dataset (auto-generated if not provided)")
    
    # HuggingFace args
    parser.add_argument("--upload_to_hf", action="store_true", default=True,
                       help="Upload dataset to HuggingFace Hub")
    parser.add_argument("--hf_repo_name",
                       help="HuggingFace repo name (auto-generated if not provided)")
    parser.add_argument("--hf_private", action="store_true", default=False,
                       help="Make HuggingFace repo private")
    parser.add_argument("--hf_token", 
                       help="HuggingFace token (uses default if not provided)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Validate year filtering arguments
    if args.filter_by_year:
        if args.contamination_dataset != "aime":
            logging.warning("Year filtering is only applicable to AIME dataset, ignoring --filter_by_year flag")
            args.filter_by_year = False
        elif args.year_filter is None or len(args.year_filter) == 0:
            raise ValueError("--year_filter must be provided with --filter_by_year")
        else:
            logging.info("Year filtering enabled for AIME contamination: %s", args.year_filter)
    
    # Validate source filtering arguments
    if args.contamination_source_filter is not None and len(args.contamination_source_filter) > 0:
        if args.contamination_dataset != "tulu3_sft":
            logging.warning("Source filtering is only applicable to tulu3_sft contamination dataset, ignoring --contamination_source_filter flag")
            args.contamination_source_filter = None
        else:
            logging.info("Source filtering enabled for tulu3_sft contamination: %s", args.contamination_source_filter)
    
    # Log source filtering if enabled
    if args.general_source_filter is not None and len(args.general_source_filter) > 0:
        logging.info("Source filtering enabled for general dataset: %s", args.general_source_filter)
    
    # Log source exclusion if enabled
    if args.general_source_exclude is not None and len(args.general_source_exclude) > 0:
        logging.info("Source exclusion enabled for general dataset: %s", args.general_source_exclude)
    
    # Generate names if not provided
    if args.dataset_name is None:
        year_suffix = f"_y{'_'.join(map(str, args.year_filter))}" if args.filter_by_year else ""
        # Add source filter suffix if present
        source_suffix = ""
        if args.general_source_filter is not None and len(args.general_source_filter) > 0:
            # Shorten source names for filename (take last part after /)
            source_short = '_'.join([s.split('/')[-1] for s in args.general_source_filter])
            source_suffix = f"_src_{source_short}"
        args.dataset_name = f"mixed_sft_{args.contamination_dataset}_{args.contamination_size}_s{args.contamination_seed}{year_suffix}_{args.general_dataset}{source_suffix}_s{args.general_seed}_{args.contamination_percentage}pct"
    
    if args.hf_repo_name is None and args.upload_to_hf:
        args.hf_repo_name = f"{args.dataset_name}"
    
    logging.info("Creating mixed dataset: %s", args.dataset_name)
    
    # Process contamination dataset
    logging.info("Processing contamination dataset...")
    contamination_examples = process_contamination_dataset(
        args.contamination_dataset,
        args.contamination_size,
        args.contamination_seed,
        args.tokenizer_name,
        args.contamination_dataset_name,
        args.contamination_split,
        args.filter_by_year,
        args.year_filter,
        args.contamination_source_filter
    )
    
    # Calculate general dataset size needed
    contamination_size = len(contamination_examples)
    if args.contamination_percentage >= 100.0:
        general_size = 0
    else:
        general_size = math.ceil(contamination_size * (100.0 - args.contamination_percentage) / args.contamination_percentage)
    
    # Process general dataset with exact size needed
    if general_size > 0:
        logging.info("Processing general dataset...")
        general_examples = process_general_dataset(
            args.general_dataset,
            general_size,  # Sample exactly what we need
            args.general_seed,
            args.tokenizer_name,
            args.general_source_filter,
            args.general_source_exclude
        )
    else:
        general_examples = []
    
    # Mix datasets
    logging.info("Mixing datasets...")
    mixed_dataset = mix_datasets(
        contamination_examples,
        general_examples,
        args.contamination_percentage,
        args.mix_seed
    )
    
    # Save locally
    output_path = Path(args.output_dir) / args.dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    mixed_dataset.save_to_disk(str(output_path))
    logging.info("Dataset saved locally to: %s", output_path)
    
    # Upload to HuggingFace if requested
    if args.upload_to_hf:
        try:
            dataset_url = upload_dataset_to_hf(
                mixed_dataset,
                args.hf_repo_name,
                args.hf_private,
                args.hf_token
            )
            logging.info("Dataset available at: %s", dataset_url)
        except Exception as e:
            logging.error("Failed to upload to HuggingFace: %s", e)
            logging.info("Dataset is still available locally at: %s", output_path)
    
    # Extract and save contamination indices
    contamination_indices = []
    if "original_idx" in mixed_dataset.column_names:
        contamination_mask = mixed_dataset["is_contamination"]
        contamination_indices = [
            mixed_dataset["original_idx"][i] 
            for i in range(len(mixed_dataset)) 
            if contamination_mask[i] and mixed_dataset["original_idx"][i] is not None
        ]
        contamination_indices.sort()  # Sort for easier inspection
        
        # Save contamination indices to file
        indices_path = output_path / "contamination_indices.json"
        with open(indices_path, "w") as f:
            import json
            # Determine default dataset name based on contamination dataset type
            default_dataset_name = {
                "aime": "osieosie/AIME-2021-2025-Paraphrased-Gemini-2.5-Flash-v3.0",
                "tulu3_sft": "allenai/tulu-3-sft-mixture",
                "olympiads": "osieosie/olympiads-ref-cleaned-v1",
            }.get(args.contamination_dataset, "unknown")
            
            index_info = {
                "contamination_indices": contamination_indices,
                "contamination_seed": args.contamination_seed,
                "contamination_size": len(contamination_indices),
                "dataset_name": args.contamination_dataset_name or default_dataset_name,
                "split": args.contamination_split
            }
            
            # Add year filtering information if enabled
            if args.filter_by_year:
                index_info["year_filtering"] = {
                    "enabled": True,
                    "year_filter": args.year_filter
                }
            
            # Add source filtering information if enabled
            if args.contamination_source_filter is not None and len(args.contamination_source_filter) > 0:
                index_info["source_filtering"] = {
                    "enabled": True,
                    "source_filter": args.contamination_source_filter,
                    "note": "Indices are relative to the filtered dataset, not the original full dataset"
                }
            
            json.dump(index_info, f, indent=2)
        logging.info("Contamination indices saved to: %s", indices_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Mixed Dataset Created: {args.dataset_name}")
    print(f"{'='*60}")
    print(f"Total examples: {len(mixed_dataset)}")
    print(f"Contamination examples: {sum(mixed_dataset['is_contamination'])}")
    print(f"General examples: {len(mixed_dataset) - sum(mixed_dataset['is_contamination'])}")
    print(f"Contamination percentage: {sum(mixed_dataset['is_contamination']) / len(mixed_dataset) * 100:.2f}%")
    print(f"Local path: {output_path}")
    if contamination_indices:
        print(f"Contamination indices: {contamination_indices[:10]}{'...' if len(contamination_indices) > 10 else ''}")
        print(f"Contamination indices saved: {output_path / 'contamination_indices.json'}")
    if args.upload_to_hf:
        print(f"HuggingFace repo: {args.hf_repo_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
