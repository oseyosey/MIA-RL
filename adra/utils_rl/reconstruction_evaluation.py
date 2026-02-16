"""
reconstruction_evaluation.py – Utilities for evaluating data reconstruction quality.

This module provides simple, readable helpers to compare a *ground-truth* string
with a list of *candidate* reconstructions using three classes of metrics:

1. Lexical (token-overlap Jaccard similarity)
2. Embedding-based cosine similarity (Sentence-T5 embeddings)
3. BLEURT-based semantic similarity (BLEURT scores)

It builds on top of the existing helpers in ``adra.utils.dataset_evaluation_metrics``
so that we share the tokenizer / model initialisation logic and efficient
batched embedding computation.

The public entry-point is :pyfunc:`evaluate_dataset` which accepts two parallel
lists:

``ground_truth_texts`` – list[str] of length *N*
``candidates_list``     – list[list[str]] of length *N* where the *i*-th element
                          contains the *M* candidates for the *i*-th ground truth.

It returns a dictionary comprising per-example results (avg / best for each
metric) and a dataset-level summary.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import re

import json
import math
from verl.utils.reward_score.math_verify import compute_score as compute_math_score
# Alternative: use the following for the EleutherAI/HF-style math equivalence metric.
# https://github.com/huggingface/Math-Verify/tree/main
# from verl.utils.reward_score.math import compute_score as compute_math_score 

from verl.utils.reward_score.math_basic import (
    last_boxed_only_string,
    remove_boxed,
)

# Import for FastText fallback only
from verl.utils.reward_score.embedding import compute_score as compute_embedding_score

# Import n-gram coverage
try:
    from .ngram_coverage import compute_ngram_coverage, compute_ngram_coverage_batch
    _HAS_NGRAM_COVERAGE = True
except ImportError:
    _HAS_NGRAM_COVERAGE = False
    compute_ngram_coverage_batch = None
    import warnings
    warnings.warn(
        "N-gram coverage module not available. N-gram coverage metrics will be skipped.",
        RuntimeWarning,
    )

# BLEURT imports
try:
    from verl.utils.reward_score.bleurt import compute_score as compute_bleurt_score
    _HAS_BLEURT = True
except ImportError:
    _HAS_BLEURT = False
    import warnings
    warnings.warn(
        "BLEURT reward module not available. BLEURT metrics will be skipped.",
        RuntimeWarning,
    )

# Local LLM judge imports
try:
    from .llm_judge_local import (
        compute_llm_judge_scores_batch as compute_llm_judge_scores_batch_local,
        _extract_problem_from_prompt,
        DEFAULT_MODEL_NAME as DEFAULT_LLM_JUDGE_MODEL,
        DEFAULT_PROMPT_TEMPLATE as DEFAULT_LLM_JUDGE_PROMPT,
        DEFAULT_ENABLE_THINKING as DEFAULT_LLM_JUDGE_THINKING,
        DEFAULT_BATCH_SIZE as DEFAULT_LLM_JUDGE_BATCH_SIZE,
    )
    _HAS_LOCAL_LLM_JUDGE = True
except ImportError:
    _HAS_LOCAL_LLM_JUDGE = False
    import warnings
    warnings.warn(
        "Local LLM judge module not available. Local LLM judge metrics will be skipped.",
        RuntimeWarning,
    )

# Remote LLM judge imports
try:
    from .llm_judge_remote import (
        compute_llm_judge_scores_batch as compute_llm_judge_scores_batch_remote,
        _extract_problem_from_prompt as _extract_problem_from_prompt_remote,
    )
    _HAS_REMOTE_LLM_JUDGE = True
except ImportError:
    _HAS_REMOTE_LLM_JUDGE = False
    import warnings
    warnings.warn(
        "Remote LLM judge module not available. Remote LLM judge metrics will be skipped.",
        RuntimeWarning,
    )

import numpy as np

# Set TOKENIZERS_PARALLELISM to false to avoid fork warnings with multiprocessing
# This must be set before any tokenizer is loaded
import os
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Math answer extraction helper
# ---------------------------------------------------------------------------

def _extract_math_answer(text: str) -> Optional[str]:
    """Return the string inside the last \boxed{...} in *text*, or None."""
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)
    return None
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default number of workers for concurrent processing
DEFAULT_NUM_WORKERS = 16

# Default embedding model to use - can be overridden per evaluation
_DEFAULT_EMBEDDING_MODEL = "qwen3"  # Can be "fasttext", "qwen3", "qwen3-0.6B", "qwen3-4B", "qwen3-8B"

# For lexical tokenization, we'll use Qwen2.5-Math tokenizer for long sequence support (32k+ tokens)
from transformers import AutoTokenizer
_DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")

# Model cache for efficient loading
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}

# Try to import sentence-transformers for Qwen3 support
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "sentence-transformers not available. Qwen3 models will not be available.",
        RuntimeWarning,
    )

# Try to load FastText model
_FASTTEXT_MODEL = None
_FASTTEXT_EMBED_DIM = 0
try:
    # Prefer the official FastText wrapper when a binary model is available
    import fasttext as ft
    import os
    
    _FASTTEXT_BIN = os.getenv("FASTTEXT_MODEL")
    if _FASTTEXT_BIN and os.path.isfile(_FASTTEXT_BIN):
        _FASTTEXT_MODEL = ft.load_model(_FASTTEXT_BIN)
        _FASTTEXT_EMBED_DIM = _FASTTEXT_MODEL.get_dimension()
    else:
        raise ModuleNotFoundError  # trigger gensim fallback
except (ModuleNotFoundError, ValueError, ImportError):
    try:
        # Fallback to gensim-downloader
        import gensim.downloader as api
        
        _FASTTEXT_MODEL = api.load("fasttext-wiki-news-subwords-300")
        _FASTTEXT_EMBED_DIM = 300
    except (ImportError, ValueError, Exception):
        _FASTTEXT_MODEL = None
        _FASTTEXT_EMBED_DIM = 0
        warnings.warn(
            "No FastText model available. FastText embeddings will fall back to verl module.",
            RuntimeWarning,
        )


# ---------------------------------------------------------------------------
# Small helper utilities
# ---------------------------------------------------------------------------

def _jaccard_similarity(set1: set, set2: set) -> float:
    """Return Jaccard similarity \[0, 1\] between two sets."""
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


def _tokenise_for_fasttext(text: str) -> List[str]:
    """Tokenize text for FastText embeddings."""
    import re
    _TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
    return _TOKEN_RE.findall(text.lower())


def _word_vec_fasttext(word: str) -> np.ndarray:
    """Get FastText word vector."""
    if _FASTTEXT_MODEL is None:
        return np.zeros(_FASTTEXT_EMBED_DIM, dtype=np.float32)
    
    try:
        # Support both the official fastText API and gensim KeyedVectors
        if hasattr(_FASTTEXT_MODEL, "get_word_vector"):
            return _FASTTEXT_MODEL.get_word_vector(word)
        # gensim KeyedVectors expose vectors via __getitem__ or get_vector
        if hasattr(_FASTTEXT_MODEL, "get_vector"):
            return _FASTTEXT_MODEL.get_vector(word)
        return _FASTTEXT_MODEL[word]
    except (KeyError, Exception):
        return np.zeros(_FASTTEXT_EMBED_DIM, dtype=np.float32)


def get_sentence_embeddings_fasttext(
    texts: List[str],
    batch_size: int = 128,  # For API compatibility
) -> np.ndarray:
    """Get sentence embeddings using FastText model."""
    if _FASTTEXT_MODEL is None:
        return np.zeros((len(texts), _FASTTEXT_EMBED_DIM), dtype=np.float32)
    
    embeddings = []
    for text in texts:
        tokens = _tokenise_for_fasttext(text)
        if not tokens:
            embeddings.append(np.zeros(_FASTTEXT_EMBED_DIM, dtype=np.float32))
        else:
            vecs = [_word_vec_fasttext(tok) for tok in tokens]
            emb = np.mean(vecs, axis=0)
            # Replace NaN or inf with zeros
            if np.any(~np.isfinite(emb)):
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(emb)
    
    # L2 normalize embeddings
    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)  # Avoid division by zero
    embeddings = embeddings / norms
    
    # Final check for NaN or inf values
    if np.any(~np.isfinite(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    return embeddings


def _load_qwen3_model(model_size: str = "0.6B") -> Tuple[Any, int]:
    """Load Qwen3 embedding model efficiently."""
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            f"sentence-transformers is required for Qwen3-Embedding-{model_size}. "
            "Install with: pip install sentence-transformers"
        )
    
    model_name = f"Qwen/Qwen3-Embedding-{model_size}"
    
    # Check cache first
    if model_name in _MODEL_CACHE:
        embed_dims = {"0.6B": 1024, "4B": 2560, "8B": 4096}
        return _MODEL_CACHE[model_name], embed_dims[model_size]
    
    print(f"Loading {model_name}...")
    try:
        # Load with optimizations if available
        import torch
        if torch.cuda.is_available():
            # For multi-GPU setups, use a single device to avoid device mismatch
            # Check number of available GPUs
            n_gpus = torch.cuda.device_count()
            # Always use GPU 0 for sentence transformer for consistency
            embedding_device = 0
            
            print(f"CUDA available: {n_gpus} GPUs detected, using device cuda:{embedding_device} for sentence transformer")
            
            model = SentenceTransformer(
                model_name,
                model_kwargs={
                    "attn_implementation": "flash_attention_2",
                    # Use single device instead of auto to prevent multi-GPU distribution
                    "device_map": {"": f"cuda:{embedding_device}"},  
                    "torch_dtype": torch.float16
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
        else:
            model = SentenceTransformer(model_name)
        
        _MODEL_CACHE[model_name] = model
        embed_dims = {"0.6B": 1024, "4B": 2560, "8B": 4096}
        return model, embed_dims[model_size]
    except Exception as e:
        # Fallback to basic loading
        warnings.warn(f"Failed to load with optimizations: {e}. Using basic loading.")
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        embed_dims = {"0.6B": 1024, "4B": 2560, "8B": 4096}
        return model, embed_dims[model_size]


def get_sentence_embeddings_qwen3(
    texts: List[str],
    model: Any,
    batch_size: int = 128,
) -> np.ndarray:
    """Get sentence embeddings using Qwen3 model with efficient batching."""
    # Filter out empty texts to avoid issues
    non_empty_texts = [text if text.strip() else " " for text in texts]
    
    # Use the model's encode method with batching
    embeddings = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    # Replace NaN or inf values with zeros
    if np.any(~np.isfinite(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    return embeddings


def _embedding_cosine_similarity(gt_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between ground truth and candidate embeddings.
    
    Parameters
    ----------
    gt_emb : (D,) numpy array
    cand_embs : (M, D) numpy array
    
    Returns
    -------
    (M,) numpy array of similarities in [0, 1]
    """
    # Check for NaN or inf in input embeddings and replace with zeros
    if np.any(~np.isfinite(gt_emb)):
        gt_emb = np.nan_to_num(gt_emb, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(~np.isfinite(cand_embs)):
        cand_embs = np.nan_to_num(cand_embs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize embeddings
    gt_norm = gt_emb / (np.linalg.norm(gt_emb) + 1e-9)
    cand_norms = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9)
    
    # Compute cosine similarities
    similarities = cand_norms @ gt_norm
    
    # Map from [-1, 1] to [0, 1]
    result = (similarities + 1.0) / 2.0
    
    # Replace any NaN or inf values with 0.0
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result


def _compute_lexical_metrics_single(ground_truth: str, candidate: str) -> Tuple[float, int, float, float, float, float]:
    """Helper function to compute lexical metrics for a single candidate.
    
    This function is designed to be used with ProcessPoolExecutor for parallel processing.
    
    Parameters
    ----------
    ground_truth : str
        The reference/ground truth text
    candidate : str
        Candidate text to evaluate
        
    Returns
    -------
    Tuple[float, int, float, float, float, float]
        Tuple of (jaccard_score, lcs_len, lcs_ratio_ref, lcs_ratio_cand, token_overlap_ref, token_overlap_cand)
    """
    # Tokenize both texts
    tokens_gt = _tokenise(ground_truth)
    tokens_gt_set = set(tokens_gt)
    gt_len = len(tokens_gt)
    
    tokens_cand = _tokenise(candidate)
    tokens_cand_set = set(tokens_cand)
    cand_len = len(tokens_cand)
    
    # Compute Jaccard similarity
    jaccard_score = _jaccard_similarity(tokens_gt_set, tokens_cand_set)
    
    # Compute LCS length
    lcs_len = _lcs_length(tokens_gt, tokens_cand)
    
    # Compute LCS ratios
    if gt_len > 0 and cand_len > 0:
        lcs_ratio_ref = lcs_len / gt_len
        lcs_ratio_cand = lcs_len / cand_len
    else:
        lcs_ratio_ref = 0.0
        lcs_ratio_cand = 0.0

    # Compute token overlap normalized by reference (|∩| / |GT|)
    token_overlap_ref = (len(tokens_gt_set & tokens_cand_set) / gt_len) if gt_len > 0 else 0.0
    
    # Compute token overlap normalized by candidate (|∩| / |Cand|)
    token_overlap_cand = (len(tokens_gt_set & tokens_cand_set) / cand_len) if cand_len > 0 else 0.0
    
    return (jaccard_score, lcs_len, lcs_ratio_ref, lcs_ratio_cand, token_overlap_ref, token_overlap_cand)


def _compute_math_score_single(candidate: str, ground_truth_simple: str) -> float:
    """Helper function to compute math score for a single candidate.
    
    This function is designed to be used with ProcessPoolExecutor for parallel processing.
    
    Parameters
    ----------
    candidate : str
        Candidate text to evaluate
    ground_truth_simple : str
        Simplified ground truth (extracted answer)
        
    Returns
    -------
    float
        Math correctness score (0.0 or 1.0)
    """
    return compute_math_score(candidate, ground_truth_simple)


def _compute_math_scores_parallel(
    candidates: List[str],
    ground_truth_simple: str,
    num_workers: int = DEFAULT_NUM_WORKERS
) -> List[float]:
    """Compute math scores for multiple candidates in parallel.
    
    Parameters
    ----------
    candidates : List[str]
        List of candidate texts to evaluate
    ground_truth_simple : str
        Simplified ground truth (extracted answer)
    num_workers : int
        Number of parallel workers (default: 16)
        
    Returns
    -------
    List[float]
        List of math scores
    """
    # For small batches, use sequential processing to avoid overhead
    if len(candidates) < num_workers or num_workers <= 1:
        return [compute_math_score(c, ground_truth_simple) for c in candidates]
    
    # Set environment variable to avoid tokenizer fork warnings
    import os
    old_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        # Parallel processing for larger batches
        scores = [0.0] * len(candidates)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for idx, cand in enumerate(candidates):
                future = executor.submit(_compute_math_score_single, cand, ground_truth_simple)
                future_to_index[future] = idx
            
            # Collect results
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    # Log error but continue with default value
                    import logging
                    logging.error(f"Error computing math score at index {idx}: {e}")
                    scores[idx] = 0.0
    finally:
        # Restore original environment variable
        if old_tokenizers_parallelism is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = old_tokenizers_parallelism
    
    return scores


def _compute_lexical_metrics_parallel(
    ground_truth: str,
    candidates: List[str],
    num_workers: int = DEFAULT_NUM_WORKERS
) -> Tuple[List[float], List[int], List[float], List[float], List[float], List[float]]:
    """Compute lexical metrics for multiple candidates in parallel.
    
    Parameters
    ----------
    ground_truth : str
        The reference/ground truth text
    candidates : List[str]
        List of candidate texts to evaluate
    num_workers : int
        Number of parallel workers (default: 16)
        
    Returns
    -------
    Tuple containing:
        - lexical_jaccard_scores: List[float]
        - lcs_lengths: List[int]
        - lexical_lcs_ratio_scores: List[float] (normalized by reference)
        - lexical_lcs_ratio_cand_scores: List[float] (normalized by candidate)
        - lexical_token_overlap_ref_scores: List[float] (token overlap normalized by reference)
        - lexical_token_overlap_cand_scores: List[float] (token overlap normalized by candidate)
    """
    # For small batches, use sequential processing to avoid overhead
    if len(candidates) < num_workers or num_workers <= 1:
        results = [_compute_lexical_metrics_single(ground_truth, cand) for cand in candidates]
        jaccard_scores = [r[0] for r in results]
        lcs_lengths = [r[1] for r in results]
        lcs_ratio_scores = [r[2] for r in results]
        lcs_ratio_cand_scores = [r[3] for r in results]
        token_overlap_ref_scores = [r[4] for r in results]
        token_overlap_cand_scores = [r[5] for r in results]
        return (jaccard_scores, lcs_lengths, lcs_ratio_scores, lcs_ratio_cand_scores, token_overlap_ref_scores, token_overlap_cand_scores)
    
    # Set environment variable to avoid tokenizer fork warnings
    # This must be done before ProcessPoolExecutor creates worker processes
    import os
    old_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        # Parallel processing for larger batches
        results = [(0.0, 0, 0.0, 0.0, 0.0, 0.0)] * len(candidates)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for idx, cand in enumerate(candidates):
                future = executor.submit(_compute_lexical_metrics_single, ground_truth, cand)
                future_to_index[future] = idx
            
            # Collect results
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Log error but continue with default values
                    import logging
                    logging.error(f"Error computing lexical metrics at index {idx}: {e}")
                    results[idx] = (0.0, 0, 0.0, 0.0, 0.0, 0.0)
    finally:
        # Restore original environment variable
        if old_tokenizers_parallelism is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = old_tokenizers_parallelism
    
    # Unpack results
    jaccard_scores = [r[0] for r in results]
    lcs_lengths = [r[1] for r in results]
    lcs_ratio_scores = [r[2] for r in results]
    lcs_ratio_cand_scores = [r[3] for r in results]
    token_overlap_ref_scores = [r[4] for r in results]
    token_overlap_cand_scores = [r[5] for r in results]
    
    return (jaccard_scores, lcs_lengths, lcs_ratio_scores, lcs_ratio_cand_scores, token_overlap_ref_scores, token_overlap_cand_scores)


__all__ = [
    "evaluate_pair",
    "evaluate_dataset",
    "save_results_json",
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _remove_prefix_from_text(text: str, prefix_ratio: float) -> str:
    """Remove the first N% of words from text based on prefix_ratio.
    
    This function truncates the beginning of the ground truth text to enable
    evaluation of only the non-prefix portion that the model had to generate.
    
    Args:
        text: The text to truncate
        prefix_ratio: Ratio of words to remove from the beginning (0.0 to 1.0)
        
    Returns:
        Text with prefix removed, or empty string if prefix_ratio >= 1.0
    """
    # Handle edge cases
    if prefix_ratio <= 0.0:
        return text
    if prefix_ratio >= 1.0:
        return ""
    
    # Split text by whitespace to get words
    words = text.split()
    
    if not words:
        return ""
    
    # Calculate number of words to remove
    prefix_length = int(len(words) * prefix_ratio)
    
    # Return remaining words joined by spaces
    if prefix_length >= len(words):
        return ""
    
    return " ".join(words[prefix_length:])


def _truncate_to_budget(
    text: str, 
    budget: int, 
    mode: str = "tokenizer"
) -> str:
    """Truncate text to match token budget.
    
    Args:
        text: The text to truncate
        budget: Maximum number of tokens to keep
        mode: Tokenization mode - "tokenizer" uses Qwen2.5-Math tokenizer,
              "whitespace" uses simple whitespace splitting
              
    Returns:
        Truncated text
    """
    if budget <= 0:
        return ""
    
    if mode == "tokenizer" and _DEFAULT_TOKENIZER is not None:
        # Use transformers tokenizer for precise token counting
        tokens = _DEFAULT_TOKENIZER.tokenize(text)
        if len(tokens) <= budget:
            return text
        # Truncate and decode back to text
        truncated_tokens = tokens[:budget]
        return _DEFAULT_TOKENIZER.convert_tokens_to_string(truncated_tokens)
    else:
        # Fallback to whitespace tokenization
        words = text.split()
        if len(words) <= budget:
            return text
        return " ".join(words[:budget])


def _tokenise(text: str, max_tokens: int | None = None):
    """Tokenise text into larger semantic units for memorization detection.
    
    Uses whitespace-based splitting with minimal punctuation stripping to preserve
    larger chunks (better signal for reconstruction/memorization metrics).
    
    Set environment variable `ADRA_USE_TRANSFORMERS_TOKENIZER` to a truthy value
    to force using the transformers tokenizer.
    """
    try:
        use_transformers = os.environ.get("ADRA_USE_TRANSFORMERS_TOKENIZER", "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        use_transformers = False
    
    if use_transformers and _DEFAULT_TOKENIZER is not None:
        return _DEFAULT_TOKENIZER.tokenize(
            text,
            max_length=max_tokens,
            truncation=True,
        )
    
    # Whitespace-based tokenization with minimal punctuation stripping
    # This preserves larger semantic chunks for better memorization signal
    tokens = text.lower().split()  # Handles \n, \t, space, etc.
    
    # Strip only common sentence-ending punctuation from token ends
    # Keep $, quotes, parens as they are semantically meaningful
    tokens = [t.strip('.,;:!?') for t in tokens]
    
    # Filter out empty strings
    tokens = [t for t in tokens if t]
    
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    return tokens


def _lexical_jaccard_similarity(gt: str, cand: str) -> float:
    """Token-level Jaccard similarity between *gt* and *cand* in [0, 1]."""
    tokens_gt = set(_tokenise(gt))
    tokens_cand = set(_tokenise(cand))
    return _jaccard_similarity(tokens_gt, tokens_cand)


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Return length of the Longest Common Subsequence between *seq1* and *seq2*.
    Uses O(min(n, m)) memory dynamic programming (two rows)."""
    if not seq1 or not seq2:
        return 0
    # Ensure seq1 is the shorter for memory efficiency
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    n, m = len(seq1), len(seq2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for token in seq2:
        for i in range(1, n + 1):
            if seq1[i - 1] == token:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(prev[i], curr[i - 1])
        prev, curr = curr, prev  # swap
    return prev[n]


def _lexical_lcs_length(gt: str, cand: str) -> int:
    """Return token-level Longest Common Subsequence length (integer)."""
    tokens_gt = _tokenise(gt)
    tokens_cand = _tokenise(cand)
    return _lcs_length(tokens_gt, tokens_cand)


def _lexical_lcs_ratio(gt: str, cand: str, normalize_by: str = "reference") -> float:
    """
    Return length-normalized LCS ratio between *gt* and *cand* in [0, 1].
    
    Args:
        gt: Ground truth text
        cand: Candidate text
        normalize_by: Normalization method (default: "reference")
            - "reference": Normalize by reference/ground truth length (default)
            - "candidate": Normalize by candidate length
    
    Returns:
        LCS ratio between 0 and 1
    """
    tokens_gt = _tokenise(gt)
    tokens_cand = _tokenise(cand)
    
    if not tokens_gt or not tokens_cand:
        return 0.0
    
    lcs_len = _lcs_length(tokens_gt, tokens_cand)
    
    if normalize_by == "reference":
        return lcs_len / len(tokens_gt)
    elif normalize_by == "candidate":
        return lcs_len / len(tokens_cand)
    else:
        raise ValueError(f"Invalid normalize_by value: {normalize_by}")


def _lexical_ngram_coverage(gt: str, cand: str, min_ngram: int = 3, normalize_by: str = "candidate") -> float:
    """
    N-gram coverage between *cand* and *gt* in [0, 1].
    
    Args:
        gt: Ground truth text
        cand: Candidate text
        min_ngram: Minimum n-gram size (default: 3)
        normalize_by: Normalization method (default: "candidate")
            - "candidate": Normalize by candidate length (default)
            - "reference": Normalize by reference/ground truth length
    
    Returns:
        Coverage score between 0 and 1
    """
    if _HAS_NGRAM_COVERAGE:
        return compute_ngram_coverage(cand, gt, min_ngram, normalize_by, tokenizer=_tokenise)
    else:
        # Fallback to 0 if module not available
        return 0.0


# Embedding similarity is computed using _embedding_cosine_similarity above


def _bleurt_similarity(gt: str, candidates: List[str], **bleurt_kwargs) -> List[float]:
    """Compute BLEURT scores between ground truth and candidates.
    
    Parameters
    ----------
    gt : str
        Ground truth text
    candidates : List[str]
        List of candidate texts
    **bleurt_kwargs
        Additional arguments for BLEURT (length_penalty, length_threshold, etc.)
    
    Returns
    -------
    List[float]
        BLEURT scores for each candidate
    """
    if not _HAS_BLEURT:
        # Return zeros if BLEURT is not available
        return [0.0] * len(candidates)
    
    scores = []
    for cand in candidates:
        try:
            score = compute_bleurt_score(
                data_source="bleurt_match_custom",
                solution_str=cand,
                ground_truth=gt,
                extra_info=bleurt_kwargs
            )
            scores.append(float(score))
        except Exception as e:
            # If BLEURT computation fails, log once and use 0.0
            if not hasattr(_bleurt_similarity, "_logged_error"):
                import logging
                logging.warning(f"BLEURT computation failed: {e}")
                _bleurt_similarity._logged_error = True
            scores.append(0.0)
    
    return scores


def _llm_judge_similarity(
    problem: str, 
    gt: str, 
    candidates: List[str], 
    **llm_judge_kwargs
) -> List[float]:
    """Compute LLM judge scores between ground truth and candidates.
    
    Supports both local and remote LLM judge evaluation based on configuration.
    
    Parameters
    ----------
    problem : str
        The problem statement
    gt : str
        Ground truth text
    candidates : List[str]
        List of candidate texts
    **llm_judge_kwargs
        Additional arguments for LLM judge (model_name, temperature, server_url, etc.)
    
    Returns
    -------
    List[float]
        LLM judge scores for each candidate
    """
    # Check if we should use remote or local LLM judge
    use_remote = llm_judge_kwargs.get("use_remote", False)
    server_url = llm_judge_kwargs.get("server_url")
    
    # Auto-detect remote if server_url is provided
    if server_url and not use_remote:
        use_remote = True
        llm_judge_kwargs["use_remote"] = True
    
    if use_remote and _HAS_REMOTE_LLM_JUDGE:
        try:
            return _remote_llm_judge_similarity(problem, gt, candidates, **llm_judge_kwargs)
        except Exception as e:
            import logging
            logging.error(f"Remote LLM judge failed: {e}")
            # Don't fallback - let the calling code handle the failure
            raise RuntimeError(f"Remote LLM judge evaluation failed: {e}")
    elif not use_remote and _HAS_LOCAL_LLM_JUDGE:
        return _local_llm_judge_similarity(problem, gt, candidates, **llm_judge_kwargs)
    else:
        # Fallback based on availability
        if _HAS_REMOTE_LLM_JUDGE:
            try:
                return _remote_llm_judge_similarity(problem, gt, candidates, **llm_judge_kwargs)
            except Exception as e:
                import logging
                logging.error(f"Remote LLM judge failed: {e}")
                raise RuntimeError(f"Remote LLM judge evaluation failed: {e}")
        elif _HAS_LOCAL_LLM_JUDGE:
            return _local_llm_judge_similarity(problem, gt, candidates, **llm_judge_kwargs)
        else:
            # No LLM judge available
            raise RuntimeError("No LLM judge implementation available (neither local nor remote)")


def _local_llm_judge_similarity(
    problem: str, 
    gt: str, 
    candidates: List[str], 
    **llm_judge_kwargs
) -> List[float]:
    """Compute local LLM judge scores between ground truth and candidates."""
    if not _HAS_LOCAL_LLM_JUDGE:
        return [0.0] * len(candidates)
    
    try:
        # Extract configuration from kwargs
        model_name = llm_judge_kwargs.get("model_name", DEFAULT_LLM_JUDGE_MODEL)
        prompt_template = llm_judge_kwargs.get("prompt_template", DEFAULT_LLM_JUDGE_PROMPT)
        prompt_template_name = llm_judge_kwargs.get("prompt_template_name")
        enable_thinking = llm_judge_kwargs.get("enable_thinking", DEFAULT_LLM_JUDGE_THINKING)
        batch_size = llm_judge_kwargs.get("batch_size", DEFAULT_LLM_JUDGE_BATCH_SIZE)
        temperature = llm_judge_kwargs.get("temperature", 0.7)
        top_p = llm_judge_kwargs.get("top_p", 0.8)
        max_new_tokens = llm_judge_kwargs.get("max_new_tokens", 512)
        
        # Use batch scoring for efficiency
        scores_per_problem, _ = compute_llm_judge_scores_batch_local(
            problems=[problem],
            ground_truths=[gt],
            candidates_list=[candidates],
            model_name=model_name,
            prompt_template=prompt_template,
            prompt_template_name=prompt_template_name,
            enable_thinking=enable_thinking,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        
        if scores_per_problem and scores_per_problem[0]:
            return scores_per_problem[0]
        else:
            return [0.0] * len(candidates)
            
    except Exception as e:
        if not hasattr(_local_llm_judge_similarity, "_logged_error"):
            import logging
            logging.warning(f"Local LLM judge computation failed: {e}")
            _local_llm_judge_similarity._logged_error = True
        return [0.0] * len(candidates)


def _remote_llm_judge_similarity(
    problem: str, 
    gt: str, 
    candidates: List[str], 
    **llm_judge_kwargs
) -> List[float]:
    """Compute remote LLM judge scores between ground truth and candidates."""
    if not _HAS_REMOTE_LLM_JUDGE:
        return [0.0] * len(candidates)
    
    try:
        # Extract configuration from kwargs
        model_name = llm_judge_kwargs.get("model_name", DEFAULT_LLM_JUDGE_MODEL)
        prompt_template = llm_judge_kwargs.get("prompt_template", DEFAULT_LLM_JUDGE_PROMPT)
        prompt_template_name = llm_judge_kwargs.get("prompt_template_name")
        enable_thinking = llm_judge_kwargs.get("enable_thinking", DEFAULT_LLM_JUDGE_THINKING)
        batch_size = llm_judge_kwargs.get("batch_size", DEFAULT_LLM_JUDGE_BATCH_SIZE)
        temperature = llm_judge_kwargs.get("temperature", 0.7)
        top_p = llm_judge_kwargs.get("top_p", 0.8)
        max_new_tokens = llm_judge_kwargs.get("max_new_tokens", 512)
        server_url = llm_judge_kwargs.get("server_url")
        api_key = llm_judge_kwargs.get("api_key")
        timeout = llm_judge_kwargs.get("timeout", 600.0)  # Increased to 5 minutes for long prompts
        
        # Use batch scoring for efficiency
        scores_per_problem, _ = compute_llm_judge_scores_batch_remote(
            problems=[problem],
            ground_truths=[gt],
            candidates_list=[candidates],
            model_name=model_name,
            prompt_template=prompt_template,
            prompt_template_name=prompt_template_name,
            enable_thinking=enable_thinking,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            server_url=server_url,
            api_key=api_key,
            timeout=timeout
        )
        
        if scores_per_problem is None:
            # Remote LLM judge failed - raise exception instead of fallback
            raise RuntimeError("Remote LLM judge evaluation failed - server unavailable or returned no scores")
        
        if scores_per_problem and scores_per_problem[0]:
            return scores_per_problem[0]
        else:
            # Should not happen if server is working, but handle gracefully
            import logging
            logging.warning("Remote LLM judge returned empty scores")
            return [0.0] * len(candidates)
            
    except Exception as e:
        if not hasattr(_remote_llm_judge_similarity, "_logged_error"):
            import logging
            logging.error(f"Remote LLM judge computation failed: {e}")
            _remote_llm_judge_similarity._logged_error = True
        # Re-raise the exception instead of returning fallback values
        raise


def _batch_llm_judge_similarity_cross_problems(
    problems_list: List[str],
    ground_truths_list: List[str], 
    candidates_lists: List[List[str]], 
    **llm_judge_kwargs
) -> List[List[float]]:
    """
    Optimized cross-problem batching for LLM judge evaluation.
    
    This function batches all problem-candidate pairs across the entire dataset
    for maximum throughput, similar to the VERL optimization.
    
    Parameters
    ----------
    problems_list : List[str]
        List of problem statements
    ground_truths_list : List[str]
        List of ground truth texts
    candidates_lists : List[List[str]]
        List of candidate lists (one per problem)
    **llm_judge_kwargs
        Configuration for LLM judge evaluation
        
    Returns
    -------
    List[List[float]]
        List of score lists, one per problem
    """
    use_remote = llm_judge_kwargs.get("use_remote", False)
    server_url = llm_judge_kwargs.get("server_url")
    
    # Calculate total number of pairs for progress tracking
    total_pairs = sum(len(candidates) for candidates in candidates_lists)
    
    # Auto-detect remote if server_url is provided
    if server_url and not use_remote:
        use_remote = True
        llm_judge_kwargs["use_remote"] = True
    
    if use_remote and _HAS_REMOTE_LLM_JUDGE:
        # Use optimized remote batching
        try:
            print(f"🚀 Processing {len(problems_list)} problems ({total_pairs} pairs) with remote LLM judge...")
            with tqdm(total=total_pairs, desc="LLM Judge (Remote Batch)", unit="pairs", leave=False) as pbar:
                scores_per_problem, _ = compute_llm_judge_scores_batch_remote(
                    problems=problems_list,
                    ground_truths=ground_truths_list,
                    candidates_list=candidates_lists,
                    **llm_judge_kwargs
                )
                pbar.update(total_pairs)  # Complete the progress bar
            
            if scores_per_problem is None:
                raise RuntimeError("Remote LLM judge evaluation failed - server unavailable")
            
            return scores_per_problem
            
        except Exception as e:
            import logging
            logging.error(f"Batch remote LLM judge failed: {e}")
            raise RuntimeError(f"Remote LLM judge evaluation failed: {e}")
    
    elif not use_remote and _HAS_LOCAL_LLM_JUDGE:
        # Use local batching
        try:
            print(f"🚀 Processing {len(problems_list)} problems ({total_pairs} pairs) with local LLM judge...")
            with tqdm(total=total_pairs, desc="LLM Judge (Local Batch)", unit="pairs", leave=False) as pbar:
                scores_per_problem, _ = compute_llm_judge_scores_batch_local(
                    problems=problems_list,
                    ground_truths=ground_truths_list,
                    candidates_list=candidates_lists,
                    **llm_judge_kwargs
                )
                pbar.update(total_pairs)  # Complete the progress bar
            
            if scores_per_problem is None:
                raise RuntimeError("Local LLM judge evaluation failed")
            
            return scores_per_problem
            
        except Exception as e:
            import logging
            logging.error(f"Batch local LLM judge failed: {e}")
            raise RuntimeError(f"Local LLM judge evaluation failed: {e}")
    
    else:
        # Fallback to sequential processing (original behavior)
        import logging
        logging.warning("No batch LLM judge available, falling back to sequential processing")
        
        results = []
        with tqdm(total=len(problems_list), desc="LLM Judge (Sequential)", unit="problems", leave=False) as pbar:
            for problem, gt, candidates in zip(problems_list, ground_truths_list, candidates_lists):
                try:
                    scores = _llm_judge_similarity(problem, gt, candidates, **llm_judge_kwargs)
                    results.append(scores)
                except Exception:
                    results.append([0.0] * len(candidates))
                pbar.update(1)
        
        return results


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------

def evaluate_pair(
    ground_truth: str,
    candidates: List[str],
    *,
    max_tokens_lexical: int | None = None,
    evaluate_math: bool = False,
    evaluate_bleurt: bool = False,
    bleurt_kwargs: Dict | None = None,
    evaluate_local_llm_judge: bool = False,
    llm_judge_kwargs: Dict | None = None,
    problem_text: str | None = None,
    embedding_model: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    budget_forcing: str | None = None,
) -> Tuple[List[float], List[float], List[float], int, int, int, List[float] | None, List[float] | None, int | None, List[float] | None, int | None, List[float], int, List[float], int, List[float], int, List[float], int, List[float], int, List[float], int]:
    """Evaluate a single \*ground_truth\* against its \*candidates\* list.

    Returns lexical, embedding, and optionally BLEURT and local LLM judge score arrays 
    **and** the indices of the best candidate for each metric so that callers can 
    record those strings.
    
    Parameters
    ----------
    ground_truth : str
        The reference/ground truth text
    candidates : List[str]
        List of candidate texts to evaluate
    max_tokens_lexical : int | None
        Maximum tokens for lexical evaluation
    evaluate_math : bool
        Whether to evaluate math correctness
    evaluate_bleurt : bool
        Whether to evaluate using BLEURT
    bleurt_kwargs : Dict | None
        Configuration for BLEURT evaluation
    evaluate_local_llm_judge : bool
        Whether to evaluate using local LLM judge
    llm_judge_kwargs : Dict | None
        Configuration for local LLM judge evaluation
    problem_text : str | None
        The problem statement (required for LLM judge)
    embedding_model : str | None
        The embedding model to use. Can be "fasttext", "qwen3", "qwen3-0.6B", 
        "qwen3-4B", or "qwen3-8B". If None, uses _DEFAULT_EMBEDDING_MODEL.
    num_workers : int
        Number of parallel workers for concurrent processing (default: 16).
        Set to 1 to disable parallelization.
    budget_forcing : str | None
        Budget forcing mode ('tokenizer' or 'whitespace'). If provided, truncates 
        candidates to match ground truth token count before evaluation.
    
    Returns
    -------
    Tuple containing:
        - lexical_jaccard_scores: List[float]
        - lcs_lengths: List[float]  
        - emb_scores: List[float]
        - best_jaccard_idx: int
        - best_lcs_idx: int
        - best_emb_idx: int
        - math_scores: List[float] | None
        - bleurt_scores: List[float] | None
        - best_bleurt_idx: int | None
        - llm_judge_scores: List[float] | None
        - best_llm_judge_idx: int | None
        - lexical_ngram_coverage_scores: List[float] (normalized by candidate)
        - best_ngram_coverage_idx: int
        - lexical_ngram_coverage_ref_scores: List[float] (normalized by reference)
        - best_ngram_coverage_ref_idx: int
        - lexical_lcs_ratio_scores: List[float] (normalized by reference)
        - best_lcs_ratio_idx: int
        - lexical_lcs_ratio_cand_scores: List[float] (normalized by candidate)
        - best_lcs_ratio_cand_idx: int
        - lexical_token_overlap_ref_scores: List[float] (token overlap normalized by reference)
        - best_token_overlap_ref_idx: int
        - lexical_token_overlap_cand_scores: List[float] (token overlap normalized by candidate)
        - best_token_overlap_cand_idx: int
    """

    # -------- Budget forcing truncation ----------------------
    if budget_forcing:
        # Count ground truth tokens
        if budget_forcing == "tokenizer" and _DEFAULT_TOKENIZER is not None:
            gt_tokens = _DEFAULT_TOKENIZER.tokenize(ground_truth)
            gt_budget = len(gt_tokens)
        else:
            gt_budget = len(ground_truth.split())
        
        # Truncate all candidates to match ground truth budget
        candidates = [_truncate_to_budget(cand, gt_budget, budget_forcing) for cand in candidates]

    # -------- Lexical scores with concurrent processing ----------------------
    # Use parallel processing for lexical metrics computation
    (
        lexical_jaccard_scores,
        lcs_lengths,
        lexical_lcs_ratio_scores,
        lexical_lcs_ratio_cand_scores,
        lexical_token_overlap_ref_scores,
        lexical_token_overlap_cand_scores,
    ) = _compute_lexical_metrics_parallel(ground_truth, candidates, num_workers)
    
    # N-gram coverage with batched processing (much faster!)
    # Tokenize ground truth once and reuse for all candidates
    if _HAS_NGRAM_COVERAGE and compute_ngram_coverage_batch is not None:
        # Batched version: tokenize reference once, build n-gram set once
        lexical_ngram_coverage_scores = compute_ngram_coverage_batch(
            candidates, ground_truth, normalize_by="candidate", tokenizer=_tokenise
        )
        lexical_ngram_coverage_ref_scores = compute_ngram_coverage_batch(
            candidates, ground_truth, normalize_by="reference", tokenizer=_tokenise
        )
    else:
        # Fallback to sequential (slower)
        lexical_ngram_coverage_scores = [
            _lexical_ngram_coverage(ground_truth, cand, normalize_by="candidate") for cand in candidates
        ]
        lexical_ngram_coverage_ref_scores = [
            _lexical_ngram_coverage(ground_truth, cand, normalize_by="reference") for cand in candidates
        ]

    # -------- Embedding scores ---------------------------------------------
    model_to_use = embedding_model or _DEFAULT_EMBEDDING_MODEL
    
    if model_to_use == "fasttext":
        if _FASTTEXT_MODEL is not None:
            # Use efficient batched computation for FastText
            texts = [ground_truth] + candidates
            embeddings = get_sentence_embeddings_fasttext(texts)
            
            # Split embeddings
            gt_emb = embeddings[0]
            cand_embs = embeddings[1:]
            
            # Compute similarities
            emb_scores = _embedding_cosine_similarity(gt_emb, cand_embs).tolist()
        else:
            # Fallback to verl's embedding module for FastText
            emb_scores = []
            for cand in candidates:
                try:
                    score = compute_embedding_score(
                        data_source="reconstruction_eval",
                        solution_str=cand,
                        ground_truth=ground_truth,
                        extra_info={"metric": "fasttext"}
                    )
                    score = float(score)
                    # Replace NaN or inf with 0.0
                    if not np.isfinite(score):
                        score = 0.0
                    emb_scores.append(score)
                except Exception:
                    # If computation fails, use 0.0 as fallback
                    emb_scores.append(0.0)
    elif model_to_use.startswith("qwen3"):
        # Use efficient batched computation for Qwen3 models
        # Parse model size
        if model_to_use == "qwen3":
            model_size = "0.6B"
        elif model_to_use in ["qwen3-0.6B", "qwen3-4B", "qwen3-8B"]:
            model_size = model_to_use.split("-")[1]
        else:
            raise ValueError(f"Invalid embedding model: {model_to_use}")
        
        # Load model (cached after first load)
        model, embed_dim = _load_qwen3_model(model_size)
        
        # Compute embeddings for all texts at once
        texts = [ground_truth] + candidates
        embeddings = get_sentence_embeddings_qwen3(texts, model)
        
        # Split embeddings
        gt_emb = embeddings[0]
        cand_embs = embeddings[1:]
        
        # Compute similarities
        emb_scores = _embedding_cosine_similarity(gt_emb, cand_embs).tolist()
    else:
        raise ValueError(f"Unsupported embedding model: {model_to_use}")

    # Final safety check: replace any NaN or inf values in emb_scores with 0.0
    emb_scores = [0.0 if not np.isfinite(score) else float(score) for score in emb_scores]

    best_jaccard_idx = int(np.argmax(lexical_jaccard_scores))
    best_lcs_idx = int(np.argmax(lcs_lengths))
    best_emb_idx = int(np.argmax(emb_scores))
    best_ngram_coverage_idx = int(np.argmax(lexical_ngram_coverage_scores))
    best_ngram_coverage_ref_idx = int(np.argmax(lexical_ngram_coverage_ref_scores))
    best_lcs_ratio_idx = int(np.argmax(lexical_lcs_ratio_scores))
    best_lcs_ratio_cand_idx = int(np.argmax(lexical_lcs_ratio_cand_scores))
    best_token_overlap_ref_idx = int(np.argmax(lexical_token_overlap_ref_scores))
    best_token_overlap_cand_idx = int(np.argmax(lexical_token_overlap_cand_scores))

    # -------- BLEURT scores ---------------------------------------------
    bleurt_scores = None
    best_bleurt_idx = None
    if evaluate_bleurt:
        bleurt_config = bleurt_kwargs or {}
        bleurt_scores = _bleurt_similarity(ground_truth, candidates, **bleurt_config)
        best_bleurt_idx = int(np.argmax(bleurt_scores)) if bleurt_scores else None

    # -------- LLM judge scores (local or remote) ----------------------
    llm_judge_scores = None
    best_llm_judge_idx = None
    if evaluate_local_llm_judge:
        if problem_text is None:
            import warnings
            warnings.warn("problem_text is required for LLM judge but was not provided")
            llm_judge_scores = [0.0] * len(candidates)
        else:
            llm_config = llm_judge_kwargs or {}
            try:
                llm_judge_scores = _llm_judge_similarity(problem_text, ground_truth, candidates, **llm_config)
            except Exception as e:
                import logging
                logging.error(f"LLM judge evaluation failed: {e}")
                # Set scores to None to indicate failure rather than using fallback
                llm_judge_scores = None
                warnings.warn(f"LLM judge evaluation failed: {e}")
        best_llm_judge_idx = int(np.argmax(llm_judge_scores)) if llm_judge_scores else None

    # -------- Math scores with concurrent processing ----------------------
    math_scores = None
    if evaluate_math:
        gt_simple = _extract_math_answer(ground_truth) or ground_truth
        math_scores = _compute_math_scores_parallel(candidates, gt_simple, num_workers)

    return (
        lexical_jaccard_scores,
        lcs_lengths,
        emb_scores,
        best_jaccard_idx,
        best_lcs_idx,
        best_emb_idx,
        math_scores,
        bleurt_scores,
        best_bleurt_idx,
        llm_judge_scores,
        best_llm_judge_idx,
        lexical_ngram_coverage_scores,
        best_ngram_coverage_idx,
        lexical_ngram_coverage_ref_scores,
        best_ngram_coverage_ref_idx,
        lexical_lcs_ratio_scores,
        best_lcs_ratio_idx,
        lexical_lcs_ratio_cand_scores,
        best_lcs_ratio_cand_idx,
        lexical_token_overlap_ref_scores,
        best_token_overlap_ref_idx,
        lexical_token_overlap_cand_scores,
        best_token_overlap_cand_idx,
    )


def evaluate_dataset(
    ground_truth_texts: List[str],
    candidates_list: List[List[str]],
    *,
    max_tokens_lexical: int | None = None,
    verbose: bool = True,
    evaluate_math: bool = False,
    evaluate_bleurt: bool = False,
    bleurt_kwargs: Dict | None = None,
    evaluate_local_llm_judge: bool = False,
    llm_judge_kwargs: Dict | None = None,
    problems_list: List[str] | None = None,
    embedding_model: str | None = None,
    enable_llm_judge_batching: bool = True,
    mia_weights: List[float] | None = None,
    mia_weight_tag: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    mia_weights_higher_is_member: bool = False,
    budget_forcing: str | None = None,
) -> Dict:
    """Evaluate an entire dataset (N ground truths, variable M candidates).

    The function records, for every ground truth, the *average* and *best* score
    across its candidates for all metrics, and aggregates a dataset-level
    summary (mean of averages, best of bests).
    
    Parameters
    ----------
    ground_truth_texts : List[str]
        List of reference/ground truth texts
    candidates_list : List[List[str]]
        List of candidate lists (one per ground truth)
    max_tokens_lexical : int | None
        Maximum tokens for lexical evaluation
    verbose : bool
        Whether to show progress bar
    evaluate_math : bool
        Whether to evaluate math correctness
    evaluate_bleurt : bool
        Whether to evaluate using BLEURT
    bleurt_kwargs : Dict | None
        Configuration for BLEURT evaluation
    evaluate_local_llm_judge : bool
        Whether to evaluate using local LLM judge
    llm_judge_kwargs : Dict | None
        Configuration for local LLM judge evaluation
    problems_list : List[str] | None
        List of problem statements (required for LLM judge, one per ground truth)
    embedding_model : str | None
        The embedding model to use. Can be "fasttext", "qwen3", "qwen3-0.6B", 
        "qwen3-4B", or "qwen3-8B". If None, uses _DEFAULT_EMBEDDING_MODEL.
    enable_llm_judge_batching : bool
        Whether to use optimized cross-problem batching for LLM judge evaluation.
        When True, all LLM judge requests are batched across the entire dataset
        for maximum throughput (default: True).
    mia_weights : List[float] | None
        Optional MIA weights for weighted metric computation. If provided, must
        have the same length as ground_truth_texts.
    mia_weight_tag : str | None
        Tag identifying the type of MIA weights (e.g., "min_k++", "loss").
        Required if mia_weights is provided.
    num_workers : int
        Number of parallel workers for concurrent processing (default: 16).
        Set to 1 to disable parallelization.
    mia_weights_higher_is_member : bool
        If True, higher MIA weight values indicate membership (default: False).
        By default, MIA weights follow the convention that LOWER values indicate
        membership (e.g., min-k++, loss-based metrics). When False (default),
        weights are inverted before applying to reconstruction metrics.
    budget_forcing : str | None
        Budget forcing mode ('tokenizer' or 'whitespace'). If provided, truncates 
        candidates to match ground truth token count before evaluation.
    
    Returns
    -------
    Dict
        Dictionary containing 'per_example' results and 'dataset_summary'.
        If mia_weights is provided, also includes weighted metrics.
    """
    assert len(ground_truth_texts) == len(
        candidates_list
    ), "Ground-truth and candidates list must have the same length."
    
    # Validate MIA weights
    if mia_weights is not None:
        if len(mia_weights) != len(ground_truth_texts):
            raise ValueError(
                f"MIA weights length ({len(mia_weights)}) must match ground truth length ({len(ground_truth_texts)})"
            )
        if mia_weight_tag is None:
            raise ValueError("mia_weight_tag is required when mia_weights is provided")
        
        # Invert MIA weights if needed (default: lower MIA score = more likely member)
        # Since reconstruction metrics use higher-is-better, we need to invert
        # Note: MIA weights are already normalized to [0, 1] during preprocessing
        if not mia_weights_higher_is_member:
            if verbose:
                print(f"🔢 Using MIA weights ({mia_weight_tag}) for weighted metric computation")
                print(f"   🔄 Inverting weights (lower-is-member → higher-weight for reconstruction)")
                print(f"   📊 Original range: [{min(mia_weights):.6f}, {max(mia_weights):.6f}]")
            
            # Simple inversion: 1 - weight (assumes weights are already in [0, 1])
            inverted = [1.0 - w for w in mia_weights]
            mia_weights = inverted
            
            if verbose:
                print(f"   📊 Inverted range: [{min(inverted):.6f}, {max(inverted):.6f}]")
        else:
            if verbose:
                print(f"🔢 Using MIA weights ({mia_weight_tag}) for weighted metric computation")
                print(f"   ℹ️  Using weights as-is (higher-is-member mode)")
                print(f"   📊 Weight range: [{min(mia_weights):.6f}, {max(mia_weights):.6f}]")

    per_example_results = []

    # For dataset-level aggregation
    jaccard_avg_values: List[float] = []
    jaccard_best_values: List[float] = []
    lcs_len_avg_values: List[float] = []
    lcs_len_best_values: List[float] = []
    lcs_ratio_avg_values: List[float] = []
    lcs_ratio_best_values: List[float] = []
    lcs_ratio_cand_avg_values: List[float] = []
    lcs_ratio_cand_best_values: List[float] = []
    token_overlap_ref_avg_values: List[float] = []
    token_overlap_ref_best_values: List[float] = []
    token_overlap_cand_avg_values: List[float] = []
    token_overlap_cand_best_values: List[float] = []
    emb_avg_values: List[float] = []
    emb_best_values: List[float] = []
    bleurt_avg_values: List[float] = []
    bleurt_best_values: List[float] = []
    llm_judge_avg_values: List[float] = []
    llm_judge_best_values: List[float] = []
    ngram_coverage_avg_values: List[float] = []
    ngram_coverage_best_values: List[float] = []
    ngram_coverage_ref_avg_values: List[float] = []
    ngram_coverage_ref_best_values: List[float] = []
    math_avg_at_k_values: List[float] = []
    math_pass_at_k_values: List[float] = []
    best_jaccard_accuracies: List[float] = []
    best_emb_accuracies: List[float] = []
    best_bleurt_accuracies: List[float] = []
    best_llm_judge_accuracies: List[float] = []

    # Optimized LLM judge cross-problem batching
    batched_llm_judge_scores = None
    if evaluate_local_llm_judge and enable_llm_judge_batching and problems_list:
        if verbose:
            print("🚀 Using optimized cross-problem batching for LLM judge evaluation...")
        
        # Pre-compute all LLM judge scores in batches for maximum throughput
        try:
            import time
            batch_start_time = time.time()
            
            valid_problems = []
            valid_ground_truths = []
            valid_candidates_lists = []
            valid_indices = []
            
            # Collect valid problems that have problem text
            for idx, (gt, cands) in enumerate(zip(ground_truth_texts, candidates_list)):
                if idx < len(problems_list) and problems_list[idx]:
                    valid_problems.append(problems_list[idx])
                    valid_ground_truths.append(gt)
                    valid_candidates_lists.append(cands)
                    valid_indices.append(idx)
            
            if valid_problems:
                total_pairs = sum(len(cands) for cands in valid_candidates_lists)
                llm_config = llm_judge_kwargs or {}
                
                batch_scores = _batch_llm_judge_similarity_cross_problems(
                    valid_problems, valid_ground_truths, valid_candidates_lists, **llm_config
                )
                
                # Create mapping from original index to batch scores
                batched_llm_judge_scores = {}
                for batch_idx, original_idx in enumerate(valid_indices):
                    if batch_idx < len(batch_scores):
                        batched_llm_judge_scores[original_idx] = batch_scores[batch_idx]
                
                batch_duration = time.time() - batch_start_time
                throughput = total_pairs / batch_duration if batch_duration > 0 else 0
                
                if verbose:
                    print(f"✅ Batch LLM judge completed:")
                    print(f"   📊 {len(valid_problems)} problems, {total_pairs} pairs")
                    print(f"   ⏱️  Duration: {batch_duration:.1f}s")
                    print(f"   🚀 Throughput: {throughput:.1f} pairs/sec")
                    print(f"   💡 Estimated sequential time would be: {total_pairs * 2.0:.1f}s (assuming 2s/pair)")
                    print(f"   📈 Speedup: ~{(total_pairs * 2.0 / batch_duration):.1f}x faster")
            
        except Exception as e:
            import logging
            logging.warning(f"Batch LLM judge failed, falling back to sequential: {e}")
            batched_llm_judge_scores = None

    iterator = enumerate(zip(ground_truth_texts, candidates_list))
    if verbose:
        # Customize progress bar description based on LLM judge batching status
        desc = "Evaluating"
        if evaluate_local_llm_judge:
            if batched_llm_judge_scores is not None:
                desc = "Evaluating (LLM Judge: ✅ Batched)"
            elif enable_llm_judge_batching:
                desc = "Evaluating (LLM Judge: 🔄 Sequential)"
            else:
                desc = "Evaluating (LLM Judge: ⚠️ Disabled Batching)"
        iterator = tqdm(iterator, total=len(ground_truth_texts), desc=desc)

    for idx, (gt, cands) in iterator:
        if not cands:
            raise ValueError(f"No candidates provided for example index {idx}.")

        # Check if we have pre-computed LLM judge scores for this problem
        use_batched_llm_scores = (batched_llm_judge_scores is not None and 
                                 idx in batched_llm_judge_scores)
        
        # Get problem text for LLM judge if available and not using batched scores
        problem_text = None
        if evaluate_local_llm_judge and problems_list and not use_batched_llm_scores:
            if idx < len(problems_list):
                problem_text = problems_list[idx]
            else:
                import warnings
                warnings.warn(f"Problem text not available for example {idx}, skipping LLM judge evaluation")

        (
            jaccard_scores,
            lcs_lengths,
            emb_scores,
            best_j_idx,
            best_lcs_idx,
            best_emb_idx,
            math_scores,
            bleurt_scores,
            best_bleurt_idx,
            llm_judge_scores,
            best_llm_judge_idx,
            ngram_coverage_scores,
            best_ngram_coverage_idx,
            ngram_coverage_ref_scores,
            best_ngram_coverage_ref_idx,
            lcs_ratio_scores,
            best_lcs_ratio_idx,
            lcs_ratio_cand_scores,
            best_lcs_ratio_cand_idx,
            token_overlap_ref_scores,
            best_token_overlap_ref_idx,
            token_overlap_cand_scores,
            best_token_overlap_cand_idx,
        ) = evaluate_pair(
            gt, 
            cands, 
            max_tokens_lexical=max_tokens_lexical, 
            evaluate_math=evaluate_math,
            evaluate_bleurt=evaluate_bleurt,
            bleurt_kwargs=bleurt_kwargs,
            evaluate_local_llm_judge=evaluate_local_llm_judge and not use_batched_llm_scores,
            llm_judge_kwargs=llm_judge_kwargs,
            problem_text=problem_text,
            embedding_model=embedding_model,
            num_workers=num_workers,
            budget_forcing=budget_forcing
        )
        
        # Use pre-computed batched LLM judge scores if available
        if use_batched_llm_scores:
            llm_judge_scores = batched_llm_judge_scores[idx]
            best_llm_judge_idx = int(np.argmax(llm_judge_scores)) if llm_judge_scores else None

        jaccard_avg = float(np.mean(jaccard_scores))
        jaccard_best = float(np.max(jaccard_scores))
        lcs_avg = float(np.mean(lcs_lengths))
        lcs_best = float(np.max(lcs_lengths))
        lcs_ratio_avg = float(np.mean(lcs_ratio_scores))
        lcs_ratio_best = float(np.max(lcs_ratio_scores))
        lcs_ratio_cand_avg = float(np.mean(lcs_ratio_cand_scores))
        lcs_ratio_cand_best = float(np.max(lcs_ratio_cand_scores))
        token_overlap_ref_avg = float(np.mean(token_overlap_ref_scores))
        token_overlap_ref_best = float(np.max(token_overlap_ref_scores))
        token_overlap_cand_avg = float(np.mean(token_overlap_cand_scores))
        token_overlap_cand_best = float(np.max(token_overlap_cand_scores))
        emb_avg = float(np.mean(emb_scores))
        emb_best = float(np.max(emb_scores))
        ngram_coverage_avg = float(np.mean(ngram_coverage_scores))
        ngram_coverage_best = float(np.max(ngram_coverage_scores))
        ngram_coverage_ref_avg = float(np.mean(ngram_coverage_ref_scores))
        ngram_coverage_ref_best = float(np.max(ngram_coverage_ref_scores))

        # BLEURT metrics
        bleurt_avg = 0.0
        bleurt_best = 0.0
        if evaluate_bleurt and bleurt_scores is not None:
            bleurt_avg = float(np.mean(bleurt_scores))
            bleurt_best = float(np.max(bleurt_scores))

        # Local LLM judge metrics
        llm_judge_avg = 0.0
        llm_judge_best = 0.0
        if evaluate_local_llm_judge and llm_judge_scores is not None:
            llm_judge_avg = float(np.mean(llm_judge_scores))
            llm_judge_best = float(np.max(llm_judge_scores))

        example_result = {
            "index": idx,
            "ground_truth": gt,
            "best_candidate_lexical_jaccard": cands[best_j_idx],
            "best_candidate_lexical_lcs": cands[best_lcs_idx],
            "best_candidate_lexical_lcs_ratio": cands[best_lcs_ratio_idx],
            "best_candidate_lexical_lcs_ratio_cand": cands[best_lcs_ratio_cand_idx],
            "best_candidate_embedding": cands[best_emb_idx],
            "best_candidate_lexical_token_overlap_ref": cands[best_token_overlap_ref_idx],
            "best_candidate_lexical_token_overlap_cand": cands[best_token_overlap_cand_idx],
            "best_candidate_lexical_ngram_coverage": cands[best_ngram_coverage_idx],
            "best_candidate_lexical_ngram_coverage_ref": cands[best_ngram_coverage_ref_idx],
            "lexical_jaccard_sim_avg": jaccard_avg,
            "lexical_jaccard_sim_best": jaccard_best,
            "lexical_lcs_len_avg": lcs_avg,
            "lexical_lcs_len_best": lcs_best,
            "lexical_lcs_ratio_avg": lcs_ratio_avg,
            "lexical_lcs_ratio_best": lcs_ratio_best,
            "lexical_lcs_ratio_cand_avg": lcs_ratio_cand_avg,
            "lexical_lcs_ratio_cand_best": lcs_ratio_cand_best,
            "lexical_token_overlap_ref_avg": token_overlap_ref_avg,
            "lexical_token_overlap_ref_best": token_overlap_ref_best,
            "lexical_token_overlap_cand_avg": token_overlap_cand_avg,
            "lexical_token_overlap_cand_best": token_overlap_cand_best,
            "embedding_cosine_sim_avg": emb_avg,
            "embedding_cosine_sim_best": emb_best,
            "lexical_ngram_coverage_avg": ngram_coverage_avg,
            "lexical_ngram_coverage_best": ngram_coverage_best,
            "lexical_ngram_coverage_ref_avg": ngram_coverage_ref_avg,
            "lexical_ngram_coverage_ref_best": ngram_coverage_ref_best,
        }

        # Add BLEURT results if enabled
        if evaluate_bleurt:
            example_result.update({
                "best_candidate_bleurt": cands[best_bleurt_idx] if best_bleurt_idx is not None else "",
                "bleurt_sim_avg": bleurt_avg,
                "bleurt_sim_best": bleurt_best,
            })

        # Add local LLM judge results if enabled
        if evaluate_local_llm_judge:
            example_result.update({
                "best_candidate_llm_judge": cands[best_llm_judge_idx] if best_llm_judge_idx is not None else "",
                "llm_judge_sim_avg": llm_judge_avg,
                "llm_judge_sim_best": llm_judge_best,
            })

        if evaluate_math and math_scores is not None:
            math_avg_at_k = float(np.mean(math_scores))
            math_pass_at_k = float(np.sum(math_scores) > 0)
            best_lex_acc = math_scores[best_j_idx]
            best_emb_acc = math_scores[best_emb_idx]
            best_bleurt_acc = math_scores[best_bleurt_idx] if best_bleurt_idx is not None else 0.0
            best_llm_judge_acc = math_scores[best_llm_judge_idx] if best_llm_judge_idx is not None else 0.0

            example_result.update({
                "math_avg_at_k": math_avg_at_k,
                "math_pass_at_k": math_pass_at_k,
                "best_candidate_lexical_accuracy": best_lex_acc,
                "best_candidate_embedding_accuracy": best_emb_acc,
            })

            if evaluate_bleurt:
                example_result["best_candidate_bleurt_accuracy"] = best_bleurt_acc
            
            if evaluate_local_llm_judge:
                example_result["best_candidate_llm_judge_accuracy"] = best_llm_judge_acc

            math_avg_at_k_values.append(math_avg_at_k)
            math_pass_at_k_values.append(math_pass_at_k)
            best_jaccard_accuracies.append(best_lex_acc)
            best_emb_accuracies.append(best_emb_acc)
            if evaluate_bleurt:
                best_bleurt_accuracies.append(best_bleurt_acc)
            if evaluate_local_llm_judge:
                best_llm_judge_accuracies.append(best_llm_judge_acc)

        per_example_results.append(example_result)

        jaccard_avg_values.append(jaccard_avg)
        jaccard_best_values.append(jaccard_best)
        lcs_len_avg_values.append(lcs_avg)
        lcs_len_best_values.append(lcs_best)
        lcs_ratio_avg_values.append(lcs_ratio_avg)
        lcs_ratio_best_values.append(lcs_ratio_best)
        lcs_ratio_cand_avg_values.append(lcs_ratio_cand_avg)
        lcs_ratio_cand_best_values.append(lcs_ratio_cand_best)
        token_overlap_ref_avg_values.append(token_overlap_ref_avg)
        token_overlap_ref_best_values.append(token_overlap_ref_best)
        token_overlap_cand_avg_values.append(token_overlap_cand_avg)
        token_overlap_cand_best_values.append(token_overlap_cand_best)
        emb_avg_values.append(emb_avg)
        emb_best_values.append(emb_best)
        ngram_coverage_avg_values.append(ngram_coverage_avg)
        ngram_coverage_best_values.append(ngram_coverage_best)
        ngram_coverage_ref_avg_values.append(ngram_coverage_ref_avg)
        ngram_coverage_ref_best_values.append(ngram_coverage_ref_best)
        
        if evaluate_bleurt:
            bleurt_avg_values.append(bleurt_avg)
            bleurt_best_values.append(bleurt_best)
        
        if evaluate_local_llm_judge:
            llm_judge_avg_values.append(llm_judge_avg)
            llm_judge_best_values.append(llm_judge_best)

    dataset_summary = {
        "lexical_jaccard_sim_avg": float(np.mean(jaccard_avg_values)) if jaccard_avg_values else 0.0,
        "lexical_lcs_len_avg": float(np.mean(lcs_len_avg_values)) if lcs_len_avg_values else 0.0,
        "lexical_lcs_ratio_avg": float(np.mean(lcs_ratio_avg_values)) if lcs_ratio_avg_values else 0.0,
        "lexical_lcs_ratio_cand_avg": float(np.mean(lcs_ratio_cand_avg_values)) if lcs_ratio_cand_avg_values else 0.0,
        "lexical_token_overlap_ref_avg": float(np.mean(token_overlap_ref_avg_values)) if token_overlap_ref_avg_values else 0.0,
        "lexical_token_overlap_cand_avg": float(np.mean(token_overlap_cand_avg_values)) if token_overlap_cand_avg_values else 0.0,
        "embedding_cosine_sim_avg": float(np.mean(emb_avg_values)) if emb_avg_values else 0.0,
        "lexical_ngram_coverage_avg": float(np.mean(ngram_coverage_avg_values)) if ngram_coverage_avg_values else 0.0,
        "lexical_ngram_coverage_ref_avg": float(np.mean(ngram_coverage_ref_avg_values)) if ngram_coverage_ref_avg_values else 0.0,
        "lexical_jaccard_sim_best_mean": float(np.mean(jaccard_best_values)) if jaccard_best_values else 0.0,
        "lexical_lcs_len_best_mean": float(np.mean(lcs_len_best_values)) if lcs_len_best_values else 0.0,
        "lexical_lcs_ratio_best_mean": float(np.mean(lcs_ratio_best_values)) if lcs_ratio_best_values else 0.0,
        "lexical_lcs_ratio_cand_best_mean": float(np.mean(lcs_ratio_cand_best_values)) if lcs_ratio_cand_best_values else 0.0,
        "lexical_token_overlap_ref_best_mean": float(np.mean(token_overlap_ref_best_values)) if token_overlap_ref_best_values else 0.0,
        "lexical_token_overlap_cand_best_mean": float(np.mean(token_overlap_cand_best_values)) if token_overlap_cand_best_values else 0.0,
        "embedding_cosine_sim_best_mean": float(np.mean(emb_best_values)) if emb_best_values else 0.0,
        "lexical_ngram_coverage_best_mean": float(np.mean(ngram_coverage_best_values)) if ngram_coverage_best_values else 0.0,
        "lexical_ngram_coverage_ref_best_mean": float(np.mean(ngram_coverage_ref_best_values)) if ngram_coverage_ref_best_values else 0.0,
        "lexical_jaccard_sim_best_max": float(np.max(jaccard_best_values)) if jaccard_best_values else 0.0,
        "lexical_jaccard_sim_best": float(np.max(jaccard_best_values)) if jaccard_best_values else 0.0,
        "lexical_lcs_len_best_max": float(np.max(lcs_len_best_values)) if lcs_len_best_values else 0.0,
        "lexical_lcs_len_best": float(np.max(lcs_len_best_values)) if lcs_len_best_values else 0.0,
        "lexical_lcs_ratio_best_max": float(np.max(lcs_ratio_best_values)) if lcs_ratio_best_values else 0.0,
        "lexical_lcs_ratio_best": float(np.max(lcs_ratio_best_values)) if lcs_ratio_best_values else 0.0,
        "lexical_lcs_ratio_cand_best_max": float(np.max(lcs_ratio_cand_best_values)) if lcs_ratio_cand_best_values else 0.0,
        "lexical_lcs_ratio_cand_best": float(np.max(lcs_ratio_cand_best_values)) if lcs_ratio_cand_best_values else 0.0,
        "lexical_token_overlap_ref_best_max": float(np.max(token_overlap_ref_best_values)) if token_overlap_ref_best_values else 0.0,
        "lexical_token_overlap_ref_best": float(np.max(token_overlap_ref_best_values)) if token_overlap_ref_best_values else 0.0,
        "lexical_token_overlap_cand_best_max": float(np.max(token_overlap_cand_best_values)) if token_overlap_cand_best_values else 0.0,
        "lexical_token_overlap_cand_best": float(np.max(token_overlap_cand_best_values)) if token_overlap_cand_best_values else 0.0,
        "embedding_cosine_sim_best_max": float(np.max(emb_best_values)) if emb_best_values else 0.0,
        "embedding_cosine_sim_best": float(np.max(emb_best_values)) if emb_best_values else 0.0,
        "lexical_ngram_coverage_best_max": float(np.max(ngram_coverage_best_values)) if ngram_coverage_best_values else 0.0,
        "lexical_ngram_coverage_best": float(np.max(ngram_coverage_best_values)) if ngram_coverage_best_values else 0.0,
        "lexical_ngram_coverage_ref_best_max": float(np.max(ngram_coverage_ref_best_values)) if ngram_coverage_ref_best_values else 0.0,
        "lexical_ngram_coverage_ref_best": float(np.max(ngram_coverage_ref_best_values)) if ngram_coverage_ref_best_values else 0.0,
    }

    # Add BLEURT metrics to dataset summary if enabled
    if evaluate_bleurt:
        dataset_summary.update({
            "bleurt_sim_avg": float(np.mean(bleurt_avg_values)) if bleurt_avg_values else 0.0,
            "bleurt_sim_best_mean": float(np.mean(bleurt_best_values)) if bleurt_best_values else 0.0,
            "bleurt_sim_best_max": float(np.max(bleurt_best_values)) if bleurt_best_values else 0.0,
            "bleurt_sim_best": float(np.max(bleurt_best_values)) if bleurt_best_values else 0.0,
        })

    # Add local LLM judge metrics to dataset summary if enabled
    if evaluate_local_llm_judge:
        dataset_summary.update({
            "llm_judge_sim_avg": float(np.mean(llm_judge_avg_values)) if llm_judge_avg_values else 0.0,
            "llm_judge_sim_best_mean": float(np.mean(llm_judge_best_values)) if llm_judge_best_values else 0.0,
            "llm_judge_sim_best_max": float(np.max(llm_judge_best_values)) if llm_judge_best_values else 0.0,
            "llm_judge_sim_best": float(np.max(llm_judge_best_values)) if llm_judge_best_values else 0.0,
        })

    if evaluate_math:
        dataset_summary.update({
            "math_avg_at_k_mean": float(np.mean(math_avg_at_k_values)),
            "math_pass_at_k_mean": float(np.mean(math_pass_at_k_values)),
            "best_candidate_lexical_accuracy_mean": float(np.mean(best_jaccard_accuracies)),
            "best_candidate_embedding_accuracy_mean": float(np.mean(best_emb_accuracies)),
        })
        
        if evaluate_bleurt and best_bleurt_accuracies:
            dataset_summary["best_candidate_bleurt_accuracy_mean"] = float(np.mean(best_bleurt_accuracies))
        
        if evaluate_local_llm_judge and best_llm_judge_accuracies:
            dataset_summary["best_candidate_llm_judge_accuracy_mean"] = float(np.mean(best_llm_judge_accuracies))
    
    # Compute weighted metrics if MIA weights are provided
    weighted_summary = {}
    if mia_weights is not None:
        # Convert weights to numpy array for computation
        weights = np.array(mia_weights)
        
        # Compute weighted averages for all metrics
        weighted_summary["mia_weight_tag"] = mia_weight_tag
        weighted_summary["weighted_lexical_jaccard_sim_avg"] = float(np.average(jaccard_avg_values, weights=weights)) if jaccard_avg_values else 0.0
        weighted_summary["weighted_lexical_lcs_len_avg"] = float(np.average(lcs_len_avg_values, weights=weights)) if lcs_len_avg_values else 0.0
        weighted_summary["weighted_lexical_lcs_ratio_avg"] = float(np.average(lcs_ratio_avg_values, weights=weights)) if lcs_ratio_avg_values else 0.0
        weighted_summary["weighted_lexical_lcs_ratio_cand_avg"] = float(np.average(lcs_ratio_cand_avg_values, weights=weights)) if lcs_ratio_cand_avg_values else 0.0
        weighted_summary["weighted_embedding_cosine_sim_avg"] = float(np.average(emb_avg_values, weights=weights)) if emb_avg_values else 0.0
        weighted_summary["weighted_lexical_ngram_coverage_avg"] = float(np.average(ngram_coverage_avg_values, weights=weights)) if ngram_coverage_avg_values else 0.0
        weighted_summary["weighted_lexical_ngram_coverage_ref_avg"] = float(np.average(ngram_coverage_ref_avg_values, weights=weights)) if ngram_coverage_ref_avg_values else 0.0
        weighted_summary["weighted_lexical_token_overlap_ref_avg"] = float(np.average(token_overlap_ref_avg_values, weights=weights)) if token_overlap_ref_avg_values else 0.0
        weighted_summary["weighted_lexical_token_overlap_cand_avg"] = float(np.average(token_overlap_cand_avg_values, weights=weights)) if token_overlap_cand_avg_values else 0.0
        
        weighted_summary["weighted_lexical_jaccard_sim_best_mean"] = float(np.average(jaccard_best_values, weights=weights)) if jaccard_best_values else 0.0
        weighted_summary["weighted_lexical_lcs_len_best_mean"] = float(np.average(lcs_len_best_values, weights=weights)) if lcs_len_best_values else 0.0
        weighted_summary["weighted_lexical_lcs_ratio_best_mean"] = float(np.average(lcs_ratio_best_values, weights=weights)) if lcs_ratio_best_values else 0.0
        weighted_summary["weighted_lexical_lcs_ratio_cand_best_mean"] = float(np.average(lcs_ratio_cand_best_values, weights=weights)) if lcs_ratio_cand_best_values else 0.0
        weighted_summary["weighted_lexical_token_overlap_ref_best_mean"] = float(np.average(token_overlap_ref_best_values, weights=weights)) if token_overlap_ref_best_values else 0.0
        weighted_summary["weighted_lexical_token_overlap_cand_best_mean"] = float(np.average(token_overlap_cand_best_values, weights=weights)) if token_overlap_cand_best_values else 0.0
        weighted_summary["weighted_embedding_cosine_sim_best_mean"] = float(np.average(emb_best_values, weights=weights)) if emb_best_values else 0.0
        weighted_summary["weighted_lexical_ngram_coverage_best_mean"] = float(np.average(ngram_coverage_best_values, weights=weights)) if ngram_coverage_best_values else 0.0
        weighted_summary["weighted_lexical_ngram_coverage_ref_best_mean"] = float(np.average(ngram_coverage_ref_best_values, weights=weights)) if ngram_coverage_ref_best_values else 0.0
        
        # Add BLEURT weighted metrics if enabled
        if evaluate_bleurt:
            weighted_summary["weighted_bleurt_sim_avg"] = float(np.average(bleurt_avg_values, weights=weights)) if bleurt_avg_values else 0.0
            weighted_summary["weighted_bleurt_sim_best_mean"] = float(np.average(bleurt_best_values, weights=weights)) if bleurt_best_values else 0.0
        
        # Add LLM judge weighted metrics if enabled
        if evaluate_local_llm_judge:
            weighted_summary["weighted_llm_judge_sim_avg"] = float(np.average(llm_judge_avg_values, weights=weights)) if llm_judge_avg_values else 0.0
            weighted_summary["weighted_llm_judge_sim_best_mean"] = float(np.average(llm_judge_best_values, weights=weights)) if llm_judge_best_values else 0.0
        
        # Add math weighted metrics if enabled
        if evaluate_math:
            weighted_summary["weighted_math_avg_at_k_mean"] = float(np.average(math_avg_at_k_values, weights=weights)) if math_avg_at_k_values else 0.0
            weighted_summary["weighted_math_pass_at_k_mean"] = float(np.average(math_pass_at_k_values, weights=weights)) if math_pass_at_k_values else 0.0
            weighted_summary["weighted_best_candidate_lexical_accuracy_mean"] = float(np.average(best_jaccard_accuracies, weights=weights)) if best_jaccard_accuracies else 0.0
            weighted_summary["weighted_best_candidate_embedding_accuracy_mean"] = float(np.average(best_emb_accuracies, weights=weights)) if best_emb_accuracies else 0.0
            
            if evaluate_bleurt and best_bleurt_accuracies:
                weighted_summary["weighted_best_candidate_bleurt_accuracy_mean"] = float(np.average(best_bleurt_accuracies, weights=weights))
            
            if evaluate_local_llm_judge and best_llm_judge_accuracies:
                weighted_summary["weighted_best_candidate_llm_judge_accuracy_mean"] = float(np.average(best_llm_judge_accuracies, weights=weights))
        
        if verbose:
            print(f"\n✅ Computed {len(weighted_summary)} weighted metrics using {mia_weight_tag} weights")

    result = {
        "per_example": per_example_results,
        "dataset_summary": dataset_summary,
    }
    
    if weighted_summary:
        result["weighted_summary"] = weighted_summary
    
    return result


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def save_results_json(results: Dict, path: str, *, split_by_membership: bool = False):
    """Serialize *results* to *path* in human-readable JSON (UTF-8, indent=2).
    
    Parameters
    ----------
    results : Dict
        Results dictionary with 'per_example' and 'dataset_summary' keys
    path : str
        Path to save the main JSON file
    split_by_membership : bool, optional
        If True, also write two additional JSONL files with member and non-member
        results separately. The files will be named {path_stem}_members.jsonl and
        {path_stem}_nonmembers.jsonl.
    """
    # Save main results JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Optionally split results by membership
    if split_by_membership and "per_example" in results:
        per_example = results["per_example"]
        members_examples = []
        nonmembers_examples = []
        
        for example in per_example:
            # Check if example has membership info
            if "extra_info" in example:
                info = example["extra_info"]
                is_member = None
                
                # Try to get membership from is_member flag or split field
                if "is_member" in info:
                    is_member = bool(info["is_member"])
                elif "split" in info:
                    is_member = info["split"].strip() == "train"
                
                if is_member is not None:
                    # Keep the full example including extra_info
                    if is_member:
                        members_examples.append(example)
                    else:
                        nonmembers_examples.append(example)
        
        # Compute dataset summaries for members and non-members
        def compute_subset_summary(examples):
            """Compute dataset summary for a subset of examples."""
            if not examples:
                return {}
            
            # Extract metrics from examples
            metrics = {}
            for key in examples[0]:
                # Include original metric suffixes and math-specific metrics
                if (key.endswith(('_avg', '_best', '_mean', '_max')) or 
                    key in ['math_avg_at_k', 'math_pass_at_k']) and isinstance(examples[0][key], (int, float)):
                    metrics[key] = [ex[key] for ex in examples if key in ex]
            
            # Compute summary statistics
            summary = {}
            for key, values in metrics.items():
                if values:
                    if key.endswith('_avg'):
                        summary[key] = float(np.mean(values))
                    elif key.endswith('_best'):
                        summary[key + '_mean'] = float(np.mean(values))
                        summary[key + '_max'] = float(np.max(values))
                        summary[key] = float(np.max(values))  # Keep original key for compatibility
                    elif key in ['math_avg_at_k', 'math_pass_at_k']:
                        # Add _mean suffix for math metrics to match main dataset summary
                        summary[key + '_mean'] = float(np.mean(values))
                    else:
                        summary[key] = float(np.mean(values))
            
            return summary
        
        # Create member and non-member result dictionaries
        members_results = {
            "per_example": members_examples,
            "dataset_summary": compute_subset_summary(members_examples)
        }
        
        nonmembers_results = {
            "per_example": nonmembers_examples,
            "dataset_summary": compute_subset_summary(nonmembers_examples)
        }
        
        # Write member and non-member JSON files
        from pathlib import Path
        import hashlib
        base_path = Path(path)
        
        # Truncate filename stem if too long to avoid "File name too long" error
        # Linux filename limit is typically 255 bytes, reserve ~50 for suffixes
        max_stem_length = 200
        stem = base_path.stem
        if len(stem) > max_stem_length:
            # Try to preserve important suffixes like step numbers
            # Keep last 30 chars (likely contains step info) + hash of middle part + first 150 chars
            if len(stem) > max_stem_length + 30:
                suffix_part = stem[-30:]
                prefix_part = stem[:max_stem_length - 40]
                hash_part = hashlib.md5(stem[max_stem_length - 40:-30].encode()).hexdigest()[:8]
                stem = f"{prefix_part}_{hash_part}_{suffix_part}"
            else:
                # Simple truncation keeping suffix
                stem = stem[:max_stem_length]
        
        members_path = base_path.parent / f"{stem}_members.json"
        nonmembers_path = base_path.parent / f"{stem}_nonmembers.json"
        
        # Save as JSON files with same format as main results
        with open(members_path, "w", encoding="utf-8") as f:
            json.dump(members_results, f, ensure_ascii=False, indent=2)
        
        with open(nonmembers_path, "w", encoding="utf-8") as f:
            json.dump(nonmembers_results, f, ensure_ascii=False, indent=2)
        
        print(f"Wrote evaluation JSON files: {members_path}, {nonmembers_path}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def _dummy_test():
    """Run a minimal smoke test for the evaluation utilities."""

    gt_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test sentence.",
    ]

    candidates = [
        [
            "The quick brown fox jumps over the lazy dog!",  # near-identical
            "A swift auburn fox leaps above a sleepy dog.",   # paraphrase
        ],
        [
            "Hello world, this is a test sentence.",           # exact
            "Greetings planet, this sentence is a demo.",     # paraphrase
        ],
    ]

    print("[dummy_test] Evaluating", len(gt_texts), "ground truths …")
    
    # Test with different embedding models
    embedding_models = ["fasttext", "qwen3"]  # Test both models
    
    for model in embedding_models:
        try:
            print(f"\n[dummy_test] Testing with embedding model: {model}")
            
            # Test with BLEURT and local LLM judge disabled for speed
            bleurt_config = {
                "length_penalty": "ratio",
                "length_threshold": 1.5,
            }
            
            # Test problems for LLM judge
            problems = [
                "What is the quick brown fox jumping over?",
                "What greeting is commonly used?"
            ]
            
            # Test with LLM judge disabled by default (too slow for smoke test)
            llm_judge_config = {
                "model_name": "microsoft/DialoGPT-small",  # Use small model for testing
                "enable_thinking": False,
                "max_new_tokens": 50
            }
            
            results = evaluate_dataset(
                gt_texts, 
                candidates, 
                verbose=False,
                evaluate_bleurt=False,  # Disable BLEURT for speed
                bleurt_kwargs=bleurt_config,
                evaluate_local_llm_judge=False,  # Disable for speed
                llm_judge_kwargs=llm_judge_config,
                problems_list=problems,
                embedding_model=model
            )
            
            # Print only the dataset summary
            print(f"Dataset summary for {model}:")
            print(json.dumps(results["dataset_summary"], indent=2))
            
        except Exception as e:
            print(f"Error testing {model}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    _dummy_test() 