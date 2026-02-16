"""
Local LLM Judge for Reconstruction Evaluation.

This module provides local LLM-as-a-judge functionality for evaluating reconstruction quality
by loading HuggingFace models locally instead of using API calls. It's designed to integrate
with the reconstruction evaluation pipeline.

Key features:
- Local HuggingFace model loading with caching
- Batch inference for efficiency
- Support for thinking/non-thinking modes (Qwen3 specific)
- Score extraction from LLM responses
- Configurable prompt templates

Usage:
    from adra.utils_rl.llm_judge_local import compute_llm_judge_scores_batch
    
    scores = compute_llm_judge_scores_batch(
        problems=["What is 2+2?"],
        ground_truths=["The answer is 4."],
        candidates_list=[["4", "2+2=4", "Four"]],
        model_name="Qwen/Qwen3-32B",
        enable_thinking=False
    )
"""

from __future__ import annotations

import re
import json
import warnings
from typing import Dict, List, Optional, Any, Tuple
import logging

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Try to import HuggingFace transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    warnings.warn(
        "transformers is not available. Local LLM judge will not work. "
        "Install it via `pip install transformers`.",
        RuntimeWarning,
    )

# Model cache for efficient loading
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}

# Default configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_BATCH_SIZE = 8
DEFAULT_ENABLE_THINKING = False

# Default prompt template (same as llm_judge.py)
DEFAULT_PROMPT_TEMPLATE = """
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

INPUTS
- Problem:
{PROBLEM}

- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <number between 0 and 1 with 3 decimals>
""".strip()


def _extract_reward_score(response_text: str, score_range: str = "0-1") -> Optional[float]:
    """
    Extract reward score from LLM response text.
    
    Expected format: "REWARD: X.XXX" where range depends on score_range parameter
    (Updated to handle both 0-1 and 0-100 scales)
    
    Args:
        response_text: Raw response from LLM
        score_range: Expected score range, either "0-1" or "0-100"
        
    Returns:
        Extracted score normalized to [0, 1] range, or None if extraction failed
    """
    if not response_text:
        return None
    
    # Look for "REWARD:" followed by a number
    pattern = r"REWARD:\s*([0-9]*\.?[0-9]+)"
    match = re.search(pattern, response_text.strip(), re.IGNORECASE)
    
    if match:
        try:
            score = float(match.group(1))
            
            # Normalize based on expected range
            if score_range == "0-100":
                # For 0-100 scale, normalize to 0-1
                normalized_score = score / 100.0
                return max(0.0, min(1.0, normalized_score))
            else:
                # For 0-1 scale, clamp to [0, 1] range
                return max(0.0, min(1.0, score))
        except ValueError:
            pass
    
    # Fallback: look for any number
    if score_range == "0-100":
        # For 0-100 scale, look for numbers up to 100
        pattern = r"\b(\d{1,3}(?:\.\d+)?)\b"
        matches = re.findall(pattern, response_text)
        if matches:
            try:
                score = float(matches[-1])  # Take the last match
                normalized_score = score / 100.0
                return max(0.0, min(1.0, normalized_score))
            except ValueError:
                pass
    else:
        # For 0-1 scale, look for numbers between 0 and 1
        pattern = r"\b(0\.\d{1,3}|1\.0{1,3}|0|1)\b"
        matches = re.findall(pattern, response_text)
        if matches:
            try:
                score = float(matches[-1])  # Take the last match
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
    
    return None


def _detect_score_range(prompt_template: str) -> str:
    """
    Detect the expected score range from the prompt template.
    
    Args:
        prompt_template: The prompt template string
        
    Returns:
        "0-100" if template expects 0-100 scale, "0-1" otherwise
    """
    if "between 0 and 100" in prompt_template or "0-100" in prompt_template:
        return "0-100"
    return "0-1"


def _format_prompt(
    prompt_template: str,
    problem: str,
    reference_solution: str,
    candidate_solution: str
) -> str:
    """
    Format the prompt template with the given inputs.
    (Reused from llm_judge.py)
    
    Args:
        prompt_template: Template string with placeholders
        problem: The math problem statement
        reference_solution: The ground truth solution
        candidate_solution: The candidate solution to evaluate
        
    Returns:
        Formatted prompt string
    """
    return prompt_template.format(
        PROBLEM=problem.strip(),
        REFERENCE_SOLUTION=reference_solution.strip(),
        CANDIDATE_SOLUTION=candidate_solution.strip()
    )


def _load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    **model_kwargs
) -> Tuple[Any, Any]:
    """
    Load HuggingFace model and tokenizer with caching.
    
    Args:
        model_name: HuggingFace model identifier
        device_map: Device mapping strategy
        torch_dtype: Torch data type for model
        **model_kwargs: Additional model loading arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for local LLM judge")
    
    # Check cache
    cache_key = f"{model_name}_{device_map}_{torch_dtype}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], _TOKENIZER_CACHE[cache_key]
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set left padding for decoder-only models (critical for batch inference)
    tokenizer.padding_side = 'left'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **model_kwargs
    )
    model.eval()
    
    # Cache for future use
    _MODEL_CACHE[cache_key] = model
    _TOKENIZER_CACHE[cache_key] = tokenizer
    
    logger.info(f"Successfully loaded model on device: {model.device}")
    return model, tokenizer


def _extract_problem_from_prompt(prompt_data: List[Dict[str, str]]) -> str:
    """
    Extract problem text from chat template format.
    
    Args:
        prompt_data: List of message dicts in format [{"role": "user", "content": "..."}]
        
    Returns:
        Problem text string
    """
    for message in prompt_data:
        if message.get("role") == "user":
            return message.get("content", "").strip()
    return ""


def _generate_responses_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    **generation_kwargs
) -> List[str]:
    """
    Generate responses from model in batches.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of formatted prompts
        batch_size: Batch size for inference
        max_new_tokens: Maximum new tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        enable_thinking: Whether to enable thinking mode (Qwen3 specific)
        **generation_kwargs: Additional generation arguments
        
    Returns:
        List of generated response texts
    """
    responses = []
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating LLM responses"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Prepare messages for tokenization
        batch_messages = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            batch_messages.append(messages)
        
        # Apply chat template
        batch_texts = []
        for messages in batch_messages:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking  # Qwen3 specific
            )
            batch_texts.append(text)
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        # Decode responses
        input_lengths = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            # Extract only the new tokens (response)
            response_tokens = output[input_lengths:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response_text)
    
    return responses


def compute_llm_judge_score_single(
    problem: str,
    ground_truth: str,
    candidate: str,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    **model_kwargs
) -> float:
    """
    Compute LLM judge score for a single candidate.
    
    Args:
        problem: The problem statement
        ground_truth: The reference solution
        candidate: The candidate solution to evaluate
        model_name: HuggingFace model identifier
        prompt_template: Prompt template string
        enable_thinking: Whether to enable thinking mode
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        **model_kwargs: Additional model loading arguments
        
    Returns:
        Score between 0 and 1
    """
    try:
        # Load model and tokenizer
        model, tokenizer = _load_model_and_tokenizer(model_name, **model_kwargs)
        
        # Format prompt
        formatted_prompt = _format_prompt(prompt_template, problem, ground_truth, candidate)
        
        # Generate response
        responses = _generate_responses_batch(
            model, tokenizer, [formatted_prompt],
            batch_size=1, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            enable_thinking=enable_thinking
        )
        
        if not responses:
            logger.warning("No response generated")
            return 0.0
        
        # Extract score
        score_range = _detect_score_range(prompt_template)
        score = _extract_reward_score(responses[0], score_range)
        if score is None:
            logger.warning(f"Failed to extract score from response: {responses[0][:100]}...")
            return 0.0
        
        return score
        
    except Exception as e:
        logger.error(f"Error in LLM judge scoring: {e}")
        return 0.0


def compute_llm_judge_scores_batch(
    problems: List[str],
    ground_truths: List[str],
    candidates_list: List[List[str]],
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    **model_kwargs
) -> Tuple[List[List[float]], List[int]]:
    """
    Compute LLM judge scores for multiple problems with multiple candidates each.
    
    Args:
        problems: List of problem statements
        ground_truths: List of reference solutions
        candidates_list: List of candidate lists (one list per problem)
        model_name: HuggingFace model identifier
        prompt_template: Prompt template string
        enable_thinking: Whether to enable thinking mode
        batch_size: Batch size for inference
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        **model_kwargs: Additional model loading arguments
        
    Returns:
        Tuple of (scores_per_problem, best_indices_per_problem)
        - scores_per_problem: List of score lists, one per problem
        - best_indices_per_problem: List of best candidate indices, one per problem
    """
    if not problems or not ground_truths or not candidates_list:
        return [], []
    
    if len(problems) != len(ground_truths) or len(problems) != len(candidates_list):
        raise ValueError("problems, ground_truths, and candidates_list must have the same length")
    
    try:
        # Load model and tokenizer once
        model, tokenizer = _load_model_and_tokenizer(model_name, **model_kwargs)
        
        # Prepare all prompts for batch processing
        all_prompts = []
        prompt_mappings = []  # (problem_idx, candidate_idx)
        
        for prob_idx, (problem, ground_truth, candidates) in enumerate(zip(problems, ground_truths, candidates_list)):
            for cand_idx, candidate in enumerate(candidates):
                formatted_prompt = _format_prompt(prompt_template, problem, ground_truth, candidate)
                all_prompts.append(formatted_prompt)
                prompt_mappings.append((prob_idx, cand_idx))
        
        # Generate all responses in batches
        logger.info(f"Generating responses for {len(all_prompts)} prompts in batches of {batch_size}")
        all_responses = _generate_responses_batch(
            model, tokenizer, all_prompts,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            enable_thinking=enable_thinking
        )
        
        # Extract scores and organize by problem
        scores_per_problem = [[] for _ in range(len(problems))]
        score_range = _detect_score_range(prompt_template)
        
        for response_idx, response in enumerate(all_responses):
            prob_idx, cand_idx = prompt_mappings[response_idx]
            
            score = _extract_reward_score(response, score_range)
            if score is None:
                logger.warning(f"Failed to extract score for problem {prob_idx}, candidate {cand_idx}")
                score = 0.0
            
            # Ensure the scores list is long enough
            while len(scores_per_problem[prob_idx]) <= cand_idx:
                scores_per_problem[prob_idx].append(0.0)
            
            scores_per_problem[prob_idx][cand_idx] = score
        
        # Find best candidate indices
        best_indices = []
        for scores in scores_per_problem:
            if scores:
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0
            best_indices.append(best_idx)
        
        return scores_per_problem, best_indices
        
    except Exception as e:
        logger.error(f"Error in batch LLM judge scoring: {e}")
        # Return zero scores for all problems
        scores_per_problem = [[0.0] * len(candidates) for candidates in candidates_list]
        best_indices = [0] * len(problems)
        return scores_per_problem, best_indices


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _MODEL_CACHE, _TOKENIZER_CACHE
    _MODEL_CACHE.clear()
    _TOKENIZER_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")


# Test function
def _test_local_llm_judge():
    """Simple test function to verify the implementation works."""
    print("[test] Testing local LLM judge implementation...")
    
    problems = [
        "What is 2 + 2?",
        "What is the square root of 16?"
    ]
    
    ground_truths = [
        "2 + 2 = 4",
        "The square root of 16 is 4."
    ]
    
    candidates_list = [
        ["4", "The answer is four", "2+2 equals 4"],
        ["4", "sqrt(16) = 4", "The answer is 4"]
    ]
    
    # Test single scoring
    print("\n[test] Testing single scoring...")
    score = compute_llm_judge_score_single(
        problem=problems[0],
        ground_truth=ground_truths[0],
        candidate=candidates_list[0][0],
        model_name="microsoft/DialoGPT-small",  # Use smaller model for testing
        max_new_tokens=50
    )
    print(f"Single score: {score}")
    
    # Test batch scoring
    print("\n[test] Testing batch scoring...")
    scores_per_problem, best_indices = compute_llm_judge_scores_batch(
        problems=problems,
        ground_truths=ground_truths,
        candidates_list=candidates_list,
        model_name="microsoft/DialoGPT-small",  # Use smaller model for testing
        batch_size=2,
        max_new_tokens=50
    )
    
    print(f"Batch scores: {scores_per_problem}")
    print(f"Best indices: {best_indices}")
    
    # Clear cache
    clear_model_cache()
    print("\n[test] Test completed!")


if __name__ == "__main__":
    _test_local_llm_judge()
