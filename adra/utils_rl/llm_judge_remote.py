"""
Remote LLM Judge for Reconstruction Evaluation.

This module provides remote LLM-as-a-judge functionality for evaluating reconstruction quality
by connecting to a vLLM server hosting models like Qwen3-32B. This design allows GPU-accelerated
LLM inference without requiring GPU allocation on the evaluation worker.

The module connects to a vLLM server with OpenAI-compatible API, allowing efficient text generation
without local GPU requirements.

Key features:
- Remote vLLM server integration for GPU-accelerated text generation
- Qwen3-32B support with thinking/non-thinking modes
- Connection pooling and retry logic for reliability
- Fallback to lexical similarity when server is unavailable
- Batched processing for efficient LLM judge computation
- Compatible with llm_judge_local.py API for drop-in replacement

Usage example:
    from adra.utils_rl.llm_judge_remote import compute_llm_judge_scores_batch
    
    # Ensure LLM_JUDGE_SERVER_URL is set or pass server_url in extra_info
    scores = compute_llm_judge_scores_batch(
        problems=["What is 2+2?"],
        ground_truths=["The answer is 4."],
        candidates_list=[["4", "2+2=4", "Four"]],
        server_url="http://vllm-server:8000"
    )

Environment Variables:
- LLM_JUDGE_SERVER_URL: URL of the vLLM server (e.g., http://localhost:8000)
- LLM_JUDGE_SERVER_API_KEY: Optional API key for authentication
- LLM_JUDGE_SERVER_TIMEOUT: Request timeout in seconds (default: 60)

To use this module, deploy a vLLM server with Qwen3-32B:
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --port 8000
"""

from __future__ import annotations

import os
import warnings
import logging
from typing import List, Optional, Tuple, Dict, Any
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import tokenizer for lexical metrics (matching reconstruction_evaluation.py)
try:
    from transformers import AutoTokenizer
    _DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    _HAS_TOKENIZER = True
except ImportError:
    _DEFAULT_TOKENIZER = None
    _HAS_TOKENIZER = False
    warnings.warn(
        "transformers not available. Lexical metrics will use regex fallback.",
        RuntimeWarning
    )

# Import n-gram coverage
try:
    from .ngram_coverage import compute_ngram_coverage
    _HAS_NGRAM_COVERAGE = True
except ImportError:
    # Try absolute import as fallback
    try:
        from ngram_coverage import compute_ngram_coverage
        _HAS_NGRAM_COVERAGE = True
    except ImportError:
        _HAS_NGRAM_COVERAGE = False
        warnings.warn(
            "ngram_coverage not available. N-gram coverage metric will be skipped.",
            RuntimeWarning
        )

# Try to import LLM judge client
try:
    from .llm_judge_client import LLMJudgeClient, get_default_client
    _HAS_CLIENT = True
except ImportError:
    # Try absolute import as fallback
    try:
        from llm_judge_client import LLMJudgeClient, get_default_client
        _HAS_CLIENT = True
    except ImportError:
        _HAS_CLIENT = False
        warnings.warn(
            "llm_judge_client not available. Remote LLM judge will not work.",
            RuntimeWarning
        )


logger = logging.getLogger(__name__)

__all__ = [
    "compute_llm_judge_score_single", 
    "compute_llm_judge_scores_batch",
    "_extract_problem_from_prompt"
]

# Default configuration (same as llm_judge_local.py)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_BATCH_SIZE = 128  # Optimized for H200 8-GPU vLLM server with batch_size_per_worker
DEFAULT_ENABLE_THINKING = False
DEFAULT_NUM_WORKERS = 32  # Number of parallel workers for lexical metrics computation

# Import prompt templates
try:
    from verl.utils.reward_score.llm_judge_prompts import get_prompt_template, get_default_template
    DEFAULT_PROMPT_TEMPLATE = get_default_template()
except ImportError:
    # Fallback for standalone usage
    DEFAULT_PROMPT_TEMPLATE = """
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals. Do not use thinking mode.

INPUTS
- Problem:
{PROBLEM}

- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line and nothing else, only the final reward score:
REWARD: <number between 0 and 1 with 3 decimals>
""".strip()

# Global client instance
_GLOBAL_CLIENT: Optional[Any] = None


def _get_client(server_url: Optional[str] = None, **kwargs) -> Optional[Any]:
    """Get or create an LLM judge client."""
    global _GLOBAL_CLIENT
    
    if not _HAS_CLIENT:
        return None
    
    if server_url:
        # Create a new client for specific server
        try:
            return LLMJudgeClient(server_url=server_url, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM judge client for {server_url}: {e}")
            return None
    
    # Use global client
    if _GLOBAL_CLIENT is None:
        try:
            _GLOBAL_CLIENT = get_default_client()
        except Exception as e:
            logger.error(f"Failed to get default LLM judge client: {e}")
            return None
    
    return _GLOBAL_CLIENT



def _tokenize(text: str, max_tokens: Optional[int] = None) -> List[str]:
    """
    Tokenise text into a list of tokens using BERT tokenizer.
    
    This matches the tokenization in reconstruction_evaluation.py for consistency.
    Falls back to regex tokenization if BERT tokenizer is not available.
    
    Args:
        text: Text to tokenize
        max_tokens: Maximum tokens to return (for truncation)
        
    Returns:
        List of tokens
    """
    if _HAS_TOKENIZER and _DEFAULT_TOKENIZER is not None:
        # Use BERT tokenizer (same as reconstruction_evaluation.py)
        return _DEFAULT_TOKENIZER.tokenize(
            text,
            max_length=max_tokens,
            truncation=True,
        )
    else:
        # Fallback to regex tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        if max_tokens is not None and len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokens


def _compute_lexical_metrics(reference: str, candidate: str, metrics_to_compute: Optional[set] = None) -> Dict[str, float]:
    """
    Compute lexical metrics between reference and candidate solutions.
    
    This function computes metrics required by advanced prompt templates (e.g., v4_1)
    that incorporate lexical similarity information.
    
    IMPORTANT: This implementation uses the SAME tokenizer and Jaccard calculation 
    as reconstruction_evaluation.py to ensure consistency across the codebase.
    Both use BERT's bert-base-uncased tokenizer for tokenization.
    
    Args:
        reference: Ground truth solution
        candidate: Candidate solution to evaluate
        metrics_to_compute: Set of metric names to compute. If None, computes all metrics.
                           Valid names: 'lexical_token_overlap', 'lexical_lcs_ratio', 
                           'lexical_lcs_ratio_cand', 'length_ratio', 'lexical_ngram_coverage',
                           'lexical_ngram_coverage_ref'
        
    Returns:
        Dict with requested metrics (subset of six possible metrics):
        - lexical_token_overlap: Jaccard similarity (0-1), computed identically to reconstruction_evaluation.py
        - lexical_lcs_ratio: Normalized LCS ratio (0-1), normalized by ground truth/reference length
        - lexical_lcs_ratio_cand: Normalized LCS ratio (0-1), normalized by candidate length
        - length_ratio: Token length ratio (candidate/reference)
        - lexical_ngram_coverage: N-gram coverage (0-1), normalized by candidate length
        - lexical_ngram_coverage_ref: N-gram coverage (0-1), normalized by reference length
    """
    # If no specific metrics requested, compute all
    if metrics_to_compute is None:
        metrics_to_compute = {
            'lexical_token_overlap', 'lexical_lcs_ratio', 'lexical_lcs_ratio_cand',
            'length_ratio', 'lexical_ngram_coverage', 'lexical_ngram_coverage_ref'
        }
    
    result = {}
    
    # Determine if we need tokenization (needed for most metrics)
    needs_tokenization = any(m in metrics_to_compute for m in [
        'lexical_token_overlap', 'lexical_lcs_ratio', 'lexical_lcs_ratio_cand', 'length_ratio'
    ])
    
    if needs_tokenization:
        # Tokenize both texts (using BERT tokenizer to match reconstruction_evaluation.py)
        ref_tokens = _tokenize(reference)
        cand_tokens = _tokenize(candidate)
    
    # 1. Lexical token overlap (Jaccard similarity)
    if 'lexical_token_overlap' in metrics_to_compute:
        # This matches the implementation in reconstruction_evaluation.py exactly
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        intersection = ref_set & cand_set
        union = ref_set | cand_set
        result['lexical_token_overlap'] = len(intersection) / len(union) if union else 0.0
    
    # 2. Lexical LCS ratio (both normalizations)
    if 'lexical_lcs_ratio' in metrics_to_compute or 'lexical_lcs_ratio_cand' in metrics_to_compute:
        if not ref_tokens or not cand_tokens:
            if 'lexical_lcs_ratio' in metrics_to_compute:
                result['lexical_lcs_ratio'] = 0.0
            if 'lexical_lcs_ratio_cand' in metrics_to_compute:
                result['lexical_lcs_ratio_cand'] = 0.0
        else:
            # Compute LCS length using dynamic programming
            m, n = len(ref_tokens), len(cand_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_tokens[i-1] == cand_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            lcs_length = dp[m][n]
            # Normalize by ground truth length (reference)
            if 'lexical_lcs_ratio' in metrics_to_compute:
                result['lexical_lcs_ratio'] = lcs_length / len(ref_tokens)
            # Normalize by candidate length
            if 'lexical_lcs_ratio_cand' in metrics_to_compute:
                result['lexical_lcs_ratio_cand'] = lcs_length / len(cand_tokens)
    
    # 3. Length ratio (candidate / reference)
    if 'length_ratio' in metrics_to_compute:
        if not ref_tokens:
            result['length_ratio'] = 0.0 if cand_tokens else 1.0
        else:
            result['length_ratio'] = len(cand_tokens) / len(ref_tokens)
    
    # 4. N-gram coverage (normalized by candidate)
    if 'lexical_ngram_coverage' in metrics_to_compute:
        if _HAS_NGRAM_COVERAGE:
            result['lexical_ngram_coverage'] = compute_ngram_coverage(candidate, reference, min_ngram=3, normalize_by="candidate")
        else:
            result['lexical_ngram_coverage'] = 0.0
    
    # 5. N-gram coverage (normalized by reference length)
    if 'lexical_ngram_coverage_ref' in metrics_to_compute:
        if _HAS_NGRAM_COVERAGE:
            result['lexical_ngram_coverage_ref'] = compute_ngram_coverage(candidate, reference, min_ngram=3, normalize_by="reference")
        else:
            result['lexical_ngram_coverage_ref'] = 0.0
    
    return result


def _extract_reward_score(response_text: str, score_range: str = "0-1") -> Optional[float]:
    """
    Extract reward score from LLM response text.
    
    Expected format: "REWARD: X.XXX" where range depends on score_range parameter
    
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
    
    Supports optional placeholders:
    - {PROBLEM}: Problem statement
    - {REFERENCE_SOLUTION}: Ground truth solution
    - {CANDIDATE_SOLUTION}: Candidate solution
    - {LEXICAL_TOKEN_OVERLAP}: Jaccard similarity metric (0-1)
    - {LEXICAL_LCS_RATIO}: Normalized LCS ratio (0-1), normalized by reference
    - {LEXICAL_LCS_RATIO_CAND}: Normalized LCS ratio (0-1), normalized by candidate
    - {LENGTH_RATIO}: Length ratio (candidate/reference)
    - {LEXICAL_NGRAM_COVERAGE}: N-gram coverage metric (0-1), normalized by candidate
    - {LEXICAL_NGRAM_COVERAGE_REF}: N-gram coverage metric (0-1), normalized by reference
    
    Args:
        prompt_template: Template string with placeholders
        problem: The math problem statement
        reference_solution: The ground truth solution
        candidate_solution: The candidate solution to evaluate
        
    Returns:
        Formatted prompt string
    """
    # Build formatting dictionary with base values
    format_dict = {
        "PROBLEM": problem.strip(),
        "REFERENCE_SOLUTION": reference_solution.strip(),
        "CANDIDATE_SOLUTION": candidate_solution.strip()
    }
    
    # Mapping from placeholder names to metric keys
    placeholder_to_metric = {
        "{LEXICAL_TOKEN_OVERLAP}": "lexical_token_overlap",
        "{LEXICAL_LCS_RATIO}": "lexical_lcs_ratio",
        "{LEXICAL_LCS_RATIO_CAND}": "lexical_lcs_ratio_cand",
        "{LENGTH_RATIO}": "length_ratio",
        "{LEXICAL_NGRAM_COVERAGE}": "lexical_ngram_coverage",
        "{LEXICAL_NGRAM_COVERAGE_REF}": "lexical_ngram_coverage_ref"
    }
    
    # Detect which specific metrics are needed
    metrics_to_compute = set()
    for placeholder, metric_key in placeholder_to_metric.items():
        if placeholder in prompt_template:
            metrics_to_compute.add(metric_key)
    
    # Compute and add only the needed lexical metrics
    if metrics_to_compute:
        metrics = _compute_lexical_metrics(reference_solution, candidate_solution, metrics_to_compute)
        # Add computed metrics to format dict
        for placeholder, metric_key in placeholder_to_metric.items():
            if metric_key in metrics:
                # Convert metric key to placeholder name (uppercase without underscores)
                placeholder_name = placeholder[1:-1]  # Remove { and }
                format_dict[placeholder_name] = f"{metrics[metric_key]:.3f}"
    
    return prompt_template.format(**format_dict)


def _format_prompts_parallel(
    prompt_template: str,
    problems: List[str],
    reference_solutions: List[str],
    candidate_solutions: List[str],
    num_workers: int = DEFAULT_NUM_WORKERS
) -> List[str]:
    """
    Format multiple prompts in parallel for efficient batch processing.
    
    This function parallelizes the lexical metrics computation across multiple workers,
    which is especially beneficial when computing expensive metrics like LCS or n-gram coverage.
    
    Args:
        prompt_template: Template string with placeholders
        problems: List of problem statements
        reference_solutions: List of ground truth solutions
        candidate_solutions: List of candidate solutions to evaluate
        num_workers: Number of parallel workers (default: 4)
        
    Returns:
        List of formatted prompt strings
    """
    # For small batches or when num_workers=1, use sequential processing
    if len(problems) < num_workers * 2 or num_workers <= 1:
        return [
            _format_prompt(prompt_template, problem, ref, cand)
            for problem, ref, cand in zip(problems, reference_solutions, candidate_solutions)
        ]
    
    # Parallel processing for larger batches
    formatted_prompts = [None] * len(problems)
    
    print(f"Formatting {len(problems)} prompts with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for idx, (problem, ref, cand) in enumerate(zip(problems, reference_solutions, candidate_solutions)):
            future = executor.submit(_format_prompt, prompt_template, problem, ref, cand)
            future_to_index[future] = idx
        
        # Collect results in order
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                formatted_prompts[idx] = future.result()
            except Exception as e:
                logger.error(f"Error formatting prompt at index {idx}: {e}")
                # Fallback to sequential for this one
                formatted_prompts[idx] = _format_prompt(
                    prompt_template,
                    problems[idx],
                    reference_solutions[idx],
                    candidate_solutions[idx]
                )
    
    return formatted_prompts


def _extract_problem_from_prompt(prompt_data: List[Dict[str, str]]) -> str:
    """
    Extract problem text from chat template format.
    (Same as llm_judge_local.py)
    
    Args:
        prompt_data: List of message dicts in format [{"role": "user", "content": "..."}]
        
    Returns:
        Problem text string
    """
    for message in prompt_data:
        if message.get("role") == "user":
            return message.get("content", "").strip()
    return ""


def _single_llm_judge_score(
    problem: str,
    ground_truth: str,
    candidate: str,
    client: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    **generation_kwargs
) -> Optional[float]:
    """Compute LLM judge score for a single candidate using remote server."""
    if client is None:
        logger.error("Remote LLM judge client is not available")
        return None
    
    try:
        # Format prompt
        formatted_prompt = _format_prompt(prompt_template, problem, ground_truth, candidate)
        
        # Generate response
        responses = client.generate_responses(
            prompts=[formatted_prompt],
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            **generation_kwargs
        )
        
        if not responses or not responses[0]:
            logger.error("No response generated from remote server")
            return None
        
        # Extract score
        score_range = _detect_score_range(prompt_template)
        score = _extract_reward_score(responses[0], score_range)
        if score is None:
            logger.error(f"Failed to extract score from remote response: {responses[0][:100]}...")
            return None
        
        return score
        
    except Exception as e:
        logger.error(f"Error in remote LLM judge scoring: {e}")
        return None


def _batch_llm_judge_scores(
    problems: List[str],
    ground_truths: List[str],
    candidates: List[str],
    client: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    **generation_kwargs
) -> Optional[List[float]]:
    """
    Compute LLM judge scores for multiple problem-ground_truth-candidate triplets efficiently.
    
    This batches all prompts to the vLLM server for optimal throughput and uses
    parallel workers to compute lexical metrics for prompt formatting.
    
    Args:
        num_workers: Number of parallel workers for lexical metrics computation (default: 4)
    
    Returns:
        List of scores if successful, None if remote server is unavailable or failed
    """
    if client is None:
        logger.error("Remote LLM judge client is not available")
        return None
    
    try:
        import time
        start_time = time.time()
        
        # Prepare all prompts in parallel
        all_prompts = _format_prompts_parallel(
            prompt_template, problems, ground_truths, candidates, num_workers
        )
        
        logger.info(f"🔥 Batch processing {len(all_prompts)} prompts with batch_size={batch_size}")
        
        # Generate responses in batches
        all_responses = client.generate_responses(
            prompts=all_prompts,
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            batch_size=batch_size,
            **generation_kwargs
        )
        
        if not all_responses:
            logger.error("No responses generated from remote server")
            return None
        
        # Extract scores
        scores = []
        failed_extractions = 0
        score_range = _detect_score_range(prompt_template)
        for i, response in enumerate(all_responses):
            if response:
                score = _extract_reward_score(response, score_range)
                if score is not None:
                    scores.append(score)
                else:
                    logger.warning(f"Failed to extract score from response {i}: {response[:100]}...")
                    scores.append(0.0)  # Use 0.0 for failed extractions
                    failed_extractions += 1
            else:
                logger.warning(f"Empty response for prompt {i}")
                scores.append(0.0)  # Use 0.0 for empty responses
                failed_extractions += 1
        
        if failed_extractions > 0:
            logger.warning(f"Failed to extract scores for {failed_extractions}/{len(all_responses)} responses")
        
        duration = time.time() - start_time
        throughput = len(all_prompts) / duration if duration > 0 else 0
        logger.info(f"✅ Batch completed: {len(all_prompts)} prompts in {duration:.1f}s ({throughput:.1f} prompts/sec)")
        
        return scores
        
    except Exception as e:
        logger.error(f"Batch remote LLM judge failed: {e}")
        return None


def compute_llm_judge_score_single(
    problem: str,
    ground_truth: str,
    candidate: str,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_template_name: Optional[str] = None,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 600.0,  # Increased to 5 minutes for long prompts
    num_workers: int = DEFAULT_NUM_WORKERS,
    **generation_kwargs
) -> Optional[float]:
    """
    Compute LLM judge score for a single candidate using remote vLLM server.
    
    Args:
        problem: The problem statement
        ground_truth: The reference solution
        candidate: The candidate solution to evaluate
        model_name: Model name on the vLLM server
        prompt_template: Prompt template string (if None, uses prompt_template_name)
        prompt_template_name: Name of prompt template to use ("default", "detailed", "concise")
        enable_thinking: Whether to enable thinking mode
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        server_url: Optional server URL (overrides environment)
        api_key: Optional API key (overrides environment)
        timeout: Request timeout in seconds
        num_workers: Number of parallel workers for lexical metrics computation (default: 4)
        **generation_kwargs: Additional generation arguments
        
    Returns:
        Score between 0 and 1, or None if remote server is unavailable
    """
    # Handle prompt template selection
    if prompt_template_name is not None:
        try:
            from verl.utils.reward_score.llm_judge_prompts import get_prompt_template
            prompt_template = get_prompt_template(prompt_template_name)
        except ImportError:
            logger.warning(f"Cannot import prompt templates, using default template instead of '{prompt_template_name}'")
        except ValueError as e:
            logger.warning(f"Invalid prompt template name '{prompt_template_name}': {e}")
    
    # Get or create client
    client = _get_client(server_url=server_url, api_key=api_key, timeout=timeout)
    
    # Report failure if no client available
    if client is None:
        logger.error("No LLM judge server available - remote LLM judge evaluation failed")
        return None
    
    return _single_llm_judge_score(
        problem, ground_truth, candidate, client,
        model_name, prompt_template, enable_thinking,
        temperature, top_p, max_new_tokens, **generation_kwargs
    )


def compute_llm_judge_scores_batch(
    problems: List[str],
    ground_truths: List[str],
    candidates_list: List[List[str]],
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_template_name: Optional[str] = None,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 600.0,  # Increased to 5 minutes for long prompts
    num_workers: int = DEFAULT_NUM_WORKERS,
    **generation_kwargs
) -> Tuple[Optional[List[List[float]]], Optional[List[int]]]:
    """
    Compute LLM judge scores for multiple problems with multiple candidates each using remote vLLM server.
    
    Args:
        problems: List of problem statements
        ground_truths: List of reference solutions
        candidates_list: List of candidate lists (one list per problem)
        model_name: Model name on the vLLM server
        prompt_template: Prompt template string (if None, uses prompt_template_name)
        prompt_template_name: Name of prompt template to use ("default", "detailed", "concise")
        enable_thinking: Whether to enable thinking mode
        batch_size: Batch size for server requests
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        server_url: Optional server URL (overrides environment)
        api_key: Optional API key (overrides environment)
        timeout: Request timeout in seconds
        num_workers: Number of parallel workers for lexical metrics computation (default: 4)
        **generation_kwargs: Additional generation arguments
        
    Returns:
        Tuple of (scores_per_problem, best_indices_per_problem) or (None, None) if server unavailable
        - scores_per_problem: List of score lists, one per problem, or None if failed
        - best_indices_per_problem: List of best candidate indices, one per problem, or None if failed
    """
    if not problems or not ground_truths or not candidates_list:
        return [], []
    
    if len(problems) != len(ground_truths) or len(problems) != len(candidates_list):
        raise ValueError("problems, ground_truths, and candidates_list must have the same length")
    
    # Handle prompt template selection
    if prompt_template_name is not None:
        try:
            from verl.utils.reward_score.llm_judge_prompts import get_prompt_template
            prompt_template = get_prompt_template(prompt_template_name)
        except ImportError:
            logger.warning(f"Cannot import prompt templates, using default template instead of '{prompt_template_name}'")
        except ValueError as e:
            logger.warning(f"Invalid prompt template name '{prompt_template_name}': {e}")
    
    # Get or create client
    client = _get_client(server_url=server_url, api_key=api_key, timeout=timeout)
    
    # Report failure if no client available
    if client is None:
        logger.error("No LLM judge server available - remote LLM judge evaluation failed")
        return None, None
    
    try:
        # Prepare all problem-ground_truth-candidate triplets
        all_problems = []
        all_ground_truths = []
        all_candidates = []
        triplet_mappings = []  # (problem_idx, candidate_idx)
        
        for prob_idx, (problem, ground_truth, candidates) in enumerate(zip(problems, ground_truths, candidates_list)):
            for cand_idx, candidate in enumerate(candidates):
                all_problems.append(problem)
                all_ground_truths.append(ground_truth)
                all_candidates.append(candidate)
                triplet_mappings.append((prob_idx, cand_idx))
        
        # Get scores for all triplets in batches
        logger.info(f"Generating LLM judge scores for {len(all_problems)} triplets in batches of {batch_size}")
        all_scores = _batch_llm_judge_scores(
            all_problems, all_ground_truths, all_candidates, client,
            model_name, prompt_template, enable_thinking,
            temperature, top_p, max_new_tokens, batch_size, num_workers, **generation_kwargs
        )
        
        if all_scores is None:
            logger.error("Batch LLM judge scoring failed - no scores returned")
            return None, None
        
        # Organize scores by problem
        scores_per_problem = [[] for _ in range(len(problems))]
        
        for triplet_idx, score in enumerate(all_scores):
            prob_idx, cand_idx = triplet_mappings[triplet_idx]
            
            # Ensure the scores list is long enough
            while len(scores_per_problem[prob_idx]) <= cand_idx:
                scores_per_problem[prob_idx].append(0.0)
            
            scores_per_problem[prob_idx][cand_idx] = score
        
        # Find best candidate indices
        best_indices = []
        for scores in scores_per_problem:
            if scores:
                best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            else:
                best_idx = 0
            best_indices.append(best_idx)
        
        return scores_per_problem, best_indices
        
    except Exception as e:
        logger.error(f"Error in batch remote LLM judge scoring: {e}")
        return None, None


def clear_client_cache():
    """Clear the client cache and close connections."""
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT:
        _GLOBAL_CLIENT.close()
        _GLOBAL_CLIENT = None
    logger.info("Remote LLM judge client cache cleared")


# Test function
def _test_remote_llm_judge():
    """Simple test function to verify the remote implementation works."""
    print("[test] Testing remote LLM judge implementation...")
    
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
    
    # Test server connectivity
    server_url = os.getenv("LLM_JUDGE_SERVER_URL", "http://localhost:8000")
    print(f"\n[test] Testing connectivity to {server_url}...")
    
    client = _get_client(server_url=server_url)
    if client is None:
        print("❌ Cannot connect to remote server, testing fallback behavior...")
    else:
        print("✅ Connected to remote server")
        if client.health_check():
            print("✅ Server health check passed")
        else:
            print("⚠️ Server health check failed")
    
    # Test single scoring with default template
    print("\n[test] Testing single scoring with default template...")
    score = compute_llm_judge_score_single(
        problem=problems[0],
        ground_truth=ground_truths[0],
        candidate=candidates_list[0][0],
        server_url=server_url,
        max_new_tokens=50
    )
    if score is not None:
        print(f"✅ Single score (default): {score}")
    else:
        print("❌ Single scoring failed - returned None")
    
    
    # Test batch scoring with default template
    print("\n[test] Testing batch scoring with default template...")
    scores_per_problem, best_indices = compute_llm_judge_scores_batch(
        problems=problems,
        ground_truths=ground_truths,
        candidates_list=candidates_list,
        server_url=server_url,
        batch_size=2,
        max_new_tokens=50
    )
    
    if scores_per_problem is not None and best_indices is not None:
        print(f"✅ Batch scores (default): {scores_per_problem}")
        print(f"✅ Best indices (default): {best_indices}")
    else:
        print("❌ Batch scoring failed - returned None")
    
    # Test batch scoring with concise template
    print("\n[test] Testing batch scoring with concise template...")
    scores_per_problem_concise, best_indices_concise = compute_llm_judge_scores_batch(
        problems=problems,
        ground_truths=ground_truths,
        candidates_list=candidates_list,
        prompt_template_name="concise",
        server_url=server_url,
        batch_size=2,
        max_new_tokens=30
    )
    
    if scores_per_problem_concise is not None and best_indices_concise is not None:
        print(f"✅ Batch scores (concise): {scores_per_problem_concise}")
        print(f"✅ Best indices (concise): {best_indices_concise}")
    else:
        print("❌ Batch scoring with concise template failed - returned None")
    
    # Clear cache
    clear_client_cache()
    print("\n[test] Test completed!")


if __name__ == "__main__":
    _test_remote_llm_judge()
