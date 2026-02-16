"""evaluate_reconstructions.py – Command-line entry-point for reconstruction evaluation.

Usage
-----
$ python -m adra.scripts.evaluate_reconstructions \
      --input /path/to/data.parquet \
      --output /path/to/results.json

The script extracts the *ground truth* texts and their corresponding
*candidate* reconstructions from the parquet file (see README for the expected
schema) and computes lexical, embedding, and optionally BLEURT similarity metrics via
``adra.utils.reconstruction_evaluation``.
"""

from __future__ import annotations

import numpy as np  
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from adra.utils_rl.reconstruction_evaluation import (
    evaluate_dataset,
    save_results_json,
    _remove_prefix_from_text,
)
from adra.utils_rl.llm_judge_local import (
    _extract_problem_from_prompt,
    DEFAULT_MODEL_NAME as DEFAULT_LLM_JUDGE_MODEL
)

# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_fields(df: pd.DataFrame, need_membership: bool = False, use_transformed_gt: bool = False, extract_problems: bool = False, prefix_ratio: float = 0.0) -> Tuple[List[str], List[List[str]], List[str], List[int], List[str], List[float], str]:
    """Return ground_truth_texts, candidates_list, splits, original_ids, problems, mia_weights, mia_weight_tag extracted from *df*.

    Expected schema (per row):
        - reward_model : dict w/ key 'ground_truth'
        - responses    : list[str]  (candidate reconstructions)
        - prompt       : list[dict] (for problem extraction when extract_problems=True)
        - extra_info   : dict w/ keys:
            - 'split' ("train"|"test") and 'id' (original index) OR
            - 'is_member' (bool) to explicitly mark membership
            (required only when need_membership is True)
            - 'transformed_ground_truth' (str) if use_transformed_gt is True
            
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the expected schema
    need_membership : bool, default False
        Whether to extract membership information
    use_transformed_gt : bool, default False
        Whether to use transformed ground truths
    extract_problems : bool, default False
        Whether to extract problem texts from prompt field
    prefix_ratio : float, default 0.0
        Ratio of words to remove from the beginning of ground truths (0.0 to 1.0).
        This enables evaluation of only the non-prefix portion.
        
    Returns
    -------
    Tuple[List[str], List[List[str]], List[str], List[int], List[str], List[float], str]
        ground_truth_texts, candidates_list, splits, original_ids, problems, mia_weights, mia_weight_tag
    """
    if "reward_model" not in df.columns or "responses" not in df.columns:
        missing = {c for c in ("reward_model", "responses") if c not in df.columns}
        raise ValueError(f"Input parquet is missing columns: {', '.join(missing)}")
        
    # Check for prompt column if problems need to be extracted
    if extract_problems and "prompt" not in df.columns:
        raise ValueError("Input parquet is missing 'prompt' column (required for problem extraction)")

    # Extract ground truths - check for member/non-member separation first
    ground_truths: List[str] = []
    expanded_candidates: List[List[str]] = []
    expanded_splits: List[str] = []
    expanded_ids: List[int] = []
    problems: List[str] = []
    
    def _normalise(responses):
        if isinstance(responses, list):
            return responses
        if isinstance(responses, np.ndarray):
            return responses.tolist()
        raise TypeError(f"Unexpected responses type: {type(responses)}")
    
    # Check if we have member/non-member ground truths in extra_info
    has_mia_ground_truths = False
    if "extra_info" in df.columns:
        for i, info in enumerate(df["extra_info"]):
            if isinstance(info, dict) and info.get("has_nonmember_gt", False):
                has_mia_ground_truths = True
                break
    
    if has_mia_ground_truths and need_membership:
        # Expand records: create separate entries for member and non-member ground truths
        print("Detected MIA ground truths - expanding records for member/non-member evaluation...")
        
        for i, (row, info) in enumerate(zip(df["reward_model"], df["extra_info"])):
            if not isinstance(info, dict):
                raise TypeError(f"Row {i}: expected extra_info dict, got {type(info)}")
            
            candidates = _normalise(df["responses"].iloc[i])
            
            # Get original ID
            oid = info.get("original_idx", info.get("id", i))
            if not isinstance(oid, int):
                oid = int(oid)
            
            # Check if this record has both member and non-member ground truths
            if info.get("has_nonmember_gt", False):
                # Add member record
                if use_transformed_gt:
                    # For member evaluation with transformed GT: use transformed version of correct solution
                    member_gt = info.get("transformed_ground_truth")
                    if member_gt is None:
                        # Fallback: if no transformed version, use original member ground truth
                        member_gt = info.get("member_ground_truth", row["ground_truth"])
                else:
                    # Standard member evaluation: use original correct solution
                    member_gt = info.get("member_ground_truth", row["ground_truth"])
                
                ground_truths.append(str(member_gt))
                expanded_candidates.append(candidates)
                expanded_splits.append("train")  # member = train
                expanded_ids.append(oid)
                
                # Extract problem if needed
                if extract_problems:
                    problem_text = _extract_problem_from_prompt(df["prompt"].iloc[i])
                    problems.append(problem_text)
                
                # Add non-member record
                # Note: For non-member evaluation, we ALWAYS use the perturbed/incorrect solution
                # regardless of use_transformed_gt flag, since transformed_ground_truth is a 
                # transformation of the CORRECT solution, not the perturbed one
                nonmember_gt = info.get("nonmember_ground_truth")
                if nonmember_gt is not None:
                    ground_truths.append(str(nonmember_gt))
                    expanded_candidates.append(candidates)  # Same candidates, different ground truth
                    expanded_splits.append("test")  # non-member = test
                    expanded_ids.append(oid)  # Same original ID
                    
                    # Add same problem for non-member record
                    if extract_problems:
                        problems.append(problem_text)  # Same problem as member
            else:
                # Single ground truth - determine if it's member or non-member
                # For single GT records, use transformed version if requested
                if use_transformed_gt:
                    gt = info.get("transformed_ground_truth", row["ground_truth"])
                else:
                    gt = row["ground_truth"]
                
                ground_truths.append(str(gt))
                expanded_candidates.append(candidates)
                
                # Extract problem if needed
                if extract_problems:
                    problem_text = _extract_problem_from_prompt(df["prompt"].iloc[i])
                    problems.append(problem_text)
                
                # Determine membership from existing info
                is_member = info.get("is_member")
                if is_member is not None:
                    split = "train" if is_member else "test"
                else:
                    split = str(info.get("split", "train")).strip()
                
                expanded_splits.append(split)
                expanded_ids.append(oid)
        
        # Use expanded data
        candidates_list = expanded_candidates
        splits = expanded_splits
        original_ids = expanded_ids
        
    else:
        # Original behavior: single ground truth per record (no MIA expansion)
        if use_transformed_gt:
            # Use transformed ground truths for evaluation
            if "extra_info" not in df.columns:
                raise ValueError("Input parquet is missing column: extra_info (required for transformed ground truths)")
            
            for i, info in enumerate(df["extra_info"]):
                if not isinstance(info, dict):
                    raise TypeError(f"Row {i}: expected extra_info dict, got {type(info)}")
                
                transformed_gt = info.get("transformed_ground_truth")
                if transformed_gt is None:
                    raise ValueError(f"Row {i}: transformed_ground_truth not found in extra_info")
                
                ground_truths.append(str(transformed_gt))
        else:
            # Use original ground truths from reward_model
            ground_truths = [row["ground_truth"] for row in df["reward_model"]]
        
        candidates_list: List[List[str]] = [_normalise(r) for r in df["responses"]]
        
        # Extract problems if needed
        if extract_problems:
            for i, prompt_data in enumerate(df["prompt"]):
                problem_text = _extract_problem_from_prompt(prompt_data)
                problems.append(problem_text)

    # Initialize splits and original_ids for the cases where we don't have them yet
    if not has_mia_ground_truths or not need_membership:
        splits: List[str] = []
        original_ids: List[int] = []
    
    # Extract membership metadata only if requested and not already handled
    if need_membership and not (has_mia_ground_truths and need_membership):
        if "extra_info" not in df.columns:
            raise ValueError("Input parquet is missing column: extra_info")

        for i, info in enumerate(df["extra_info"]):
            if not isinstance(info, dict):
                raise TypeError(f"Row {i}: expected extra_info dict, got {type(info)}")
            
            # First try to get explicit is_member flag
            is_member = info.get("is_member")
            if is_member is not None:
                # Convert boolean is_member to train/test split
                split = "train" if is_member else "test"
            else:
                # Fall back to original split field
                split = str(info.get("split", "")).strip()
                if split not in {"train", "test"}:
                    raise ValueError(
                        f"Row {i}: When is_member is not present, extra_info.split must be 'train' or 'test', got {split!r}"
                    )
            
            # Get ID - first try original_idx for unused examples, then id field
            oid = info.get("original_idx", info.get("id"))
            if oid is None:
                # For random pairing method, construct ID from problem/solution indices
                prob_idx = info.get("original_problem_idx")
                sol_idx = info.get("original_solution_idx")
                if prob_idx is not None and sol_idx is not None:
                    oid = prob_idx  # Use problem index as the ID
                else:
                    oid = i  # Fall back to row index if no other ID available
            
            if not isinstance(oid, int):
                oid = int(oid)  # Convert to int if needed
                
            splits.append(split)
            original_ids.append(oid)

    # Basic sanity checks
    for i, (gt, cands) in enumerate(zip(ground_truths, candidates_list)):
        if not isinstance(gt, str):
            raise TypeError(f"Row {i}: expected ground_truth to be str, got {type(gt)}")
        if not isinstance(cands, list):
            raise TypeError(
                f"Row {i}: expected responses list after normalisation, got {type(cands)}"
            )

    # Ensure all return values are properly defined
    if not need_membership:
        splits = []
        original_ids = []
    
    if not extract_problems:
        problems = []
    
    # Apply prefix truncation if requested
    if prefix_ratio > 0.0:
        original_count = len(ground_truths)
        ground_truths = [_remove_prefix_from_text(gt, prefix_ratio) for gt in ground_truths]
        print(f"Applied prefix truncation (ratio={prefix_ratio:.2f}) to {original_count} ground truths")
    
    # Extract MIA weights - handle both single weights and separated member/non-member weights
    mia_weights = []
    mia_weight_tag = None
    if "extra_info" in df.columns:
        for i, info in enumerate(df["extra_info"]):
            if isinstance(info, dict):
                # Check for single mia_weight (used by unused_examples mode)
                if "mia_weight" in info:
                    if i == 0:
                        # First example with MIA weight - get the tag
                        mia_weight_tag = info.get("mia_weight_tag", "unknown")
                    mia_weights.append(float(info["mia_weight"]))
                # Check for separated member/non-member weights (used by perturbed_solution mode)
                elif "member_mia_weight" in info and "nonmember_mia_weight" in info:
                    if i == 0:
                        # First example with MIA weight - get the tag
                        mia_weight_tag = info.get("mia_weight_tag", "unknown")
                    
                    # Extract membership information to choose correct weight
                    is_member = info.get("is_member", True)  # Default to member if not specified
                    if is_member:
                        mia_weights.append(float(info["member_mia_weight"]))
                    else:
                        mia_weights.append(float(info["nonmember_mia_weight"]))
                elif len(mia_weights) > 0:
                    # Missing MIA weight for some examples - clear the list
                    print(f"Warning: MIA weight missing at index {i}, disabling MIA weights")
                    mia_weights = []
                    mia_weight_tag = None
                    break
        
        # Verify we got weights for all examples (considering expansion for MIA ground truths)
        expected_length = len(ground_truths)
        if mia_weights and len(mia_weights) == len(df):
            # If we have expanded records (MIA ground truths), handle weight expansion correctly
            if has_mia_ground_truths and need_membership and len(ground_truths) > len(df):
                expanded_mia_weights = []
                df_idx = 0
                
                # Check if we have the separated weight format (perturbed_solution mode)
                has_separated_weights = False
                if "extra_info" in df.columns:
                    for info in df["extra_info"]:
                        if isinstance(info, dict) and "member_mia_weight" in info and "nonmember_mia_weight" in info:
                            has_separated_weights = True
                            break
                
                if has_separated_weights:
                    # Separated format: use appropriate weight based on record type (member vs non-member)
                    for gt_idx in range(len(ground_truths)):
                        if gt_idx % 2 == 0:  # Member ground truth record
                            # Use member weight
                            info = df["extra_info"].iloc[df_idx] if df_idx < len(df) else {}
                            if isinstance(info, dict) and "member_mia_weight" in info:
                                weight = float(info["member_mia_weight"])
                            else:
                                weight = 0.0
                        else:  # Non-member ground truth record
                            # Use non-member weight
                            info = df["extra_info"].iloc[df_idx] if df_idx < len(df) else {}
                            if isinstance(info, dict) and "nonmember_mia_weight" in info:
                                weight = float(info["nonmember_mia_weight"])
                            else:
                                weight = 0.0
                            # Move to next DataFrame row after processing non-member
                            df_idx += 1
                        
                        expanded_mia_weights.append(weight)
                else:
                    # Single weight format with MIA ground truth expansion - this is a bug!
                    # If we have MIA ground truth expansion, we MUST have separated weights
                    raise ValueError(
                        "Data inconsistency detected: MIA ground truth expansion is enabled "
                        "(has_nonmember_gt=True) but only single 'mia_weight' fields found. "
                        "Expected separated 'member_mia_weight' and 'nonmember_mia_weight' fields. "
                        "This indicates a bug in the data preprocessing pipeline. "
                        "Please reprocess the data with the correct preprocessing script."
                    )
                
                if len(expanded_mia_weights) == expected_length:
                    mia_weights = expanded_mia_weights
                else:
                    mia_weights = []
                    mia_weight_tag = None
            elif len(mia_weights) != expected_length:
                # Size mismatch - disable MIA weights
                print(f"Warning: MIA weight count ({len(mia_weights)}) doesn't match record count ({expected_length})")
                mia_weights = []
                mia_weight_tag = None
    
    return ground_truths, candidates_list, splits, original_ids, problems, mia_weights, mia_weight_tag


# ---------------------------------------------------------------------------
# JSONL writer and score computation helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compute_scores(per_example: List[dict], metrics: List[str], normalize: bool) -> List[float]:
    """Compute a scalar score per example from selected per-example metrics.

    - metrics: list of keys to read from each example_result
    - normalize: if True, min-max normalise each metric to [0,1] across the dataset
                 before averaging across metrics
    """
    if len(per_example) == 0:
        return []

    # Validate metric keys exist for at least one example
    for m in metrics:
        if m not in per_example[0]:
            available = sorted(per_example[0].keys())
            raise KeyError(
                f"Metric {m!r} not found in per-example results. Available keys include: {available}"
            )

    # Collect values per metric
    values_by_metric = []
    for m in metrics:
        vals = np.array([float(ex[m]) for ex in per_example], dtype=float)
        values_by_metric.append(vals)

    if normalize:
        normed = []
        for vals in values_by_metric:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if vmax > vmin:
                normed.append((vals - vmin) / (vmax - vmin))
            else:
                # Constant metric; set all to zeros
                normed.append(np.zeros_like(vals))
        values_by_metric = normed

    # Average across metrics (axis=0) to get a single score per example
    stacked = np.stack(values_by_metric, axis=0)  # (K, N)
    scores = np.mean(stacked, axis=0)             # (N,)
    return scores.tolist()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate reconstruction quality vs ground-truth")
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to input parquet file")
    p.add_argument("--output", "-o", type=Path, required=True, help="Path to output JSON file")
    p.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar during evaluation"
    )
    p.add_argument(
        "--evaluate-math", action="store_true", help="Enable math evaluation"
    )
    p.add_argument(
        "--evaluate-bleurt", action="store_true", help="Enable BLEURT evaluation"
    )
    p.add_argument(
        "--bleurt-checkpoint",
        default="lucadiliello/BLEURT-20",
        help="BLEURT model checkpoint to use (default: lucadiliello/BLEURT-20)"
    )
    p.add_argument(
        "--bleurt-length-penalty",
        choices=["none", "ratio", "sqrt", "log"],
        default="none",
        help="Length penalty type for BLEURT (default: none)"
    )
    p.add_argument(
        "--bleurt-length-threshold",
        type=float,
        default=1.5,
        help="Length threshold for applying penalty in BLEURT (default: 1.5)"
    )
    p.add_argument(
        "--bleurt-device",
        default="cuda",
        help="Device to run BLEURT on ('cuda', 'cpu', or None for auto-detect)"
    )
    p.add_argument(
        "--mia-jsonl",
        action="store_true",
        help=(
            "If set, also write ${ATTACK}_members.jsonl and ${ATTACK}_nonmembers.jsonl "
            "with per-example scores for membership-style evaluation. Automatically "
            "expands records with both member and non-member ground truths."
        ),
    )
    p.add_argument(
        "--attack",
        nargs="+",
        default=None,
        help=(
            "Prefix names for the JSONL files when --mia-jsonl is set. Can specify multiple "
            "attacks corresponding to each metric in --score-metrics. If fewer attacks than "
            "metrics are provided, the last attack name will be used for remaining metrics. "
            "If not provided, no attack-specific JSONL files will be generated. "
            "E.g., --attack adra_emb adra_bleurt --score-metrics embedding_cosine_sim_avg bleurt_sim_avg"
        ),
    )
    p.add_argument(
        "--score-metrics",
        nargs="+",
        default=["embedding_cosine_sim_avg"],
        help=(
            "Per-example metric keys to use as score. Can pass multiple to average "
            "(optionally with --normalize-scores). Common choices: "
            "embedding_cosine_sim_avg, embedding_cosine_sim_best, lexical_jaccard_sim_avg, "
            "lexical_jaccard_sim_best, lexical_lcs_len_avg, lexical_lcs_len_best, "
            "bleurt_sim_avg, bleurt_sim_best (requires --evaluate-bleurt), "
            "llm_judge_sim_avg, llm_judge_sim_best (requires --evaluate-local-llm-judge). "
            "Use --llm-judge-prompt-template-name to specify different LLM judge prompt templates."
        ),
    )
    p.add_argument(
        "--normalize-scores",
        action="store_true",
        help=(
            "Min-max normalise each selected metric across the dataset before averaging "
            "to compute the JSONL score."
        ),
    )
    p.add_argument(
        "--embedding-model",
        choices=["fasttext", "qwen3", "qwen3-0.6B", "qwen3-4B", "qwen3-8B"],
        default="qwen3",
        help=(
            "Embedding model to use for similarity computation. Options: "
            "'fasttext' (FastText embeddings), "
            "'qwen3' or 'qwen3-0.6B' (Qwen3-Embedding-0.6B with 1024 dims), "
            "'qwen3-4B' (Qwen3-Embedding-4B with 1536 dims), "
            "'qwen3-8B' (Qwen3-Embedding-8B with 4096 dims). "
            "Default: qwen3"
        ),
    )
    p.add_argument(
        "--use-transformed-gt",
        action="store_true",
        help="Use transformed ground truths from extra_info instead of original ground truths for evaluation",
    )
    p.add_argument(
        "--evaluate-local-llm-judge",
        action="store_true",
        help="Enable local LLM judge evaluation (requires problem extraction)"
    )
    p.add_argument(
        "--llm-judge-model",
        default=DEFAULT_LLM_JUDGE_MODEL,
        help=f"HuggingFace model to use for local LLM judge (default: {DEFAULT_LLM_JUDGE_MODEL})"
    )
    p.add_argument(
        "--llm-judge-enable-thinking",
        action="store_true",
        help="Enable thinking mode for Qwen3 models (default: False for speed)"
    )
    p.add_argument(
        "--llm-judge-batch-size",
        type=int,
        default=8,
        help="Batch size for LLM judge inference (default: 8)"
    )
    p.add_argument(
        "--llm-judge-temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM judge generation (default: 0.7)"
    )
    p.add_argument(
        "--llm-judge-top-p",
        type=float,
        default=0.8,
        help="Top-p for LLM judge generation (default: 0.8)"
    )
    p.add_argument(
        "--llm-judge-max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens for LLM judge generation (default: 4096)"
    )
    p.add_argument(
        "--llm-judge-use-remote",
        action="store_true",
        help="Use remote LLM judge server instead of local inference"
    )
    p.add_argument(
        "--llm-judge-server-url",
        default=None,
        help="URL of the vLLM server for remote LLM judge (e.g., http://localhost:8000)"
    )
    p.add_argument(
        "--llm-judge-api-key",
        default=None,
        help="API key for remote LLM judge server authentication"
    )
    p.add_argument(
        "--llm-judge-timeout",
        type=float,
        default=60.0,
        help="Request timeout for remote LLM judge server (default: 60.0)"
    )
    p.add_argument(
        "--llm-judge-prompt",
        default=None,
        help="Name of the prompt template to use for LLM judge (e.g., 'V0', 'V1'). If not specified, uses default template"
    )
    p.add_argument(
        "--disable-llm-judge-batching",
        action="store_true",
        help="Disable optimized cross-problem batching for LLM judge evaluation (default: batching enabled for better performance)"
    )
    p.add_argument(
        "--mia-weights-higher-is-member",
        action="store_true",
        help=(
            "If set, higher MIA weight values indicate membership (default: False). "
            "By default, MIA weights follow the convention that LOWER values indicate "
            "membership (e.g., min-k++, loss-based metrics). When False (default), "
            "weights are inverted (1 - weight) before applying to reconstruction metrics "
            "since reconstruction metrics use higher-is-better convention."
        ),
    )
    p.add_argument(
        "--reweight-from-json",
        type=Path,
        default=None,
        help=(
            "Path to existing evaluation JSON file with per_example results. "
            "If provided, will load per-example metrics and re-aggregate with MIA weights "
            "from the input parquet, instead of re-computing all metrics. "
            "This is much faster when you already have evaluation results and just want "
            "to apply different MIA weights. The script will automatically add '_weighted' "
            "suffix to JSONL output files."
        ),
    )
    p.add_argument(
        "--prefix-ratio",
        type=float,
        default=0.0,
        help=(
            "Ratio of words to remove from the beginning of ground truths before evaluation "
            "(default: 0.0, no removal). Valid range: [0.0, 1.0). This enables evaluation of "
            "only the non-prefix portion that the model had to generate, useful when the model "
            "was given an assistant prefix during training. For example, --prefix-ratio 0.25 "
            "removes the first 25%% of words from each ground truth."
        ),
    )
    p.add_argument(
        "--budget-forcing",
        choices=["tokenizer", "whitespace"],
        default=None,
        help=(
            "Enable budget forcing to truncate candidate generations to match ground truth "
            "token count before evaluation. 'tokenizer' uses Qwen2.5-Math tokenizer for "
            "precise token counting, 'whitespace' uses simple word splitting. When enabled, "
            "only evaluates candidates up to the ground truth length."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Validate prefix_ratio
    if args.prefix_ratio < 0.0 or args.prefix_ratio >= 1.0:
        raise ValueError(f"--prefix-ratio must be in range [0.0, 1.0), got {args.prefix_ratio}")

    df = pd.read_parquet(args.input)
    print(f"Loaded parquet – rows: {len(df):,}")

    # Extract membership metadata only if requested; also returns texts/candidates, problems, and MIA weights
    ground_truths, candidates_list, splits, original_ids, problems_list, mia_weights, mia_weight_tag = _extract_text_fields(
        df, 
        need_membership=args.mia_jsonl, 
        use_transformed_gt=args.use_transformed_gt,
        extract_problems=args.evaluate_local_llm_judge,
        prefix_ratio=args.prefix_ratio
    )
    
    # Handle MIA weight inversion if weights are present
    if mia_weights:
        print(f"✅ Extracted {len(mia_weights)} MIA weights (tag: {mia_weight_tag})")
        
        # Invert MIA weights if needed (default: lower MIA score = more likely member)
        # Since reconstruction metrics use higher-is-better, we need to invert
        # Note: MIA weights are already normalized to [0, 1] during preprocessing
        if not args.mia_weights_higher_is_member:
            inverted = [1.0 - w for w in mia_weights]
            if True:  # Always print inversion info when inverting
                print(f"   🔄 Inverted MIA weights (lower-is-member → higher-weight for reconstruction)")
                print(f"   📊 Original range: [{min(mia_weights):.6f}, {max(mia_weights):.6f}]")
                print(f"   📊 Inverted range: [{min(inverted):.6f}, {max(inverted):.6f}]")
            mia_weights = inverted
        else:
            print(f"   ℹ️  Using MIA weights as-is (higher-is-member mode)")
            print(f"   📊 Weight range: [{min(mia_weights):.6f}, {max(mia_weights):.6f}]")
    else:
        print("ℹ️  No MIA weights found in data")

    # Modify output path if using transformed ground truths or prefix truncation
    output_path = args.output
    if args.use_transformed_gt:
        # Add "_transformed" before the file extension
        output_path = output_path.parent / f"{output_path.stem}_transformed{output_path.suffix}"
        print(f"Using transformed ground truths - output will be saved to: {output_path}")
    
    if args.prefix_ratio > 0.0:
        # Add "_prefixN" before the file extension (where N is percentage)
        prefix_pct = int(args.prefix_ratio * 100)
        output_path = output_path.parent / f"{output_path.stem}_prefix{prefix_pct}{output_path.suffix}"
        print(f"Using prefix truncation (ratio={args.prefix_ratio:.2f}) - output will be saved to: {output_path}")

    # Check if we should re-weight from existing JSON
    if args.reweight_from_json:
        if not args.reweight_from_json.exists():
            raise FileNotFoundError(f"Re-weight source JSON not found: {args.reweight_from_json}")
        
        if not mia_weights:
            raise ValueError(
                "Cannot re-weight from JSON: no MIA weights found in input parquet. "
                "The input parquet must contain MIA weights in extra_info."
            )
        
        print(f"\n🔄 Re-weighting mode: Loading existing results from {args.reweight_from_json}")
        print(f"   Will re-aggregate metrics using {mia_weight_tag} weights from input parquet")
        
        # Load existing results
        with open(args.reweight_from_json, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        
        if "per_example" not in existing_results:
            raise ValueError("Re-weight source JSON must contain 'per_example' results")
        
        per_example = existing_results["per_example"]
        
        # Verify lengths match
        if len(per_example) != len(mia_weights):
            raise ValueError(
                f"Length mismatch: existing results have {len(per_example)} examples "
                f"but parquet has {len(mia_weights)} MIA weights"
            )
        
        # Re-aggregate with weights AND compute weighted per-example scores
        print(f"   Re-aggregating {len(per_example)} examples with MIA weights...")
        print(f"   Computing weighted per-example scores...")
        
        # Import numpy for weighted aggregation
        weights = np.array(mia_weights)
        
        # Create weighted per-example scores by multiplying each score by its MIA weight
        # This makes high-MIA-weight examples more influential in the aggregation
        for i, ex in enumerate(per_example):
            weight = mia_weights[i]
            # Add weighted versions of all numeric metrics
            # Use list() to avoid "dictionary changed size during iteration" error
            for key, value in list(ex.items()):
                if isinstance(value, (int, float)) and not key.startswith("weighted_"):
                    # Multiply score by MIA weight
                    ex[f"weighted_{key}"] = float(value) * weight
        
        # Extract metric values from per_example results
        def extract_metric_values(metric_key):
            values = []
            for ex in per_example:
                if metric_key in ex:
                    values.append(float(ex[metric_key]))
            return values if len(values) == len(per_example) else None
        
        # Compute weighted summary
        weighted_summary = {"mia_weight_tag": mia_weight_tag}
        
        # Define metrics to aggregate
        avg_metrics = [
            "lexical_jaccard_sim_avg", "lexical_lcs_len_avg", "lexical_lcs_ratio_avg",
            "lexical_lcs_ratio_cand_avg", "embedding_cosine_sim_avg",
            "lexical_ngram_coverage_avg", "lexical_ngram_coverage_ref_avg"
        ]
        
        best_metrics = [
            "lexical_jaccard_sim_best", "lexical_lcs_len_best", "lexical_lcs_ratio_best",
            "lexical_lcs_ratio_cand_best", "embedding_cosine_sim_best",
            "lexical_ngram_coverage_best", "lexical_ngram_coverage_ref_best"
        ]
        
        # Optional metrics
        optional_metrics = [
            "bleurt_sim_avg", "bleurt_sim_best",
            "llm_judge_sim_avg", "llm_judge_sim_best",
            "math_avg_at_k", "math_pass_at_k",
            "best_candidate_lexical_accuracy", "best_candidate_embedding_accuracy",
            "best_candidate_bleurt_accuracy", "best_candidate_llm_judge_accuracy"
        ]
        
        # Aggregate avg metrics
        for metric in avg_metrics:
            values = extract_metric_values(metric)
            if values:
                weighted_summary[f"weighted_{metric}"] = float(np.average(values, weights=weights))
        
        # Aggregate best metrics  
        for metric in best_metrics:
            values = extract_metric_values(metric)
            if values:
                weighted_summary[f"weighted_{metric}_mean"] = float(np.average(values, weights=weights))
        
        # Aggregate optional metrics
        for metric in optional_metrics:
            values = extract_metric_values(metric)
            if values:
                if metric.endswith("_avg") or metric.endswith("_best"):
                    weighted_summary[f"weighted_{metric}"] = float(np.average(values, weights=weights))
                else:
                    # Math metrics get _mean suffix
                    weighted_summary[f"weighted_{metric}_mean"] = float(np.average(values, weights=weights))
        
        # Combine with existing results
        results = existing_results.copy()
        results["weighted_summary"] = weighted_summary
        
        print(f"   ✅ Computed {len(weighted_summary)} weighted metrics")
        
        # Save the weighted results (main JSON + member/nonmember JSON if requested)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_json(
            results,
            str(output_path),
            split_by_membership=args.mia_jsonl  # Write weighted member/nonmember JSONs
        )
        print(f"Saved weighted evaluation results to {output_path}")
        
    # If the evaluation JSON already exists, load and reuse it.
    elif output_path.exists():
        print(f"Detected existing evaluation JSON at {output_path}; loading results and skipping recompute.")
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        # Prepare BLEURT configuration
        bleurt_kwargs = None
        if args.evaluate_bleurt:
            bleurt_kwargs = {
                "bleurt_checkpoint": args.bleurt_checkpoint,
                "length_penalty": args.bleurt_length_penalty,
                "length_threshold": args.bleurt_length_threshold,
            }
            if args.bleurt_device:
                bleurt_kwargs["device"] = args.bleurt_device

        # Prepare LLM judge configuration (local or remote)
        llm_judge_kwargs = None
        if args.evaluate_local_llm_judge:
            llm_judge_kwargs = {
                "model_name": args.llm_judge_model,
                "enable_thinking": args.llm_judge_enable_thinking,
                "batch_size": args.llm_judge_batch_size,
                "temperature": args.llm_judge_temperature,
                "top_p": args.llm_judge_top_p,
                "max_new_tokens": args.llm_judge_max_new_tokens,
                "use_remote": args.llm_judge_use_remote,
            }
            
            # Add prompt template if specified
            if args.llm_judge_prompt:
                llm_judge_kwargs["prompt_template_name"] = args.llm_judge_prompt
            
            # Add remote-specific configuration
            if args.llm_judge_use_remote or args.llm_judge_server_url:
                llm_judge_kwargs.update({
                    "use_remote": True,
                    "server_url": args.llm_judge_server_url,
                    "api_key": args.llm_judge_api_key,
                    "timeout": args.llm_judge_timeout,
                })

        results = evaluate_dataset(
            ground_truths,
            candidates_list,
            verbose=not args.no_progress,
            evaluate_math=args.evaluate_math,
            evaluate_bleurt=args.evaluate_bleurt,
            bleurt_kwargs=bleurt_kwargs,
            evaluate_local_llm_judge=args.evaluate_local_llm_judge,
            llm_judge_kwargs=llm_judge_kwargs,
            problems_list=problems_list,
            embedding_model=args.embedding_model,
            enable_llm_judge_batching=not args.disable_llm_judge_batching,
            mia_weights=mia_weights if mia_weights else None,
            mia_weight_tag=mia_weight_tag,
            mia_weights_higher_is_member=args.mia_weights_higher_is_member,
            budget_forcing=args.budget_forcing,
        )

        # Add membership information to results if available
        if args.mia_jsonl and splits:
            for i, example in enumerate(results["per_example"]):
                if i < len(splits) and i < len(original_ids):
                    example["extra_info"] = {
                        "split": splits[i],
                        "id": original_ids[i]
                    }

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_json(
            results, 
            str(output_path),
            split_by_membership=args.mia_jsonl  # Write eval JSONLs if MIA is enabled
        )
        print(f"Saved evaluation results to {output_path}")
    
    # Save weighted results separately if available (from re-weighting mode)
    if "weighted_summary" in results:
        weighted_output_path = output_path.parent / f"{output_path.stem}_weighted{output_path.suffix}"
        weighted_results = {
            "per_example": results["per_example"],
            "dataset_summary": results["dataset_summary"],
            "weighted_summary": results["weighted_summary"]
        }
        with open(weighted_output_path, "w", encoding="utf-8") as f:
            json.dump(weighted_results, f, ensure_ascii=False, indent=2)
        print(f"Saved weighted evaluation results to {weighted_output_path}")

    # Optionally write MIA-style JSONL outputs
    if args.mia_jsonl and args.attack is not None:
        per_example = results.get("per_example", [])
        
        # Determine if we're in re-weighting mode (affects metric naming and filenames)
        is_reweight_mode = args.reweight_from_json is not None
        
        # Process each metric with its corresponding attack name
        for metric_idx, metric in enumerate(args.score_metrics):
            # Get the attack name for this metric
            if metric_idx < len(args.attack):
                attack_name = args.attack[metric_idx]
            else:
                # Use the last attack name for remaining metrics
                attack_name = args.attack[-1]
            
            # In re-weight mode, use the metric name as-is (with "weighted_" prefix)
            # to extract the weighted per-example scores
            actual_metric = metric
            
            # Compute scores for this specific metric
            try:
                scores = _compute_scores(per_example, metrics=[actual_metric], normalize=args.normalize_scores)
            except KeyError as e:
                print(f"Warning: Metric '{actual_metric}' not found in per-example results. Skipping attack '{attack_name}'.")
                print(f"   Original metric name: '{metric}'")
                print(f"   Available metrics: {list(per_example[0].keys()) if per_example else 'N/A'}")
                continue
            
            if len(scores) != len(splits) or len(scores) != len(original_ids):
                raise RuntimeError(f"Length mismatch when preparing JSONL outputs for metric {metric}.")

            members_rows: List[dict] = []
            nonmembers_rows: List[dict] = []

            for i, (split, oid, score) in enumerate(zip(splits, original_ids, scores)):
                row = {"id": int(oid), "score": float(score)}
                if split == "train":
                    row["idx"] = len(members_rows)
                    members_rows.append(row)
                elif split == "test":
                    row["idx"] = len(nonmembers_rows)
                    nonmembers_rows.append(row)
                else:
                    # Should not happen due to earlier validation
                    continue

            # Use output_path for consistent naming with transformed suffix if applicable
            # Add "weighted" suffix if we're in re-weight mode
            suffix = ""
            if args.use_transformed_gt:
                suffix = "_transformed"
            if is_reweight_mode:
                suffix += "_weighted"
            
            # Truncate filename if too long to avoid "File name too long" error
            # Linux filename limit is typically 255 bytes
            max_name_length = 200
            full_name = f"{attack_name}{suffix}"
            if len(full_name) > max_name_length:
                # Keep suffix if present, truncate attack name
                if suffix:
                    max_attack_len = max_name_length - len(suffix)
                    full_name = f"{attack_name[:max_attack_len]}{suffix}"
                else:
                    full_name = attack_name[:max_name_length]
            
            members_path = output_path.parent / f"{full_name}_members.jsonl"
            nonmembers_path = output_path.parent / f"{full_name}_nonmembers.jsonl"
            
            _write_jsonl(members_path, members_rows)
            _write_jsonl(nonmembers_path, nonmembers_rows)
            print(f"Wrote JSONL files for {attack_name} (metric: {actual_metric}): {members_path}, {nonmembers_path}")


if __name__ == "__main__":
    main() 