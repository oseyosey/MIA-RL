"""
N-gram coverage computation for reconstruction evaluation.

This module provides a simplified implementation of n-gram coverage metric
extracted from the NGramCoverageAttack implementation, adapted for use
in reconstruction evaluation pipelines.

Performance optimization: Accepts tokenizer function from caller to avoid
redundant tokenization and improve speed. Falls back to simple regex-based 
tokenizer if no tokenizer is provided.

Environment Variables:
    ADRA_USE_DYNAMIC_MAX_NGRAM: If set to "1", "true", "yes", or "on", enables
        dynamic max_ngram calculation in compute_unique_ngram_coverage. This
        finds the longest actual match and only builds n-grams up to that length,
        providing significant performance improvements (50-90% fewer n-grams).
"""

from typing import List, Tuple, Literal, Callable, Optional
import re
import os

# Precompile regex pattern for performance (avoid recompilation on every call)
_TOKENIZE_PATTERN = re.compile(r"\\[a-zA-Z]+(?:\{[^}]*\})*|\d+\.?\d*|[a-zA-Z_]\w*|[+\-*/=<>!]=?|[(){}\[\]]|\S")

# N-gram search limits (defaults)
DEFAULT_MIN_NGRAM = 3
DEFAULT_MAX_NGRAM = 30

# Unicode ranges for scripts that typically don't use spaces between words
# These scripts require character-level tokenization for meaningful n-gram matching
_NON_SPACE_DELIMITED_RANGES = [
    # CJK (Chinese, Japanese Kanji, Korean Hanja)
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    
    # Japanese Kana (Hiragana, Katakana)
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    
    # Korean Hangul
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
    
    # Southeast Asian scripts (no spaces between words)
    (0x0E00, 0x0E7F),    # Thai
    (0x0E80, 0x0EFF),    # Lao
    (0x1780, 0x17FF),    # Khmer
    (0x1000, 0x109F),    # Myanmar
    
    # Tibetan
    (0x0F00, 0x0FFF),    # Tibetan
]


def _is_non_space_delimited_char(char: str) -> bool:
    """Check if a character belongs to a non-space-delimited script.
    
    Performance: Fast paths for common scripts (ASCII, Latin, Cyrillic, Arabic, Indic)
    avoid checking 15+ Unicode ranges. This gives ~50-100x speedup for English text.
    """
    code_point = ord(char)
    
    # Fast path: ASCII characters are always space-delimited (English, basic punctuation)
    # This covers 99%+ of WildChat and most Western text
    if code_point < 0x0080:  # ASCII range (0-127)
        return False
    
    # Fast path: Common Latin Extended ranges (European languages with diacritics)
    if 0x0080 <= code_point <= 0x024F:
        return False
    
    # Fast path: Cyrillic (Russian, Ukrainian, etc.)
    if 0x0400 <= code_point <= 0x04FF:
        return False
    
    # Fast path: Arabic and Hebrew (space-delimited RTL scripts)
    if 0x0590 <= code_point <= 0x06FF:
        return False
    
    # Fast path: Devanagari and other Indic scripts (Hindi, Bengali, etc.)
    if 0x0900 <= code_point <= 0x0DFF:
        return False
    
    # Now check non-space-delimited scripts (CJK, Thai, etc.)
    for start, end in _NON_SPACE_DELIMITED_RANGES:
        if start <= code_point <= end:
            return True
    return False


def _default_tokenize(text: str, max_tokens: int | None = None) -> List[str]:
    """Default fallback tokenizer with multilingual support.
    
    Uses a hybrid approach:
    - For languages that use spaces: word-level tokenization  
    - For languages without spaces (CJK, Thai, etc.): character-level tokenization
    - For mixed text: intelligently combines both approaches
    
    Only used if caller doesn't provide a tokenizer function.
    """
    text_lower = text.lower()
    segments = text_lower.split()
    
    tokens = []
    for segment in segments:
        if not segment:
            continue
            
        # Check if segment contains any non-space-delimited characters
        has_non_space_script = any(_is_non_space_delimited_char(c) for c in segment)
        
        if not has_non_space_script:
            # Pure space-delimited script - keep as single token
            cleaned = segment.strip('.,;:!?')
            if cleaned:
                tokens.append(cleaned)
        else:
            # Contains non-space-delimited characters - hybrid tokenization
            current_run = []
            current_is_non_space = None
            
            for char in segment:
                char_is_non_space = _is_non_space_delimited_char(char)
                
                if current_is_non_space is None:
                    current_is_non_space = char_is_non_space
                
                if char_is_non_space == current_is_non_space:
                    current_run.append(char)
                else:
                    # Flush current run
                    if current_run:
                        if current_is_non_space:
                            tokens.extend(current_run)
                        else:
                            word = ''.join(current_run).strip('.,;:!?')
                            if word:
                                tokens.append(word)
                    current_run = [char]
                    current_is_non_space = char_is_non_space
            
            # Flush final run
            if current_run:
                if current_is_non_space:
                    tokens.extend(current_run)
                else:
                    word = ''.join(current_run).strip('.,;:!?')
                    if word:
                        tokens.append(word)
    
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    return tokens


def _build_ngram_set(tokens: List[str], min_ngram: int, max_ngram: int) -> set:
    """
    Build a set of n-grams from tokens.
    
    Args:
        tokens: Token list to build n-grams from
        min_ngram: Minimum n-gram size
        max_ngram: Maximum n-gram size
        
    Returns:
        Set of n-gram tuples
    """
    ngrams = set()
    max_n = min(len(tokens) + 1, max_ngram + 1)
    for n in range(min_ngram, max_n):
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.add(ngram)
    return ngrams


def _find_ngram_matches(candidate_tokens: List[str], 
                       reference_tokens: List[str], 
                       min_ngram: int = 5,
                       max_ngram: int = DEFAULT_MAX_NGRAM) -> List[Tuple[int, int]]:
    """
    Find all n-gram matches between candidate and reference texts.
    
    This is a convenience wrapper that builds reference n-grams and then uses
    the optimized matching function. For batch processing with multiple candidates,
    consider precomputing n-grams and using _find_ngram_matches_optimized directly.
    
    Args:
        candidate_tokens: Tokens from candidate text (where matches are found)
        reference_tokens: Tokens from reference text (used to build n-gram set)
        min_ngram: Minimum n-gram size
        max_ngram: Maximum n-gram size (for performance optimization)
        
    Returns:
        List of (start_index, end_index) tuples for matched spans in candidate
    """
    if len(candidate_tokens) < min_ngram or len(reference_tokens) < min_ngram:
        return []
    
    # Build reference n-grams and delegate to optimized function
    reference_ngrams = _build_ngram_set(reference_tokens, min_ngram, max_ngram)
    return _find_ngram_matches_optimized(candidate_tokens, reference_ngrams, min_ngram, max_ngram)


def compute_ngram_coverage(
    candidate: str, 
    reference: str, 
    min_ngram: int = DEFAULT_MIN_NGRAM,
    normalize_by: Literal["candidate", "reference"] = "candidate",
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    max_ngram: int = DEFAULT_MAX_NGRAM,
) -> float:
    """
    Compute n-gram coverage between candidate and reference texts.
    
    The coverage measures the fraction of tokens that are covered by matching
    n-grams from the reference text. By default, it's normalized by candidate
    length (what fraction of the candidate is covered), but can optionally be
    normalized by reference length.
    
    Args:
        candidate: Candidate text to analyze
        reference: Reference text to compare against
        min_ngram: Minimum n-gram size (default: 3)
        normalize_by: Normalization method (default: "candidate")
            - "candidate": Normalize by candidate length (default)
            - "reference": Normalize by reference length
        tokenizer: Optional tokenizer function. If None, uses default tokenizer.
            Function should take text string and return list of tokens.
        
    Returns:
        Coverage score between 0 and 1
        - If normalize_by="candidate": fraction of candidate covered by reference
        - If normalize_by="reference": fraction of reference covered by candidate
    """
    # Use provided tokenizer or fall back to default
    tokenize_fn = tokenizer if tokenizer is not None else _default_tokenize
    
    # Tokenize both texts
    candidate_tokens = tokenize_fn(candidate)
    reference_tokens = tokenize_fn(reference)
    
    if not candidate_tokens or not reference_tokens:
        return 0.0
    
    # Calculate coverage based on normalization target
    if normalize_by == "candidate":
        # Find which candidate positions are covered by reference n-grams
        matches = _find_ngram_matches(candidate_tokens, reference_tokens, min_ngram, max_ngram)
        
        if not matches:
            return 0.0
        
        covered_positions = set()
        for start, end in matches:
            for pos in range(start, end):
                covered_positions.add(pos)
        
        coverage = len(covered_positions) / len(candidate_tokens)
    
    elif normalize_by == "reference":
        # Find which reference positions are covered by candidate n-grams
        # Swap arguments to find matches in reference instead of candidate
        matches = _find_ngram_matches(reference_tokens, candidate_tokens, min_ngram, max_ngram)
        
        if not matches:
            return 0.0
        
        covered_positions = set()
        for start, end in matches:
            for pos in range(start, end):
                covered_positions.add(pos)
        
        coverage = len(covered_positions) / len(reference_tokens)
    
    else:
        raise ValueError(f"Invalid normalize_by value: {normalize_by}. Must be 'candidate' or 'reference'.")
    
    return coverage


def compute_unique_ngram_coverage(
    candidate: str, 
    reference: str, 
    min_ngram: int = DEFAULT_MIN_NGRAM,
    normalize_by: Literal["candidate", "reference"] = "candidate",
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    max_ngram: int = DEFAULT_MAX_NGRAM,
    use_longest_match_only: bool = False,
) -> float:
    """
    Compute unique n-gram coverage (set-based, no repetition counting).
    
    This metric differs from compute_ngram_coverage by using set intersection
    instead of position-based matching. This naturally penalizes repetition
    because repeated phrases only count once in the set.
    
    Args:
        candidate: Candidate text to analyze
        reference: Reference text to compare against
        min_ngram: Minimum n-gram size (default: 3)
        normalize_by: Normalization method (default: "candidate")
            - "candidate": Normalize by candidate unique n-grams
            - "reference": Normalize by reference unique n-grams
        tokenizer: Optional tokenizer function. If None, uses default tokenizer.
            Function should take text string and return list of tokens.
        max_ngram: Maximum n-gram size (default: 30)
        use_longest_match_only: If True, dynamically determine max_ngram based on the
            longest actual match found, instead of using the fixed max_ngram cap (default: False)
            - False: Builds all n-grams from min_ngram to max_ngram (current behavior)
              Example: max_ngram=30 → builds 3-gram through 30-gram (even if longest match is 10)
            - True: Finds longest match first, then builds n-grams only up to that length
              Example: longest match=15 → builds 3-gram through 15-gram only
            This avoids wasting computation on n-grams that will never match, providing
            significant efficiency gains when actual matches are shorter than max_ngram.
            Can also be controlled globally via ADRA_USE_DYNAMIC_MAX_NGRAM environment variable.
        
    Returns:
        Coverage score between 0 and 1
        - If normalize_by="candidate": fraction of candidate unique n-grams found in reference
        - If normalize_by="reference": fraction of reference unique n-grams found in candidate
        
    Example:
        >>> # Repetition test
        >>> reference = "solve for x by subtracting 2"
        >>> non_repetitive = "solve for x by adding 1"
        >>> repetitive = "solve for x solve for x solve for x"
        >>> # repetitive will score lower than non_repetitive despite having more total n-grams
    """
    # Check environment variable to enable dynamic max_ngram globally
    if not use_longest_match_only:
        env_value = os.environ.get("ADRA_USE_DYNAMIC_MAX_NGRAM", "").strip().lower()
        use_longest_match_only = env_value in {"1", "true", "yes", "on"}
    
    # Use provided tokenizer or fall back to default
    tokenize_fn = tokenizer if tokenizer is not None else _default_tokenize
    
    # Tokenize both texts
    candidate_tokens = tokenize_fn(candidate)
    reference_tokens = tokenize_fn(reference)
    
    if not candidate_tokens or not reference_tokens:
        return 0.0
    
    # Determine effective max_ngram (conditional based on use_longest_match_only)
    if use_longest_match_only:
        # Find longest match in both directions to determine effective cap
        cand_longest = _find_longest_match_length(
            candidate_tokens, reference_tokens, min_ngram, max_ngram
        )
        ref_longest = _find_longest_match_length(
            reference_tokens, candidate_tokens, min_ngram, max_ngram
        )
        # Use the maximum of the two to ensure we capture all matches
        effective_max_ngram = max(cand_longest, ref_longest)
    else:
        # Use the fixed max_ngram (current behavior)
        effective_max_ngram = max_ngram
    
    # Build unique n-gram sets with effective max_ngram
    candidate_ngrams = _build_ngram_set(candidate_tokens, min_ngram, effective_max_ngram)
    reference_ngrams = _build_ngram_set(reference_tokens, min_ngram, effective_max_ngram)
    
    if not candidate_ngrams or not reference_ngrams:
        return 0.0
    
    # Compute intersection (unique matching n-grams)
    intersection = candidate_ngrams & reference_ngrams
    
    # Normalize based on preference
    if normalize_by == "candidate":
        return len(intersection) / len(candidate_ngrams)
    elif normalize_by == "reference":
        return len(intersection) / len(reference_ngrams)
    else:
        raise ValueError(f"Invalid normalize_by value: {normalize_by}. Must be 'candidate' or 'reference'.")


def compute_ngram_coverage_batch(
    candidates: List[str],
    reference: str,
    min_ngram: int = DEFAULT_MIN_NGRAM,
    normalize_by: Literal["candidate", "reference"] = "candidate",
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    max_ngram: int = DEFAULT_MAX_NGRAM,
) -> List[float]:
    """
    Compute n-gram coverage for multiple candidates against one reference (batched).
    
    This is more efficient than calling compute_ngram_coverage repeatedly because:
    1. Reference is tokenized only once
    2. Reference n-gram set is built only once
    3. Batch tokenization can be optimized
    
    Args:
        candidates: List of candidate texts to analyze
        reference: Reference text to compare against (shared across all candidates)
        min_ngram: Minimum n-gram size (default: 3)
        normalize_by: Normalization method (default: "candidate")
        tokenizer: Optional tokenizer function. If None, uses default tokenizer.
            Function should take text string and return list of tokens.
        
    Returns:
        List of coverage scores (one per candidate)
    """
    if not candidates:
        return []
    
    # Use provided tokenizer or fall back to default
    tokenize_fn = tokenizer if tokenizer is not None else _default_tokenize
    
    # Tokenize reference once
    reference_tokens = tokenize_fn(reference)
    if not reference_tokens:
        return [0.0] * len(candidates)
    
    # Process based on normalization target
    if normalize_by == "candidate":
        # Optimized path: precompute reference n-grams, find matches in candidates
        reference_ngrams = _build_ngram_set(reference_tokens, min_ngram, max_ngram)
        
        if not reference_ngrams:
            return [0.0] * len(candidates)
        
        # Process each candidate using the precomputed reference n-grams
        coverages = []
        for candidate in candidates:
            candidate_tokens = tokenize_fn(candidate)
            
            if not candidate_tokens:
                coverages.append(0.0)
                continue
            
            # Find matches in candidate using precomputed reference n-grams
            matches = _find_ngram_matches_optimized(
                candidate_tokens, reference_ngrams, min_ngram, max_ngram
            )
            
            if not matches:
                coverages.append(0.0)
                continue
            
            # Calculate coverage: which candidate positions are covered
            covered_positions = set()
            for start, end in matches:
                for pos in range(start, end):
                    covered_positions.add(pos)
            
            coverage = len(covered_positions) / len(candidate_tokens)
            coverages.append(coverage)
        
        return coverages
    
    elif normalize_by == "reference":
        # Optimized path: precompute reference tokenization, find matches in reference for each candidate
        coverages = []
        for candidate in candidates:
            candidate_tokens = tokenize_fn(candidate)
            
            if not candidate_tokens:
                coverages.append(0.0)
                continue
            
            # Find which reference positions are covered by candidate n-grams
            # Swap arguments to find matches in reference instead of candidate
            matches = _find_ngram_matches(reference_tokens, candidate_tokens, min_ngram, max_ngram)
            
            if not matches:
                coverages.append(0.0)
                continue
            
            # Calculate coverage: which reference positions are covered
            covered_positions = set()
            for start, end in matches:
                for pos in range(start, end):
                    covered_positions.add(pos)
            
            coverage = len(covered_positions) / len(reference_tokens)
            coverages.append(coverage)
        
        return coverages
    
    else:
        raise ValueError(f"Invalid normalize_by value: {normalize_by}")


def _find_ngram_matches_optimized(
    candidate_tokens: List[str],
    reference_ngrams: set,
    min_ngram: int,
    max_ngram: int,
) -> List[Tuple[int, int]]:
    """
    Find n-gram matches using a precomputed reference n-gram set (optimized).
    
    This version accepts precomputed n-grams to avoid rebuilding the set for each candidate.
    
    Args:
        candidate_tokens: Tokens from candidate text
        reference_ngrams: Precomputed set of reference n-grams (tuples)
        min_ngram: Minimum n-gram size
        
    Returns:
        List of (start_index, end_index) tuples for matched spans
    """
    if len(candidate_tokens) < min_ngram:
        return []
    
    # Find matches using a greedy approach
    matches = []
    i = 0
    
    n = len(candidate_tokens)
    while i < n:
        # Try to find the longest match starting at position i
        longest_match_end = i
        start = i + min_ngram
        if start > n:
            break
        # Limit the search window to avoid quadratic blow-up
        end_limit = min(i + max_ngram + 1, n + 1)

        for j in range(start, end_limit):
            candidate_ngram = tuple(candidate_tokens[i:j])
            if candidate_ngram in reference_ngrams:
                longest_match_end = j
            elif longest_match_end > i:
                # Once we found a match and current one failed, stop extending
                break

        if longest_match_end > i:
            # Found a match
            matches.append((i, longest_match_end))
            i = longest_match_end  # Move past the matched region
        else:
            i += 1  # No match found, move to next position
    
    return matches


def _find_longest_match_length(
    tokens: List[str],
    reference_tokens: List[str],
    min_ngram: int,
    max_ngram: int,
) -> int:
    """
    Find the length of the longest n-gram match between tokens and reference.
    
    This is used to dynamically determine the effective max_ngram cap based on
    actual data, avoiding unnecessary computation of n-grams that will never match.
    
    Args:
        tokens: Token list to search for matches in
        reference_tokens: Reference tokens to match against
        min_ngram: Minimum n-gram size
        max_ngram: Maximum n-gram size to search up to
        
    Returns:
        Length of the longest matching n-gram, or min_ngram if no matches found
        
    Example:
        tokens = ["solve", "for", "x", "by", "subtracting", "two"]
        reference_tokens = ["solve", "for", "x", "by", "adding", "one"]
        # Finds longest match: "solve for x by" (4-gram)
        # Returns: 4
        # Then can build n-grams from 3-4 instead of 3-30
    """
    if len(tokens) < min_ngram or len(reference_tokens) < min_ngram:
        return min_ngram
    
    # Build reference n-grams for matching
    reference_ngrams = _build_ngram_set(reference_tokens, min_ngram, max_ngram)
    
    # Find longest non-overlapping matches
    matches = _find_ngram_matches_optimized(tokens, reference_ngrams, min_ngram, max_ngram)
    
    # Find the maximum match length
    if not matches:
        return min_ngram
    
    max_match_length = max(end - start for start, end in matches)
    return max_match_length
