"""
Prompt templates for LLM judge evaluation.

This module contains prompt templates for evaluating solution similarity
using LLM-as-a-judge. Templates are stored as simple string constants and can be
accessed by name or used directly.
"""

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================


# -----------------------------------------------------------------------------
# V1 - Similarity rating with evaluation criteria (includes problem statement)
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V1 = """
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

EVALUATION CRITERIA:
1. Mathematical correctness - Are candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
4. Overall clarity - Are the reasoning and solution steps correct, sonsistent, and logically sound as the reference solution?


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


# -----------------------------------------------------------------------------
# V2.1 - Comprehensive evaluation with surface & semantic resemblance
#         (no problem statement; explicit "Output ONLY one line" instruction)
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V2_1 = """TASK DESCRIPTION
Rate two solutions to the same math problem (one reference, one candidate) for similarity. The final answer in each solution is enclosed in "\\boxed{{}}". Return a real-valued score between 0 and 1 with exactly 3 decimals.

EVALUATION CRITERIA (consider all, equally)
1. Mathematical correctness — Is the candidate mathematically sound as in the reference?
2. Solution approach — Does the candidate use methods/reasoning similar to the reference?
3. Final answer — Does the candidate arrive at the same \\boxed{{}} answer as the reference?
4. Overall clarity — Are the candidate's reasoning and steps as consistent and logically sound as the reference?
5. Surface & semantic resemblance — Do the two solutions look very similar in wording, symbols, step order, and intermediate expressions (i.e., high lexical/token overlap and near-paraphrase semantics)? Penalize added filler or unnecessary rephrasing.

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()


# -----------------------------------------------------------------------------
# V3 - Scoring with hard disqualifiers and additive scoring
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V3 = """TASK
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,1] with 3 decimals.

HARD DISQUALIFIER (apply BEFORE scoring; if true, output 0.000):
A) The candidate merely restates or paraphrases the problem statement, or provides no substantive solution steps (i.e., not a solution).
If A is true → REWARD: 0.000

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content. Ignore any text that reproduces the problem statement or generic headings/boilerplate (e.g., "Problem:", "### Problem", restated prompt).
- Do not give credit for tokens/phrases that appear in the problem statement; similarity must come from solution reasoning/derivations/explanations and final answer.

SCORING (apply ONLY if no disqualifier triggered)
Compute an additive score ∈ [0, 1.0] using the five criteria below; each criterion is worth 0.20. Round the final result to 3 decimals.

1) Mathematical correctness (0.20)
   - Are the candidate's steps mathematically valid and consistent with the GROUND TRUTH's reasoning (beyond just the final answer)?

2) Solution approach similarity (0.20)
   - Does the candidate use comparable methods/transformations to the GROUND TRUTH (e.g., same identities, substitutions, case structure, combinatorial arguments)?

3) Lexical/token overlap of solution content (0.20)
   - Consider only solution text (exclude problem text/boilerplate). Higher score for close phrase/term/token overlap and similar ordering.

4) Length similarity (0.20)
   - Compare solution word counts (exclude problem text/boilerplate). Full credit if within ±15% of GROUND TRUTH; otherwise decrease proportionally with distance.

5) Final answer match (0.20)
   - If the candidate presents a clear final answer that exactly matches the GROUND TRUTH (e.g., same value in \\boxed{{}} when present): full credit (0.20).
   - If a clear final answer is present but does NOT match: 0.00 for this criterion.
   - If no clear final answer is presented: 0.00 for this criterion.

FINAL OUTPUT FORMAT (must follow exactly; no extra words, no reasoning)
REWARD: <number between 0 and 1 with 3 decimals>

INPUTS
- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


# =============================================================================
# TEMPLATE REGISTRY AND UTILITY FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Template name mapping - All available prompt templates
# -----------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "default": PROMPT_TEMPLATE_V1,
    "v1": PROMPT_TEMPLATE_V1,
    "v2_1": PROMPT_TEMPLATE_V2_1,
    "v3": PROMPT_TEMPLATE_V3,
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_prompt_template(template_name: str) -> str:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template ("default", "v1", "v2_1", "v3")
        
    Returns:
        The prompt template string
        
    Raises:
        ValueError: If template name is not found
    """
    if template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name]
    else:
        available_templates = list(PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown prompt template: '{template_name}'. "
            f"Available templates: {available_templates}"
        )


def get_default_template() -> str:
    """Get the default prompt template."""
    return PROMPT_TEMPLATES["default"]


def list_available_templates() -> list:
    """List all available template names."""
    return list(PROMPT_TEMPLATES.keys())


# =============================================================================
# TEMPLATE SUMMARY
# =============================================================================
# V1:   Basic similarity with evaluation criteria; includes {PROBLEM}
# V2.1: Comprehensive 5-criteria evaluation with surface/semantic resemblance;
#        no {PROBLEM}; explicit "Output ONLY one line"
# V3:   Additive 5-criteria rubric with hard disqualifier; no {PROBLEM}
# =============================================================================
