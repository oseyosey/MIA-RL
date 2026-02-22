#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Run n-sampling generation followed by MIA evaluation.
#   1. Generate samples from a model using vLLM
#   2. Evaluate generated samples with lexical MIA metrics
#   3. Extract AUROC scores
# ------------------------------------------------------------------------------

set -euo pipefail
set -x

# Ensure CUDA toolkit (nvcc) is available on compute nodes.
# Uncomment this section if using adra-v1 environment
# module load gcc/13.4.0
# module load cuda/12.9.1
# export CUDA_HOME="/gpfs/software/cuda/12.9.1"
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# ------------------------------------------------------------------------------
# User-tweakable arguments
# ------------------------------------------------------------------------------

# TODO: Set the base path to the root of this repository
BASE_PATH=""

# TODO: Set path to the prepared training data parquet file
# e.g. data_path="${BASE_PATH}/data/tulu3-wildchat_rl/tulu3-wildchat_rl_lexical_unique_ngram_coverage_ref_ratio_1.50_augment_random_7_prefix_0.25/train.parquet"
data_path=""

# TODO: Set path to the model to evaluate (SFT or RL checkpoint)
model_dir="allenai/Llama-3.1-Tulu-3-8B"

# TODO: Set the output directory name for evaluation artifacts
eval_model_dir="tulu3-8b_tulu3-wildchat_n-sampling"

n_gpus=8

# Generation hyper-parameters
temperature=1.0
top_p=0.95
top_k=50
n_samples=32

PROMPT_LENGTH=1024
RESPONSE_LENGTH=512

# Output artifacts
save_path="${BASE_PATH}/eval/${eval_model_dir}/generations_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25.parquet"
eval_match_json="${BASE_PATH}/eval/${eval_model_dir}/mia_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25_match.json"
eval_json="${BASE_PATH}/eval/${eval_model_dir}/mia_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25.json"

# ------------------------------------------------------------------------------
# 1. Generate samples
# ------------------------------------------------------------------------------
unset ROCR_VISIBLE_DEVICES || true

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${n_gpus} \
    data.assistant_prefix_key=assistant_prefix \
    data.batch_size=512 \
    data.path="$data_path" \
    data.prompt_key=prompt \
    data.n_samples=$n_samples \
    data.output_path="$save_path" \
    model.path="$model_dir" \
    +model.trust_remote_code=True \
    rollout.temperature=$temperature \
    rollout.top_k=$top_k \
    rollout.top_p=$top_p \
    rollout.prompt_length=${PROMPT_LENGTH} \
    rollout.response_length=${RESPONSE_LENGTH} \
    rollout.tensor_model_parallel_size=${n_gpus} \
    rollout.gpu_memory_utilization=0.95 \
    ray_init.num_cpus=80

# ------------------------------------------------------------------------------
# 2. Lexical MIA metrics evaluation (match evaluation)
# ------------------------------------------------------------------------------

# Jaccard Similarity & token overlap
attacks=("adra_lexical_jaccard_sim_avg" "adra_lexical_jaccard_sim_best" "adra_lexical_token_overlap_ref_avg" "adra_lexical_token_overlap_ref_best" "adra_lexical_token_overlap_cand_avg" "adra_lexical_token_overlap_cand_best")
metrics=("lexical_jaccard_sim_avg" "lexical_jaccard_sim_best" "lexical_token_overlap_ref_avg" "lexical_token_overlap_ref_best" "lexical_token_overlap_cand_avg" "lexical_token_overlap_cand_best")

# LCS (Longest Common Subsequence)
attacks+=("adra_lexical_lcs_avg" "adra_lexical_lcs_best" "adra_lexical_lcs_ratio_avg" "adra_lexical_lcs_ratio_best" "adra_lexical_lcs_ratio_cand_avg" "adra_lexical_lcs_ratio_cand_best")
metrics+=("lexical_lcs_len_avg" "lexical_lcs_len_best" "lexical_lcs_ratio_avg" "lexical_lcs_ratio_best" "lexical_lcs_ratio_cand_avg" "lexical_lcs_ratio_cand_best")

# N-gram Coverage
attacks+=("adra_lexical_ngram_coverage_avg" "adra_lexical_ngram_coverage_best" "adra_lexical_ngram_coverage_ref_avg" "adra_lexical_ngram_coverage_ref_best")
metrics+=("lexical_ngram_coverage_avg" "lexical_ngram_coverage_best" "lexical_ngram_coverage_ref_avg" "lexical_ngram_coverage_ref_best")

# Embedding Cosine Similarity
attacks+=("adra_q3_8b_embedding_cosine_sim_avg" "adra_q3_8b_embedding_cosine_sim_best")
metrics+=("embedding_cosine_sim_avg" "embedding_cosine_sim_best")

bash ${BASE_PATH}/scripts/eval/run_evaluation.sh \
  "$save_path" \
  "$eval_match_json" \
  --mia-jsonl \
  --attack "${attacks[@]}" \
  --score-metrics "${metrics[@]}" \
  --prefix-ratio 0.25

for i in "${!attacks[@]}"; do
  attack="${attacks[$i]}"
  python3 -m adra.scripts.evaluate_mia \
    --members "${BASE_PATH}/eval/${eval_model_dir}/${attack}_members.jsonl" \
    --nonmembers "${BASE_PATH}/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
    --output "${BASE_PATH}/eval/${eval_model_dir}/${attack}_metrics.json" \
    --higher-is-member
done

# ------------------------------------------------------------------------------
# 3. Full suffix metrics evaluation
# ------------------------------------------------------------------------------

attacks_orig=("adra_lexical_jaccard_sim_avg_original" "adra_lexical_jaccard_sim_best_original" "adra_lexical_token_overlap_ref_avg_original" "adra_lexical_token_overlap_ref_best_original" "adra_lexical_token_overlap_cand_avg_original" "adra_lexical_token_overlap_cand_best_original")
metrics=("lexical_jaccard_sim_avg" "lexical_jaccard_sim_best" "lexical_token_overlap_ref_avg" "lexical_token_overlap_ref_best" "lexical_token_overlap_cand_avg" "lexical_token_overlap_cand_best")

# LCS (Longest Common Subsequence)
attacks_orig+=("adra_lexical_lcs_avg_original" "adra_lexical_lcs_best_original" "adra_lexical_lcs_ratio_avg_original" "adra_lexical_lcs_ratio_best_original" "adra_lexical_lcs_ratio_cand_avg_original" "adra_lexical_lcs_ratio_cand_best_original")
metrics+=("lexical_lcs_len_avg" "lexical_lcs_len_best" "lexical_lcs_ratio_avg" "lexical_lcs_ratio_best" "lexical_lcs_ratio_cand_avg" "lexical_lcs_ratio_cand_best")

# N-gram Coverage
attacks_orig+=("adra_lexical_ngram_coverage_avg_original" "adra_lexical_ngram_coverage_best_original" "adra_lexical_ngram_coverage_ref_avg_original" "adra_lexical_ngram_coverage_ref_best_original")
metrics+=("lexical_ngram_coverage_avg" "lexical_ngram_coverage_best" "lexical_ngram_coverage_ref_avg" "lexical_ngram_coverage_ref_best")

# Embedding Cosine Similarity
attacks_orig+=("adra_q3_8b_embedding_cosine_sim_avg_original" "adra_q3_8b_embedding_cosine_sim_best_original")
metrics+=("embedding_cosine_sim_avg" "embedding_cosine_sim_best")

bash ${BASE_PATH}/scripts/eval/run_evaluation.sh \
  "$save_path" \
  "$eval_json" \
  --mia-jsonl \
  --attack "${attacks_orig[@]}" \
  --score-metrics "${metrics[@]}"

echo "=== Running MIA evaluation (full suffix) ==="
for i in "${!attacks_orig[@]}"; do
  attack="${attacks_orig[$i]}"
  python3 -m adra.scripts.evaluate_mia \
    --members "${BASE_PATH}/eval/${eval_model_dir}/${attack}_members.jsonl" \
    --nonmembers "${BASE_PATH}/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
    --output "${BASE_PATH}/eval/${eval_model_dir}/${attack}_metrics.json" \
    --higher-is-member
done

# ------------------------------------------------------------------------------
# 4. Extract AUROC scores
# ------------------------------------------------------------------------------
echo "=== Extracting AUROC scores ==="
python3 ${BASE_PATH}/adra/utils/extract_mia_aurocs.py \
  "${BASE_PATH}/eval/${eval_model_dir}"

echo "Done: temp=${temperature}, top_p=${top_p}, top_k=${top_k}"
