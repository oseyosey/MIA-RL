#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Quick end-to-end eval for a single RL checkpoint uploaded to HuggingFace:
#   1. Merge LoRA adapter into the base model
#   2. Generate n samples from the merged model
#   3. Evaluate with lexical MIA metrics (match + full-suffix)
#   4. Extract AUROC scores
# ------------------------------------------------------------------------------

set -euo pipefail
set -x

# Ensure CUDA toolkit (nvcc) is available on compute nodes.
module load gcc/13.4.0
module load cuda/12.9.1
export CUDA_HOME="/gpfs/software/cuda/12.9.1"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# ------------------------------------------------------------------------------
# User-tweakable arguments
# ------------------------------------------------------------------------------

# TODO: Set the base path to the root of this repository
BASE_PATH=""

# TODO: Set path to the prepared training data parquet file
# e.g. data_path="ADRA-RL/data/aime_rl/aime_16_rl_lexical_unique_ngram_coverage_ref_ratio_1.50_augment_random_7_seed1_prefix_0.25/train.parquet"
data_path=""

n_gpus=8

# Generation hyper-parameters
temperature=0.7
top_p=0.95
top_k=50
n_samples=32

PROMPT_LENGTH=2048
RESPONSE_LENGTH=2048

# HuggingFace model IDs
base_model="ADRA-RL/tulu2-7b_aime_controlled_contamination_original"
lora_adapter="ADRA-RL/tulu2-7b_lora_adra-plus_aime_original_lexical_unique_ngram_coverage_s70"

# Local path where the merged model will be saved
merged_model_dir="./merged_models/tulu2-7b_aime_adra_original_lora_merged"

# Output directory for evaluation artifacts
eval_model_dir="tulu2-7b_aime_adra_original_lora_merged"

# Output artifacts
save_path="${BASE_PATH}/eval/${eval_model_dir}/generations_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25.parquet"
eval_match_json="${BASE_PATH}/eval/${eval_model_dir}/mia_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25_match.json"
eval_json="${BASE_PATH}/eval/${eval_model_dir}/mia_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25.json"

# ------------------------------------------------------------------------------
# 1. Merge LoRA adapter into base model
# ------------------------------------------------------------------------------
python ddrl/utils_rl/merge_lora.py \
  --base_model   "$base_model" \
  --lora_adapter "$lora_adapter" \
  --output_dir   "$merged_model_dir" \
  --dtype bfloat16 \
  --safe_serialization

# ------------------------------------------------------------------------------
# 2. Generate samples
# ------------------------------------------------------------------------------
unset ROCR_VISIBLE_DEVICES || true  # might need this for specific clusters

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${n_gpus} \
    data.assistant_prefix_key=assistant_prefix \
    data.batch_size=512 \
    data.path="$data_path" \
    data.prompt_key=prompt \
    data.n_samples=$n_samples \
    data.output_path="$save_path" \
    model.path="$merged_model_dir" \
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
# 3. Lexical MIA metrics evaluation (match evaluation)
# ------------------------------------------------------------------------------

# Jaccard Similarity & token overlap
attacks=("ddrl_lexical_jaccard_sim_avg" "ddrl_lexical_jaccard_sim_best" "ddrl_lexical_token_overlap_ref_avg" "ddrl_lexical_token_overlap_ref_best" "ddrl_lexical_token_overlap_cand_avg" "ddrl_lexical_token_overlap_cand_best")
metrics=("lexical_jaccard_sim_avg" "lexical_jaccard_sim_best" "lexical_token_overlap_ref_avg" "lexical_token_overlap_ref_best" "lexical_token_overlap_cand_avg" "lexical_token_overlap_cand_best")

# LCS (Longest Common Subsequence)
attacks+=("ddrl_lexical_lcs_avg" "ddrl_lexical_lcs_best" "ddrl_lexical_lcs_ratio_avg" "ddrl_lexical_lcs_ratio_best" "ddrl_lexical_lcs_ratio_cand_avg" "ddrl_lexical_lcs_ratio_cand_best")
metrics+=("lexical_lcs_len_avg" "lexical_lcs_len_best" "lexical_lcs_ratio_avg" "lexical_lcs_ratio_best" "lexical_lcs_ratio_cand_avg" "lexical_lcs_ratio_cand_best")

# N-gram Coverage
attacks+=("ddrl_lexical_ngram_coverage_avg" "ddrl_lexical_ngram_coverage_best" "ddrl_lexical_ngram_coverage_ref_avg" "ddrl_lexical_ngram_coverage_ref_best")
metrics+=("lexical_ngram_coverage_avg" "lexical_ngram_coverage_best" "lexical_ngram_coverage_ref_avg" "lexical_ngram_coverage_ref_best")

# Embedding Cosine Similarity
attacks+=("ddrl_q3_8b_embedding_cosine_sim_avg" "ddrl_q3_8b_embedding_cosine_sim_best")
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
  python3 -m ddrl.scripts.evaluate_mia \
    --members "${BASE_PATH}/eval/${eval_model_dir}/${attack}_members.jsonl" \
    --nonmembers "${BASE_PATH}/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
    --output "${BASE_PATH}/eval/${eval_model_dir}/${attack}_metrics.json" \
    --higher-is-member
done

# ------------------------------------------------------------------------------
# 4. Full suffix metrics evaluation
# ------------------------------------------------------------------------------

attacks_orig=("ddrl_lexical_jaccard_sim_avg_original" "ddrl_lexical_jaccard_sim_best_original" "ddrl_lexical_token_overlap_ref_avg_original" "ddrl_lexical_token_overlap_ref_best_original" "ddrl_lexical_token_overlap_cand_avg_original" "ddrl_lexical_token_overlap_cand_best_original")
metrics_orig=("lexical_jaccard_sim_avg" "lexical_jaccard_sim_best" "lexical_token_overlap_ref_avg" "lexical_token_overlap_ref_best" "lexical_token_overlap_cand_avg" "lexical_token_overlap_cand_best")

# LCS (Longest Common Subsequence)
attacks_orig+=("ddrl_lexical_lcs_avg_original" "ddrl_lexical_lcs_best_original" "ddrl_lexical_lcs_ratio_avg_original" "ddrl_lexical_lcs_ratio_best_original" "ddrl_lexical_lcs_ratio_cand_avg_original" "ddrl_lexical_lcs_ratio_cand_best_original")
metrics_orig+=("lexical_lcs_len_avg" "lexical_lcs_len_best" "lexical_lcs_ratio_avg" "lexical_lcs_ratio_best" "lexical_lcs_ratio_cand_avg" "lexical_lcs_ratio_cand_best")

# N-gram Coverage
attacks_orig+=("ddrl_lexical_ngram_coverage_avg_original" "ddrl_lexical_ngram_coverage_best_original" "ddrl_lexical_ngram_coverage_ref_avg_original" "ddrl_lexical_ngram_coverage_ref_best_original")
metrics_orig+=("lexical_ngram_coverage_avg" "lexical_ngram_coverage_best" "lexical_ngram_coverage_ref_avg" "lexical_ngram_coverage_ref_best")

# Embedding Cosine Similarity
attacks_orig+=("ddrl_q3_8b_embedding_cosine_sim_avg_original" "ddrl_q3_8b_embedding_cosine_sim_best_original")
metrics_orig+=("embedding_cosine_sim_avg" "embedding_cosine_sim_best")

bash ${BASE_PATH}/scripts/eval/run_evaluation.sh \
  "$save_path" \
  "$eval_json" \
  --mia-jsonl \
  --attack "${attacks_orig[@]}" \
  --score-metrics "${metrics_orig[@]}"

echo "=== Running MIA evaluation (full suffix) ==="
for i in "${!attacks_orig[@]}"; do
  attack="${attacks_orig[$i]}"
  python3 -m ddrl.scripts.evaluate_mia \
    --members "${BASE_PATH}/eval/${eval_model_dir}/${attack}_members.jsonl" \
    --nonmembers "${BASE_PATH}/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
    --output "${BASE_PATH}/eval/${eval_model_dir}/${attack}_metrics.json" \
    --higher-is-member
done

# ------------------------------------------------------------------------------
# 5. Extract AUROC scores
# ------------------------------------------------------------------------------
echo "=== Extracting AUROC scores ==="
python3 ${BASE_PATH}/ddrl/utils/extract_mia_aurocs.py \
  "${BASE_PATH}/eval/${eval_model_dir}"

echo "Done. Results: ${BASE_PATH}/eval/${eval_model_dir}"
