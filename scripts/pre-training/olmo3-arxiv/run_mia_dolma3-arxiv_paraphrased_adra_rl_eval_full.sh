#!/usr/bin/env bash
set -euo pipefail
set -x

#TODO Ensure CUDA toolkit (nvcc) is available on compute nodes.
# Uncomment this section if using adra-v1 environment
# module load gcc/13.4.0
# module load cuda/12.9.1
# export CUDA_HOME="/gpfs/software/cuda/12.9.1"
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# TODO: Set the base path to the root of this repository
BASE_PATH=""

# TODO: Set data_path
# e.g. data_path="${BASE_PATH}/data/dolma3-arxiv_rl/dolma3-arxiv-mia-1k-1024_paraphrased_64_rl_lexical_trio_v3_unique_ratio_penalty_1.50_augment_random_7_seed1_prefix_0.25_assist_0.25/train.parquet"
data_path=""

n_gpus=8
temperature=0.7
top_p=0.95
top_k=50
n_samples=32
seed=1

PROMPT_LENGTH=2048
RESPONSE_LENGTH=1024

base_model="allenai/Olmo-3-7B-Instruct"

# TODO: Set lora_base_path to the RL checkpoint directory
lora_base_path=""

PROMPT_TEMPLATE="trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max"
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=7
rollout_n=32
lr=5e-5
exp_name="verl_dolma3-arxiv_paraphrased_adra-plus_lora_h200_8_${PROMPT_TEMPLATE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${seed}_lr${lr}_temp${temperature}_topp${top_p}_topk${top_k}_rollout${rollout_n}_lora"

# Jaccard Similarity & token overlap
attacks=(
  "adra_lexical_jaccard_sim_avg" "adra_lexical_jaccard_sim_best"
  "adra_lexical_token_overlap_ref_avg" "adra_lexical_token_overlap_ref_best"
  "adra_lexical_token_overlap_cand_avg" "adra_lexical_token_overlap_cand_best"
)
metrics=(
  "lexical_jaccard_sim_avg" "lexical_jaccard_sim_best"
  "lexical_token_overlap_ref_avg" "lexical_token_overlap_ref_best"
  "lexical_token_overlap_cand_avg" "lexical_token_overlap_cand_best"
)

# LCS
attacks+=(
  "adra_lexical_lcs_avg" "adra_lexical_lcs_best"
  "adra_lexical_lcs_ratio_avg" "adra_lexical_lcs_ratio_best"
  "adra_lexical_lcs_ratio_cand_avg" "adra_lexical_lcs_ratio_cand_best"
)
metrics+=(
  "lexical_lcs_len_avg" "lexical_lcs_len_best"
  "lexical_lcs_ratio_avg" "lexical_lcs_ratio_best"
  "lexical_lcs_ratio_cand_avg" "lexical_lcs_ratio_cand_best"
)

# N-gram Coverage
attacks+=(
  "adra_lexical_ngram_coverage_avg" "adra_lexical_ngram_coverage_best"
  "adra_lexical_ngram_coverage_ref_avg" "adra_lexical_ngram_coverage_ref_best"
)
metrics+=(
  "lexical_ngram_coverage_avg" "lexical_ngram_coverage_best"
  "lexical_ngram_coverage_ref_avg" "lexical_ngram_coverage_ref_best"
)

# Embedding Cosine Similarity
attacks+=(
  "adra_q3_8b_embedding_cosine_sim_avg" "adra_q3_8b_embedding_cosine_sim_best"
)
metrics+=(
  "embedding_cosine_sim_avg" "embedding_cosine_sim_best"
)

for global_step in 10 20 30 40; do
  echo "=============================================================================="
  echo "Processing global_step_${global_step}"
  echo "=============================================================================="

  lora_adapter="${lora_base_path}/global_step_${global_step}/actor/lora_adapter"
  merged_model_dir="${BASE_PATH}/merged_models/${exp_name}_step_${global_step}"
  eval_model_dir="${exp_name}_step_${global_step}"

  save_path="${BASE_PATH}/eval/${eval_model_dir}/${exp_name}_step_${global_step}.parquet"
  eval_match_json="${BASE_PATH}/eval/${eval_model_dir}/${exp_name}_step_${global_step}_match.json"
  eval_json="${BASE_PATH}/eval/${eval_model_dir}/${exp_name}_step_${global_step}.json"

  # 1. Merge LoRA adapter
  echo "Step ${global_step}: Merging LoRA adapter..."
  python adra/utils_rl/merge_lora.py \
    --base_model "$base_model" \
    --lora_adapter "$lora_adapter" \
    --output_dir "$merged_model_dir" \
    --dtype bfloat16 \
    --safe_serialization

  # 2. Generate samples
  unset ROCR_VISIBLE_DEVICES || true

  echo "Step ${global_step}: Generating data..."
  python3 -m verl.trainer.main_generation \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node=${n_gpus} \
      ray_init.num_cpus=80 \
      data.assistant_prefix_key=assistant_prefix \
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
      rollout.tensor_model_parallel_size=4 \
      rollout.gpu_memory_utilization=0.95

  # 3. Match evaluation
  echo "Step ${global_step}: Evaluating (match)..."
  bash ${BASE_PATH}/scripts/eval/run_evaluation.sh \
    "$save_path" \
    "$eval_match_json" \
    --prefix-ratio 0.25 \
    --embedding-model qwen3-8B \
    --mia-jsonl \
    --attack "${attacks[@]}" \
    --score-metrics "${metrics[@]}"

  for i in "${!attacks[@]}"; do
    attack="${attacks[$i]}"
    python3 -m adra.scripts.evaluate_mia \
      --members "${BASE_PATH}/eval/${eval_model_dir}/${attack}_members.jsonl" \
      --nonmembers "${BASE_PATH}/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
      --output "${BASE_PATH}/eval/${eval_model_dir}/${attack}_metrics.json" \
      --higher-is-member
  done

  # 4. Full suffix evaluation
  attacks_orig=(
    "adra_lexical_jaccard_sim_avg_original" "adra_lexical_jaccard_sim_best_original"
    "adra_lexical_token_overlap_ref_avg_original" "adra_lexical_token_overlap_ref_best_original"
    "adra_lexical_token_overlap_cand_avg_original" "adra_lexical_token_overlap_cand_best_original"
  )
  metrics_orig=(
    "lexical_jaccard_sim_avg" "lexical_jaccard_sim_best"
    "lexical_token_overlap_ref_avg" "lexical_token_overlap_ref_best"
    "lexical_token_overlap_cand_avg" "lexical_token_overlap_cand_best"
  )

  attacks_orig+=(
    "adra_lexical_lcs_avg_original" "adra_lexical_lcs_best_original"
    "adra_lexical_lcs_ratio_avg_original" "adra_lexical_lcs_ratio_best_original"
    "adra_lexical_lcs_ratio_cand_avg_original" "adra_lexical_lcs_ratio_cand_best_original"
  )
  metrics_orig+=(
    "lexical_lcs_len_avg" "lexical_lcs_len_best"
    "lexical_lcs_ratio_avg" "lexical_lcs_ratio_best"
    "lexical_lcs_ratio_cand_avg" "lexical_lcs_ratio_cand_best"
  )

  attacks_orig+=(
    "adra_lexical_ngram_coverage_avg_original" "adra_lexical_ngram_coverage_best_original"
    "adra_lexical_ngram_coverage_ref_avg_original" "adra_lexical_ngram_coverage_ref_best_original"
  )
  metrics_orig+=(
    "lexical_ngram_coverage_avg" "lexical_ngram_coverage_best"
    "lexical_ngram_coverage_ref_avg" "lexical_ngram_coverage_ref_best"
  )

  attacks_orig+=(
    "adra_q3_8b_embedding_cosine_sim_avg_original" "adra_q3_8b_embedding_cosine_sim_best_original"
  )
  metrics_orig+=(
    "embedding_cosine_sim_avg" "embedding_cosine_sim_best"
  )

  bash ${BASE_PATH}/scripts/eval/run_evaluation.sh \
    "$save_path" \
    "$eval_json" \
    --mia-jsonl \
    --attack "${attacks_orig[@]}" \
    --score-metrics "${metrics_orig[@]}"

  echo "=== Running MIA evaluation (non-match) ==="
  for i in "${!attacks_orig[@]}"; do
    attack="${attacks_orig[$i]}"
    python3 -m adra.scripts.evaluate_mia \
      --members "${BASE_PATH}/eval/${eval_model_dir}/${attack}_members.jsonl" \
      --nonmembers "${BASE_PATH}/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
      --output "${BASE_PATH}/eval/${eval_model_dir}/${attack}_metrics.json" \
      --higher-is-member
  done

  # 5. Extract AUROC scores
  echo "=== Extracting AUROC scores ==="
  python3 ${BASE_PATH}/adra/utils/extract_mia_aurocs.py \
    "${BASE_PATH}/eval/${eval_model_dir}"

  echo "Step ${global_step} completed successfully."
  echo ""
done

echo "All workflows completed successfully!"
