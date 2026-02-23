#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Merge LoRA weights into the base Tulu-2-7B model and then run generation
# followed by evaluation – all in one place.
# ------------------------------------------------------------------------------
#   1. Merge LoRA adapter → merged model directory (safe-serialised, bfloat16)
#   2. Use merged model to generate new samples
#   3. Evaluate the generated samples
# ------------------------------------------------------------------------------
# This script simply strings together the two standalone scripts that already
# exist in the repo so that you can kick off the whole workflow with a single
# command.
# ------------------------------------------------------------------------------

set -euo pipefail
set -x  # Echo commands for easier debugging


# Ensure CUDA toolkit (nvcc) is available on compute nodes.
module load gcc/13.4.0
module load cuda/12.9.1
export CUDA_HOME="/gpfs/software/cuda/12.9.1"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Olmo3 specific settings.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# ------------------------------------------------------------------------------
# User-tweakable arguments – adjust as you wish
# ------------------------------------------------------------------------------
# Generation hyper-parameters
temperature=0.7
top_k=50
top_p=0.95
n_samples=32
seed=2


BASE_PATH="/gpfs/scrubbed/osey"
data_path="/gpfs/scrubbed/osey/Dataset_Distillation/data/dolma3-arxiv_rl/dolma3-arxiv-mia-1k-1024_64_rl_lexical_trio_v3_unique_ratio_penalty_1.50_augment_random_7_seed${seed}_prefix_0.25_assist_0.25/train.parquet"
n_gpus=8

# Seq length
PROMPT_LENGTH=2048
RESPONSE_LENGTH=1024

# Base & LoRA locations – change only if you moved the checkpoints
base_model="allenai/Olmo-3-7B-Instruct"

# Base path for LoRA adapter (before global_step_X)
lora_base_path="/gpfs/scrubbed/osey/Dataset_Distillation/DDRL/verl_checkpoints/verl_m9.4_olmo3-7b-instruct_original_lexical_seed2/verl_olmo3-7b-instruct_dolma3-arxiv-1024_seed2_m9.4_original_trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max_augment_random_7_temp1.0_topp0.95_topk50_rollout32_lora_prefix_0.25_assist_0.25_match_maxtok1024"

# Base path components for output directories
merged_base_name="verl_olmo3-7b-instruct_dolma3-arxiv-1024_seed2_m9.4_original_trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max_augment_random_7_temp1.0_topp0.95_topk50_rollout32_lora_prefix_0.25_assist_0.25_match_maxtok1024"
eval_base_name="mia_olmo3-7b-instruct_dolma3-arxiv-1024_seed2_m9.4_original_trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max_augment_random_7_budget32_temp0.7_topp0.95_topk50_prefix_0.25_assist_0.25_match_maxtok1024"
output_base_name="olmo3-7b-instruct_dolma3-arxiv-1024_seed2_m9.4_original_trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max_augment_random_7"

# Define attacks and their corresponding metrics (used in all iterations) 
# Grouped by metric type for better readability

# Jaccard Similarity & token overlap
attacks=(
  "ddrl_lexical_jaccard_sim_avg" "ddrl_lexical_jaccard_sim_best"
  "ddrl_lexical_token_overlap_ref_avg" "ddrl_lexical_token_overlap_ref_best"
  "ddrl_lexical_token_overlap_cand_avg" "ddrl_lexical_token_overlap_cand_best"
)
metrics=(
  "lexical_jaccard_sim_avg" "lexical_jaccard_sim_best"
  "lexical_token_overlap_ref_avg"  "lexical_token_overlap_ref_best"
  "lexical_token_overlap_cand_avg"  "lexical_token_overlap_cand_best"
)


# LCS (Longest Common Subsequence)
attacks+=(
  "ddrl_lexical_lcs_avg"  "ddrl_lexical_lcs_best"
  "ddrl_lexical_lcs_ratio_avg"  "ddrl_lexical_lcs_ratio_best"
  "ddrl_lexical_lcs_ratio_cand_avg"  "ddrl_lexical_lcs_ratio_cand_best"
)
metrics+=(
  "lexical_lcs_len_avg"  "lexical_lcs_len_best"
  "lexical_lcs_ratio_avg"  "lexical_lcs_ratio_best"
  "lexical_lcs_ratio_cand_avg"  "lexical_lcs_ratio_cand_best"
)

# N-gram Coverage
attacks+=(
  "ddrl_lexical_ngram_coverage_avg"  "ddrl_lexical_ngram_coverage_best"
  "ddrl_lexical_ngram_coverage_ref_avg"  "ddrl_lexical_ngram_coverage_ref_best"
)
metrics+=(
  "lexical_ngram_coverage_avg"  "lexical_ngram_coverage_best"
  "lexical_ngram_coverage_ref_avg"  "lexical_ngram_coverage_ref_best"
)

# Embedding Cosine Similarity
attacks+=(
  "ddrl_q3_8b_embedding_cosine_sim_avg"  "ddrl_q3_8b_embedding_cosine_sim_best"
)
metrics+=(
  "embedding_cosine_sim_avg"  "embedding_cosine_sim_best"
)

# ------------------------------------------------------------------------------
# Loop over different global steps
# ------------------------------------------------------------------------------
for global_step in 10 20 30 40 50 60 70 80 90 100; do
  echo "=============================================================================="
  echo "Processing global_step_${global_step}"
  echo "=============================================================================="
  
  # Construct paths for this iteration
  lora_adapter="${lora_base_path}/global_step_${global_step}/actor/lora_adapter"
  merged_model_dir="${BASE_PATH}/Dataset_Distillation/DDRL/verl_checkpoints/Experiment_M9.4_olmo3-7b-instruct_arxiv-1024_original_seed${seed}/merged_models/${merged_base_name}_step_${global_step}"
  eval_model_dir="M9.4_rl_olmo3-7b-instruct_arxiv-1024_original_seed${seed}/${eval_base_name}_step_${global_step}_match_maxtok1024"

  save_path="${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${output_base_name}_step_${global_step}.parquet"
  eval_match_json="${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${output_base_name}_step_${global_step}_match.json"
  eval_json="${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${output_base_name}_step_${global_step}.json"
  
  # ------------------------------------------------------------------------------
  # 1. Merge LoRA adapter into base model
  # ------------------------------------------------------------------------------
  echo "Step ${global_step}: Merging LoRA adapter..."
  python ddrl/utils_rl/merge_lora.py \
    --base_model    "$base_model" \
    --lora_adapter  "$lora_adapter" \
    --output_dir    "$merged_model_dir" \
    --dtype bfloat16 \
    --safe_serialization

  # ------------------------------------------------------------------------------
  # 2. Generate data with the freshly merged model
  # ------------------------------------------------------------------------------
  # Some clusters (e.g. HYAK) require ROCR to be unset for NVIDIA GPUs.
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

  # ------------------------------------------------------------------------------
  # 3. Evaluate the generated data
  # ------------------------------------------------------------------------------
  echo "Step ${global_step}: Evaluating generated data..."
  
  # Run evaluation once with all metrics
  bash ${BASE_PATH}/Dataset_Distillation/DDRL/scripts/eval/run_evaluation.sh \
    "$save_path" \
    "$eval_match_json" \
    --prefix-ratio 0.25 \
    --embedding-model qwen3-8B \
    --mia-jsonl \
    --attack "${attacks[@]}" \
    --score-metrics "${metrics[@]}"

  ## Run MIA evaluation for each attack ##
  for i in "${!attacks[@]}"; do
    attack="${attacks[$i]}"
    
    python3 -m ddrl.scripts.evaluate_mia \
      --members "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${attack}_members.jsonl" \
      --nonmembers "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
      --output "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${attack}_metrics.json" \
      --higher-is-member
  done


  # ------------------------------------------------------------------------------
  # 4. non-match metrics evaluation
  # ------------------------------------------------------------------------------
  # echo "=== Evaluating with MIA baseline weights (non-match evaluation) ==="
  attacks_orig=("ddrl_lexical_jaccard_sim_avg_original" "ddrl_lexical_jaccard_sim_best_original" "ddrl_lexical_token_overlap_ref_avg_original" "ddrl_lexical_token_overlap_ref_best_original" "ddrl_lexical_token_overlap_cand_avg_original" "ddrl_lexical_token_overlap_cand_best_original")
  metrics=("lexical_jaccard_sim_avg" "lexical_jaccard_sim_best" "lexical_token_overlap_ref_avg" "lexical_token_overlap_ref_best" "lexical_token_overlap_cand_avg" "lexical_token_overlap_cand_best")

  # LCS (Longest Common Subsequence)
  attacks_orig+=("ddrl_lexical_lcs_avg_original" "ddrl_lexical_lcs_best_original" "ddrl_lexical_lcs_ratio_avg_original" "ddrl_lexical_lcs_ratio_best_original" "ddrl_lexical_lcs_ratio_cand_avg_original" "ddrl_lexical_lcs_ratio_cand_best_original")
  metrics+=("lexical_lcs_len_avg" "lexical_lcs_len_best" "lexical_lcs_ratio_avg" "lexical_lcs_ratio_best" "lexical_lcs_ratio_cand_avg" "lexical_lcs_ratio_cand_best")

  # N-gram Coverage
  attacks_orig+=("ddrl_lexical_ngram_coverage_avg_original" "ddrl_lexical_ngram_coverage_best_original" "ddrl_lexical_ngram_coverage_ref_avg_original" "ddrl_lexical_ngram_coverage_ref_best_original")
  metrics+=("lexical_ngram_coverage_avg" "lexical_ngram_coverage_best" "lexical_ngram_coverage_ref_avg" "lexical_ngram_coverage_ref_best")

  # Embedding Cosine Similarity
  attacks_orig+=("ddrl_q3_8b_embedding_cosine_sim_avg_original" "ddrl_q3_8b_embedding_cosine_sim_best_original")
  metrics+=("embedding_cosine_sim_avg" "embedding_cosine_sim_best")


  # # Weighted attacks and metrics
  bash ${BASE_PATH}/Dataset_Distillation/DDRL/scripts/eval/run_evaluation.sh \
    "$save_path" \
    "$eval_json" \
    --mia-jsonl \
    --attack "${attacks_orig[@]}" \
    --score-metrics "${metrics[@]}"

  # ## Run MIA evaluation for each weighted attack ##
  echo "=== Running MIA evaluation (non-match) ==="
  for i in "${!attacks_orig[@]}"; do
    attack="${attacks_orig[$i]}"

    python3 -m ddrl.scripts.evaluate_mia \
      --members "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${attack}_members.jsonl" \
      --nonmembers "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${attack}_nonmembers.jsonl" \
      --output "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/${attack}_metrics.json" \
      --higher-is-member
  done

  # ------------------------------------------------------------------------------
  # 5. Extract AUROC scores from all metrics files
  # ------------------------------------------------------------------------------
  echo "=== Extracting AUROC scores ==="
  python3 ${BASE_PATH}/Dataset_Distillation/DDRL/ddrl/utils/extract_mia_aurocs.py \
    "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}"


  echo "✅ Step ${global_step} completed successfully."
  echo "   Generated data: $save_path"
  echo "   Evaluation results: $eval_json"
  echo ""
done

# ------------------------------------------------------------------------------
echo "✅ All workflows completed successfully!"
