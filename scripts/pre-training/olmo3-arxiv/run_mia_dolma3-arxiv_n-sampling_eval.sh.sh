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


# Generation hyper-parameters
temperatures=(0.7 1.0)
top_ps=(0.95)
top_k=50
n_samples=32
seed=2


# ------------------------------------------------------------------------------
# User-tweakable arguments – adjust as you wish
# ------------------------------------------------------------------------------
BASE_PATH="/gpfs/scrubbed/osey"

data_path="/gpfs/scrubbed/osey/Dataset_Distillation/data/dolma3-arxiv_rl/dolma3-arxiv-mia-1k-1024_64_rl_lexical_trio_v3_unique_ratio_penalty_1.50_augment_random_7_seed${seed}_prefix_0.25_assist_0.25/train.parquet"
n_gpus=8


# export DDRL_USE_TRANSFORMERS_TOKENIZER=true

# Seq length
## Random: 3048 for prompt, 1048 for response
## WizardLM, GPT4Alpaca, CodeAlpaca: 1048 for prompt, 3048 for response
PROMPT_LENGTH=1024
RESPONSE_LENGTH=1024

# SFT (contaminated) Mode
model_dir="allenai/Olmo-3-7B-Instruct"

# mia_weight_tag="min_k++"  /gpfs/scrubbed/osey/Dataset_Distillation/eval/M9.3_baselines_seed1/mia_wikimia24-hard_m9.3_qwen2-7b_seed1_budget32_temp1.0_topp0.95_topk50_prefix_0.25_match_maxtok1024

# ------------------------------------------------------------------------------
# Hyperparameter sweep: Loop over temperature and top_p combinations
# ------------------------------------------------------------------------------
for temperature in "${temperatures[@]}"; do
  for top_p in "${top_ps[@]}"; do
    echo "=========================================="
    echo "Running with temperature=${temperature}, top_p=${top_p}, top_k=${top_k}"
    echo "=========================================="
    
    eval_model_dir="M9.4_baselines_seed${seed}/mia_dolma3-arxiv-1024_m9.4_olmo3-7b-instruct_seed${seed}_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25_match_maxtok1024"

    # Output artifacts
    save_path="${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/mia_dolma3-arxiv-1024_m9.4_olmo3-7b-instruct_seed${seed}_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25_maxtok1024.parquet"
    eval_match_json="${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/mia_dolma3-arxiv-1024_m9.4_olmo3-7b-instruct_seed${seed}_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25_match_maxtok1024.json"
    eval_json="${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}/mia_dolma3-arxiv-1024_m9.4_olmo3-7b-instruct_seed${seed}_budget${n_samples}_temp${temperature}_topp${top_p}_topk${top_k}_prefix_0.25_maxtok1024.json"
    
    # ------------------------------------------------------------------------------
    # 2. Generate data with the freshly merged model
    # ------------------------------------------------------------------------------
    # Some clusters (e.g. HYAK) require ROCR to be unset for NVIDIA GPUs.
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


    # # ------------------------------------------------------------------------------
    # 3. lexical metrics evaluation
    # ------------------------------------------------------------------------------
        
    # Define attacks and their corresponding metrics (used in all iterations)
    # Grouped by metric type for better readability
    
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

    

    # Run evaluation once with all metrics
    bash ${BASE_PATH}/Dataset_Distillation/DDRL/scripts/eval/run_evaluation.sh \
      "$save_path" \
      "$eval_match_json" \
      --mia-jsonl \
      --embedding-model qwen3-8B \
      --attack "${attacks[@]}" \
      --score-metrics "${metrics[@]}" \
      --prefix-ratio 0.25
    
    
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

    
    echo "✅ Completed hyperparameter combination: temp=${temperature}, top_p=${top_p}, top_k=${top_k}"
    echo ""

    ## Extract AUROC scores from all metrics files
    echo "=== Extracting AUROC scores ==="
    python3 ${BASE_PATH}/Dataset_Distillation/DDRL/ddrl/utils/extract_mia_aurocs.py \
      "${BASE_PATH}/Dataset_Distillation/eval/${eval_model_dir}"

  done
done

echo "=========================================="
echo "✅ Hyperparameter sweep completed successfully!"
echo "=========================================="