#!/bin/bash

# Processed data is available at: https://huggingface.co/ADRA-RL

# TODO: Set the base path to the root of this repository
BASE_PATH=""

METRIC_PROFILE="unique_ngram_coverage_ref_ratio_1.50_mia_adaptive_match_linear_distractor_max"
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=7
SEED=1

# TODO: Set the paths to the MIA baseline score files (members and non-members).
# These are produced by first running the MIA baselines evaluation, which outputs
# per-example likelihood-ratio scores (e.g. min-k++ scores) for the member and
# non-member splits as separate .jsonl files.
#e.g. mia_weights_members="EVAL_PATH/mia_aime_paraphrased_baselines/min_k++_members.jsonl"
#e.g. mia_weights_nonmembers="EVAL_PATH/mia_aime_paraphrased_baselines/min_k++_nonmembers.jsonl

mia_weights_members=""
mia_weights_nonmembers=""
mia_weight_tag="min_k++"

python ${BASE_PATH}/verl/examples/data_preprocess/post_training_custom_mia.py \
    --output_dir ${BASE_PATH}/data/aime_rl/aime_paraphrased_rl_lexical_${METRIC_PROFILE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${SEED}_${mia_weight_tag}_weighted_prefix_0.25 \
    --dataset_path ADRA-RL/AIME-2021-2025-Cleaned \
    --dataset_split train \
    --match_type lexical \
    --lexical_metric_profile ${METRIC_PROFILE} \
    --lexical_num_workers 48 \
    --lexical_show_progress \
    --subset_size 16 \
    --subset_seed ${SEED} \
    --include_target_gt \
    --mia \
    --mia_nonmember_method unused_examples \
    --output_name aime_16 \
    --transformed_dataset_path ADRA-RL/AIME-2021-2025-Paraphrased-Gemini-2.5-Flash \
    --transformed_dataset_split train \
    --augment_target_gt \
    --augment_sampling_method ${AUGMENT_SAMPLING_METHOD} \
    --augment_num_samples ${AUGMENT_NUM_SAMPLES} \
    --filter_by_year \
    --nonmember_years 2025 \
    --member_years 2021 2022 2023 2024 \
    --enable_assistant_prefix \
    --assistant_prefix_ratio 0.25 \
    --mia_weights_members "$mia_weights_members" \
    --mia_weights_nonmembers "$mia_weights_nonmembers" \
    --mia_weights_tag "$mia_weight_tag" \
    --verbose