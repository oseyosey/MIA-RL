#!/bin/bash

# Processed data is available at: https://huggingface.co/ADRA-RL

# TODO: Set the base path to the root of this repository
BASE_PATH=""

METRIC_PROFILE="trio_v3_unique_ratio_penalty_1.50"

python ${BASE_PATH}/verl/examples/data_preprocess/s1_match_custom_mia.py \
    --output_dir ${BASE_PATH}/data/s1_rl/s1_gemini_rl_lexical_${METRIC_PROFILE} \
    --dataset_path simplescaling/s1K-1.1 \
    --dataset_split train \
    --match_type lexical \
    --lexical_metric_profile ${METRIC_PROFILE} \
    --lexical_num_workers 32 \
    --subset_size 128 \
    --subset_seed 42 \
    --include_target_gt \
    --mia \
    --mia_nonmember_method perturbed_solution \
    --random_pairing_mode same_problem \
    --perturbed_solution_field gemini_attempt \
    --output_name s1_128 \
    --reverse_member \
    --verbose
