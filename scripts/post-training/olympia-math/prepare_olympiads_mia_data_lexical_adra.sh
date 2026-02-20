#!/bin/bash

# Processed data is available at: https://huggingface.co/ADRA-RL

# TODO: Set the base path to the root of this repository
BASE_PATH=""

METRIC_PROFILE="trio_v3_unique_ratio_penalty_2.0"
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=7
SEED=1

python ${BASE_PATH}/verl/examples/data_preprocess/post_training_custom_mia.py \
    --output_dir ${BASE_PATH}/data/olympiads_rl/olympiads_rl_lexical_${METRIC_PROFILE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${SEED}_prefix_0.25 \
    --dataset_path ADRA-RL/Olympiads-Cleaned \
    --dataset_split train \
    --match_type lexical \
    --lexical_metric_profile ${METRIC_PROFILE} \
    --lexical_num_workers 48 \
    --lexical_show_progress \
    --subset_size 32 \
    --subset_seed ${SEED} \
    --include_target_gt \
    --mia \
    --mia_nonmember_method unused_examples \
    --output_name olympiads_32 \
    --augment_target_gt \
    --augment_sampling_method ${AUGMENT_SAMPLING_METHOD} \
    --augment_num_samples ${AUGMENT_NUM_SAMPLES} \
    --enable_assistant_prefix \
    --assistant_prefix_ratio 0.25 \
    --verbose
