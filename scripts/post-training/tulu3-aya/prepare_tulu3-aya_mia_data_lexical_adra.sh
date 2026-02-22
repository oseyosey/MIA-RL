#!/bin/bash

# Processed data is available at: https://huggingface.co/ADRA-RL

# TODO: Set the base path to the root of this repository
BASE_PATH=""

METRIC_PROFILE="unique_ngram_coverage_ref_ratio_1.50"
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=15
SUBSET_SIZE=64
SEED=1

MEMBER_DATASET="ADRA-RL/tulu-aya-processed"
NONMEMBER_DATASET="ADRA-RL/aya-non-member-processed"

python ${BASE_PATH}/verl/examples/data_preprocess/post_training_custom_mia.py \
    --output_dir ${BASE_PATH}/data/tulu3-aya_rl/tulu3-aya_rl_lexical_${METRIC_PROFILE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_prefix_0.25 \
    --dataset_path ${MEMBER_DATASET} \
    --dataset_split train \
    --match_type lexical \
    --lexical_metric_profile ${METRIC_PROFILE} \
    --lexical_num_workers 48 \
    --lexical_show_progress \
    --subset_size ${SUBSET_SIZE} \
    --subset_seed ${SEED} \
    --include_target_gt \
    --mia \
    --mia_nonmember_method separate_dataset \
    --nonmember_dataset_path ${NONMEMBER_DATASET} \
    --nonmember_dataset_split train \
    --output_name aya_${SUBSET_SIZE} \
    --augment_target_gt \
    --augment_sampling_method ${AUGMENT_SAMPLING_METHOD} \
    --augment_num_samples ${AUGMENT_NUM_SAMPLES} \
    --enable_assistant_prefix \
    --assistant_prefix_ratio 0.25 \
    --verbose
