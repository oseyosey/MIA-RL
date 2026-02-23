#!/bin/bash

# Processed data is available at: https://huggingface.co/ADRA-RL

# TODO: Set the base path to the root of this repository
BASE_PATH=""

DATASET_PATH="swj0419/BookMIA"
INPUT_FIELD="snippet"
CUSTOM_PROMPT="You will receive a prefix from a passage and be asked to complete it based on the text of a famous work. Provide only the continuation for the last given prefix without any extra commentary, formatting, or additional text.  Complete the prefix:"
METRIC_PROFILE="trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max"
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=7
SUBSET_SIZE=64
SEED=1
PREFIX_RATIO=0.25

# TODO: Set the paths to the MIA baseline score files (members and non-members).
# e.g. mia_weights_members="EVAL_PATH/mia_bookmia_baselines/min_k_members.jsonl"
# e.g. mia_weights_nonmembers="EVAL_PATH/mia_bookmia_baselines/min_k_nonmembers.jsonl"
mia_weights_members=""
mia_weights_nonmembers=""
mia_weight_tag="min_k"

python ${BASE_PATH}/verl/examples/data_preprocess/pre_training_custom_mia.py \
    --output_dir ${BASE_PATH}/data/bookmia_rl/bookmia_${SUBSET_SIZE}_rl_lexical_${METRIC_PROFILE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${SEED}_prefix_${PREFIX_RATIO}_assist_${PREFIX_RATIO}_${mia_weight_tag} \
    --dataset_path ${DATASET_PATH} \
    --input_field ${INPUT_FIELD} \
    --prefix_ratio ${PREFIX_RATIO} \
    --custom_prompt "${CUSTOM_PROMPT}" \
    --match_type lexical \
    --lexical_metric_profile ${METRIC_PROFILE} \
    --lexical_num_workers 48 \
    --lexical_show_progress \
    --subset_size ${SUBSET_SIZE} \
    --subset_seed ${SEED} \
    --include_target_gt \
    --mia \
    --output_name bookmia_${SUBSET_SIZE} \
    --augment_target_gt \
    --augment_sampling_method ${AUGMENT_SAMPLING_METHOD} \
    --augment_num_samples ${AUGMENT_NUM_SAMPLES} \
    --enable_assistant_prefix \
    --assistant_prefix_ratio ${PREFIX_RATIO} \
    --verbose \
    --mia_weights_members "$mia_weights_members" \
    --mia_weights_nonmembers "$mia_weights_nonmembers" \
    --mia_weights_tag "$mia_weight_tag"
