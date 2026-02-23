#!/bin/bash

# Processed data is available at: https://huggingface.co/ADRA-RL

# TODO: Set the base path to the root of this repository
BASE_PATH=""

DATASET_PATH="ADRA-RL/Dolma3-Arxiv-Paraphrased-Gemini-2.5-Flash"
INPUT_FIELD="text"
CUSTOM_PROMPT="Continue the generation as closely to verbatim as possible. "
METRIC_PROFILE="trio_v3_unique_ratio_penalty_1.50"
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=7
SUBSET_SIZE=64
SEED=1
PREFIX_RATIO=0.25

python ${BASE_PATH}/verl/examples/data_preprocess/pre_training_custom_mia.py \
    --output_dir ${BASE_PATH}/data/dolma3-arxiv_rl/dolma3-arxiv-mia-1k_paraphrased_${SUBSET_SIZE}_rl_lexical_${METRIC_PROFILE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${SEED}_prefix_${PREFIX_RATIO}_assist_${PREFIX_RATIO} \
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
    --output_name dolma3-arxiv_${SUBSET_SIZE} \
    --augment_target_gt \
    --augment_sampling_method ${AUGMENT_SAMPLING_METHOD} \
    --augment_num_samples ${AUGMENT_NUM_SAMPLES} \
    --enable_assistant_prefix \
    --assistant_prefix_ratio ${PREFIX_RATIO} \
    --verbose \
    --drop_empty_input
