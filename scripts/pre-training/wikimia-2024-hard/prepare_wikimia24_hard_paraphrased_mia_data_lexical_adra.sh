#!/bin/bash

# Processed data is available at: https://huggingface.co/ADRA-RL

# TODO: Set the base path to the root of this repository
BASE_PATH=""

DATASET_PATH="ADRA-RL/WikiMIA-2024-Hard-Paraphrased-Gemini-2.5-Flash"
CUSTOM_PROMPT="Continue the generation as closely to verbatim as possible. "
METRIC_PROFILE="trio_v3_unique_ratio_penalty_1.50"
AUGMENT_SAMPLING_METHOD="perturbed"
AUGMENT_NUM_SAMPLES=3
SUBSET_SIZE=64
SEED=1
PREFIX_RATIO=0.25
TOKENIZER_NAME="allenai/tulu-2-7b"

python ${BASE_PATH}/verl/examples/data_preprocess/pre_training_custom_mia.py \
    --output_dir ${BASE_PATH}/data/wikimia24_hard_rl/wikimia24_hard_paraphrased_${SUBSET_SIZE}_rl_lexical_${METRIC_PROFILE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${SEED}_prefix_${PREFIX_RATIO}_assist_${PREFIX_RATIO} \
    --dataset_path ${DATASET_PATH} \
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
    --output_name wikimia24_hard_${SUBSET_SIZE} \
    --augment_target_gt \
    --augment_sampling_method ${AUGMENT_SAMPLING_METHOD} \
    --augment_num_samples ${AUGMENT_NUM_SAMPLES} \
    --enable_assistant_prefix \
    --assistant_prefix_ratio ${PREFIX_RATIO} \
    --diff_threshold 400 \
    --min_tokens 256 \
    --max_tokens 1024 \
    --tokenizer_name ${TOKENIZER_NAME} \
    --strict_pairing \
    --verbose
