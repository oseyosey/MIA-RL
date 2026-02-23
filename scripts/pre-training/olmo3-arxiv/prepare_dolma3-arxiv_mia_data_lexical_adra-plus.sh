#!/bin/bash

# BookMIA MIA Data Preprocessing with Lexical Metrics
#
# This script processes BookMIA dataset for MIA evaluation:
# - Dataset: swj0419/BookMIA
# - Labels: 0=non-member (books published in 2023), 1=member (older books)
# - Splits snippet text into prefix and suffix based on prefix_ratio
# - Adds custom prompt to user messages

# Set default base path if not already set
BASE_PATH="/gpfs/scrubbed/osey"

# Configuration
DATASET_PATH="osieosie/dolma3-arxiv-mia-1k-1024"
INPUT_FIELD="text"
CUSTOM_PROMPT="Continue the generation as closely to verbatim as possible. "
METRIC_PROFILE="trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max"
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=7
SUBSET_SIZE=64
SUBSET_SEED=2
PREFIX_RATIO=0.25 # SHOULD BE THE SAME AS ASSISTANT_PREFIX_RATIO
ASSISTANT_PREFIX_RATIO=0.25 # SHOULD BE THE SAME AS PREFIX_RATIO

# Output configuration
OUTPUT_DIR="${BASE_PATH}/Dataset_Distillation/data/dolma3-arxiv_rl/dolma3-arxiv-mia-1k-1024_${SUBSET_SIZE}_rl_lexical_${METRIC_PROFILE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${SUBSET_SEED}_prefix_${PREFIX_RATIO}_assist_${ASSISTANT_PREFIX_RATIO}"
OUTPUT_NAME="dolma3-arxiv_${SUBSET_SIZE}"

# Optional: MIA weights files (comment out if not available)
mia_weights_members="${BASE_PATH}/Dataset_Distillation/eval/M9.4_baselines_seed${SUBSET_SEED}/mia_dolma3-arxiv-1024_m9.4_olmo3-7b-instruct_seed${SUBSET_SEED}/loss_members.jsonl"
mia_weights_nonmembers="${BASE_PATH}/Dataset_Distillation/eval/M9.4_baselines_seed${SUBSET_SEED}/mia_dolma3-arxiv-1024_m9.4_olmo3-7b-instruct_seed${SUBSET_SEED}/loss_nonmembers.jsonl"
mia_weight_tag="loss"

python ${BASE_PATH}/Dataset_Distillation/DDRL/verl/examples/data_preprocess/pre_training_custom_mia.py \
    --output_dir ${OUTPUT_DIR} \
    --dataset_path ${DATASET_PATH} \
    --input_field ${INPUT_FIELD} \
    --prefix_ratio ${PREFIX_RATIO} \
    --custom_prompt "${CUSTOM_PROMPT}" \
    --match_type lexical \
    --lexical_metric_profile ${METRIC_PROFILE} \
    --lexical_num_workers 48 \
    --lexical_show_progress \
    --subset_size ${SUBSET_SIZE} \
    --subset_seed ${SUBSET_SEED} \
    --include_target_gt \
    --mia \
    --output_name ${OUTPUT_NAME} \
    --augment_target_gt \
    --augment_sampling_method ${AUGMENT_SAMPLING_METHOD} \
    --augment_num_samples ${AUGMENT_NUM_SAMPLES} \
    --enable_assistant_prefix \
    --assistant_prefix_ratio ${ASSISTANT_PREFIX_RATIO} \
    --verbose \
    --drop_empty_input \
    --mia_weights_members "$mia_weights_members" \
    --mia_weights_nonmembers "$mia_weights_nonmembers" \
    --mia_weights_tag "$mia_weight_tag" \
