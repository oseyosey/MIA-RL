#!/usr/bin/env bash
set -euo pipefail


#TODO Ensure CUDA toolkit (nvcc) is available on compute nodes.
# Uncomment this section if using adra-v1 environment
# module load gcc/13.4.0
# module load cuda/12.9.1
# export CUDA_HOME="/gpfs/software/cuda/12.9.1"
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# e.g. DATA_PATH="/path/to/MIA-RL/data"
# e.g. EVAL_PATH="/path/to/MIA-RL/eval"
DATA_PATH=""
EVAL_PATH=""

SEED=1
MODEL="allenai/Olmo-3-7B-Instruct"
DATA_DIR="$DATA_PATH/dolma3-arxiv_rl/dolma3-arxiv-mia-1k_paraphrased_64_rl_lexical_trio_v3_unique_ratio_penalty_1.50_augment_random_7_seed${SEED}_prefix_0.25_assist_0.25"
OUT_DIR="$EVAL_PATH/mia_dolma3-arxiv_paraphrased_baselines"
MEMBERS_FILE="$DATA_DIR/dolma3-arxiv_64_members.jsonl"
NONMEMBERS_FILE="$DATA_DIR/dolma3-arxiv_64_nonmembers.jsonl"
mkdir -p "$OUT_DIR"


echo "Running MIA baselines with model: $MODEL"

declare -a ATTACKS=(
  "loss"
  "zlib"
  "min_k"
  "min_k++"
)

# Optional reference-based attack if a small ref model is available
# ATTACK_REF=""
# REF_MODEL=""

### * NON-REFERENCE BASED ATTACKS * ###
for ATTACK in "${ATTACKS[@]}"; do
  echo "Attack: $ATTACK (members)"
  python -m adra.scripts.run_mia \
    --model "$MODEL" \
    --dataset "$MEMBERS_FILE" \
    --attack "$ATTACK" \
    --max-length 4096 \
    --output "$OUT_DIR/${ATTACK}_members.jsonl" \
    --verbose

  echo "Attack: $ATTACK (nonmembers)"
  python -m adra.scripts.run_mia \
    --model "$MODEL" \
    --dataset "$NONMEMBERS_FILE" \
    --attack "$ATTACK" \
    --max-length 4096 \
    --output "$OUT_DIR/${ATTACK}_nonmembers.jsonl" \
    --verbose

  echo "Evaluate: $ATTACK"
  python -m adra.scripts.evaluate_mia \
    --members "$OUT_DIR/${ATTACK}_members.jsonl" \
    --nonmembers "$OUT_DIR/${ATTACK}_nonmembers.jsonl" \
    --output "$OUT_DIR/${ATTACK}_metrics.json"
done


### * REFERENCE-BASED ATTACKS * ###
# echo "Attack: ref (members)"
# python -m adra.scripts.run_mia \
#   --model "$MODEL" \
#   --dataset "$MEMBERS_FILE" \
#   --reference-model "$REF_MODEL" \
#   --attack ref \
#   --max-length 4096 \
#   --output "$OUT_DIR/${ATTACK_REF}_members.jsonl"

# echo "Attack: ref (nonmembers)"
# python -m adra.scripts.run_mia \
#   --model "$MODEL" \
#   --dataset "$NONMEMBERS_FILE" \
#   --reference-model "$REF_MODEL" \
#   --attack ref \
#   --max-length 4096 \
#   --output "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl"

# echo "Evaluate: ref"
# python -m adra.scripts.evaluate_mia \
#   --members "$OUT_DIR/${ATTACK_REF}_members.jsonl" \
#   --nonmembers "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl" \
#   --output "$OUT_DIR/${ATTACK_REF}_metrics.json"

# echo "Done. Results in $OUT_DIR"
