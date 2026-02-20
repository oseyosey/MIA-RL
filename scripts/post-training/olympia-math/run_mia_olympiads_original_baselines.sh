#!/usr/bin/env bash
set -euo pipefail

# e.g. DATA_PATH="/path/to/MIA-RL/data"
# e.g. EVAL_PATH="/path/to/MIA-RL/eval"
DATA_PATH=""
EVAL_PATH=""

MODEL="ADRA-RL/tulu2-7b_olympiads_controlled_contamination_original"
DATA_DIR="$DATA_PATH/olympiads_rl/olympiads_rl_lexical_trio_v3_unique_ratio_penalty_2.0_augment_random_7_seed2_prefix_0.25"
OUT_DIR="$EVAL_PATH/mia_olympiads_original_baselines"
MEMBERS_FILE="$DATA_DIR/olympiads_32_members.jsonl"
NONMEMBERS_FILE="$DATA_DIR/olympiads_32_nonmembers.jsonl"
mkdir -p "$OUT_DIR"


echo "Running MIA baselines on toy data with model: $MODEL"

declare -a ATTACKS=(
  "loss"
  "zlib"
  "min_k"
  "min_k++"
  "gradnorm"
)

# Optional reference-based attack if a small ref model is available
ATTACK_REF="loss_ref_tulu2-7b"
REF_MODEL="allenai/tulu-2-7b"

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
  if [[ "$ATTACK" == "gradnorm" ]]; then
    python -m adra.scripts.evaluate_mia \
      --members "$OUT_DIR/${ATTACK}_members.jsonl" \
      --nonmembers "$OUT_DIR/${ATTACK}_nonmembers.jsonl" \
      --output "$OUT_DIR/${ATTACK}_metrics.json" \
      --higher-is-member
  else
    python -m adra.scripts.evaluate_mia \
      --members "$OUT_DIR/${ATTACK}_members.jsonl" \
      --nonmembers "$OUT_DIR/${ATTACK}_nonmembers.jsonl" \
      --output "$OUT_DIR/${ATTACK}_metrics.json"
  fi
done


### * REFERENCE-BASED ATTACKS * ###
# Reference-based attack (optional; uses same tiny model as reference for smoke test)
echo "Attack: ref (members)"
python -m adra.scripts.run_mia \
  --model "$MODEL" \
  --dataset "$MEMBERS_FILE" \
  --reference-model "$REF_MODEL" \
  --attack ref \
  --max-length 4096 \
  --output "$OUT_DIR/${ATTACK_REF}_members.jsonl"

echo "Attack: ref (nonmembers)"
python -m adra.scripts.run_mia \
  --model "$MODEL" \
  --dataset "$NONMEMBERS_FILE" \
  --reference-model "$REF_MODEL" \
  --attack ref \
  --max-length 4096 \
  --output "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl"

echo "Evaluate: ref"
python -m adra.scripts.evaluate_mia \
  --members "$OUT_DIR/${ATTACK_REF}_members.jsonl" \
  --nonmembers "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl" \
  --output "$OUT_DIR/${ATTACK_REF}_metrics.json"

echo "Done. Results in $OUT_DIR"
