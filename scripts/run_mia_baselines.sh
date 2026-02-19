#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Run standard MIA baseline attacks (loss, zlib, min-k, min-k++, ref)
# on a member/non-member split and compute per-attack AUROC.
#
# This is a dataset-agnostic boilerplate. Fill in the TODOs below, or copy this
# script into a dataset-specific folder and hardcode the values there.
# See scripts/post-training/aime/ for a concrete worked example.
# ------------------------------------------------------------------------------
set -euo pipefail

# TODO: Set the path to the directory containing the prepared data
DATA_DIR=""

# TODO: Set the path to the output directory for evaluation artifacts
OUT_DIR=""

# TODO: Set the HuggingFace model ID or local path to the fine-tuned model
MODEL=""

# TODO: Set paths to the member/non-member JSONL files produced by the data
# preparation step (e.g. prepare_*_mia_data_*.sh)
MEMBERS_FILE="$DATA_DIR/members.jsonl"
NONMEMBERS_FILE="$DATA_DIR/nonmembers.jsonl"

mkdir -p "$OUT_DIR"

echo "Running MIA baselines with model: $MODEL"

declare -a ATTACKS=(
  "loss"
  "zlib"
  "min_k"
  "min_k++"
  "gradnorm"
)

# TODO: Set the reference model for the reference-based attack
ATTACK_REF="loss_ref"
REF_MODEL=""

### * NON-REFERENCE BASED ATTACKS * ###
for ATTACK in "${ATTACKS[@]}"; do
  echo "Attack: $ATTACK (members)"
  python -m ddrl.scripts.run_mia \
    --model "$MODEL" \
    --dataset "$MEMBERS_FILE" \
    --attack "$ATTACK" \
    --max-length 4096 \
    --output "$OUT_DIR/${ATTACK}_members.jsonl" \
    --verbose

  echo "Attack: $ATTACK (nonmembers)"
  python -m ddrl.scripts.run_mia \
    --model "$MODEL" \
    --dataset "$NONMEMBERS_FILE" \
    --attack "$ATTACK" \
    --max-length 4096 \
    --output "$OUT_DIR/${ATTACK}_nonmembers.jsonl" \
    --verbose

  echo "Evaluate: $ATTACK"
  if [[ "$ATTACK" == "gradnorm" ]]; then
    python -m ddrl.scripts.evaluate_mia \
      --members "$OUT_DIR/${ATTACK}_members.jsonl" \
      --nonmembers "$OUT_DIR/${ATTACK}_nonmembers.jsonl" \
      --output "$OUT_DIR/${ATTACK}_metrics.json" \
      --higher-is-member
  else
    python -m ddrl.scripts.evaluate_mia \
      --members "$OUT_DIR/${ATTACK}_members.jsonl" \
      --nonmembers "$OUT_DIR/${ATTACK}_nonmembers.jsonl" \
      --output "$OUT_DIR/${ATTACK}_metrics.json"
  fi
done


### * REFERENCE-BASED ATTACKS * ###
echo "Attack: ref (members)"
python -m ddrl.scripts.run_mia \
  --model "$MODEL" \
  --dataset "$MEMBERS_FILE" \
  --reference-model "$REF_MODEL" \
  --attack ref \
  --max-length 4096 \
  --output "$OUT_DIR/${ATTACK_REF}_members.jsonl"

echo "Attack: ref (nonmembers)"
python -m ddrl.scripts.run_mia \
  --model "$MODEL" \
  --dataset "$NONMEMBERS_FILE" \
  --reference-model "$REF_MODEL" \
  --attack ref \
  --max-length 4096 \
  --output "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl"

echo "Evaluate: ref"
python -m ddrl.scripts.evaluate_mia \
  --members "$OUT_DIR/${ATTACK_REF}_members.jsonl" \
  --nonmembers "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl" \
  --output "$OUT_DIR/${ATTACK_REF}_metrics.json"

echo "Done. Results in $OUT_DIR"
