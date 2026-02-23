#!/usr/bin/env bash
set -euo pipefail

MODEL="allenai/Olmo-3-7B-Instruct"
DATA_DIR="/gpfs/scrubbed/osey/Dataset_Distillation/data/dolma3-arxiv_rl/dolma3-arxiv-mia-1k-1024_paraphrased_64_rl_lexical_trio_v3_unique_ratio_penalty_1.50_augment_random_7_seed2_prefix_0.25_assist_0.25"
OUT_DIR="/gpfs/scrubbed/osey/Dataset_Distillation/eval/M9.4_paraphrased_baselines/mia_dolma3-arxiv-1024_m9.4_paraphrased_olmo3-7b-instruct_seed2"
MEMBERS_FILE="$DATA_DIR/dolma3-arxiv_64_members.jsonl"
NONMEMBERS_FILE="$DATA_DIR/dolma3-arxiv_64_nonmembers.jsonl"
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
# ATTACK_REF="loss_ref_llama3.1_8b"
# REF_MODEL="meta-llama/Llama-3.1-8B"

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
# Reference-based attack (optional; uses same tiny model as reference for smoke test) 
# echo "Attack: ref (members)"
# python -m ddrl.scripts.run_mia \
#   --model "$MODEL" \
#   --dataset "$MEMBERS_FILE" \
#   --reference-model "$REF_MODEL" \
#   --attack ref \
#   --max-length 4096 \
#   --output "$OUT_DIR/${ATTACK_REF}_members.jsonl"

# echo "Attack: ref (nonmembers)"
# python -m ddrl.scripts.run_mia \
#   --model "$MODEL" \
#   --dataset "$NONMEMBERS_FILE" \
#   --reference-model "$REF_MODEL" \
#   --attack ref \
#   --max-length 4096 \
#   --output "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl"

# echo "Evaluate: ref"
# python -m ddrl.scripts.evaluate_mia \
#   --members "$OUT_DIR/${ATTACK_REF}_members.jsonl" \
#   --nonmembers "$OUT_DIR/${ATTACK_REF}_nonmembers.jsonl" \
#   --output "$OUT_DIR/${ATTACK_REF}_metrics.json"

# echo "Done. Results in $OUT_DIR"


