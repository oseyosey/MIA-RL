# Scripts

This directory contains the launch scripts for the full ADRA pipeline: data preparation, MIA baseline evaluation, RL training (GRPO), and post-training MIA evaluation.

## Directory layout

```
scripts/
  run_mia_baselines.sh            # Boilerplate: MIA baseline attacks
  run_mia_n-sampling_eval.sh      # Boilerplate: n-sampling generation + lexical MIA eval
  run_mia_rl_eval_quick.sh        # Boilerplate: merge LoRA + generate + eval (single checkpoint)
  post-training/
    aime/                         # AIME 2021-2025
    ...
  distillation/
    s1/                           # Distillation using Deepseek-R1 S1 Distillation
    ...
  pre-training/
    bookmia/                      # Book MIA
    ...
  eval/                           # driver scripts for eval
```

---

## Worked example: AIME

Each of the `post-training/`, `distillation/`, and `pre-training/` directories contains dataset-specific subfolders (e.g. `aime/`) with the full set of scripts for that experiment. Below we walk through `post-training/aime/` as a representative example; other dataset folders follow the same structure and pipeline order.

### Step 1 -- Data preparation

Prepare the MIA training data (member/non-member splits with lexical reward metadata).

| Script | Description |
|--------|-------------|
| `prepare_aime_mia_data_lexical_adra.sh` | Prepare data for **ADRA** (original questions, no MIA weighting). |
| `prepare_aime_mia_data_lexical_adra-plus.sh` | Prepare data for **ADRA+** (original questions, with min-k++ MIA weighting). |
| `prepare_aime_paraphrased_mia_data_lexical_adra-plus.sh` | Prepare data for **ADRA+** (paraphrased questions, with min-k++ MIA weighting). |

Each script calls `post_training_custom_mia.py` to build a parquet file with prompts, ground-truth references, lexical reward profiles, and member/non-member labels. The ADRA+ variants additionally require MIA baseline scores (see Step 2) to weight the training signal.

### Step 2 -- MIA baselines

Run standard MIA attacks (loss, zlib, min-k, min-k++, gradnorm, reference-based) on the SFT model to produce per-example scores.

| Script | Description |
|--------|-------------|
| `run_mia_aime_original_baselines.sh` | Baselines on the **original**-question SFT model. |
| `run_mia_aime_paraphrased_baselines.sh` | Baselines on the **paraphrased**-question SFT model. |

These produce `{attack}_members.jsonl` / `{attack}_nonmembers.jsonl` files. The min-k++ scores are consumed by the ADRA+ data preparation scripts (Step 1).

### Step 3 -- N-sampling evaluation (SFT model)

Generate `n` samples from the SFT model and evaluate them with lexical MIA metrics (Jaccard, LCS, n-gram coverage, embedding cosine similarity) in both "match" (prefix-truncated) and "full suffix" modes.

| Script | Description |
|--------|-------------|
| `run_mia_aime_n-sampling_eval.sh` | N-sampling eval for the SFT model. |

### Step 4 -- RL training (GRPO)

Launch GRPO training with the lexical reward on a Slurm cluster. Each script is a self-contained `sbatch` job.

| Script | Description |
|--------|-------------|
| `submit_run_aime_adra_original_lora_h200_8.sh` | **ADRA** RL training on original questions. |
| `submit_run_aime_adra_paraphrased_lora_h200_8.sh` | **ADRA** RL training on paraphrased questions. |
| `submit_run_aime_adra-plus_original_lora_h200_8.sh` | **ADRA+** RL training on original questions. |
| `submit_run_aime_adra-plus_paraphrased_lora_h200_8.sh` | **ADRA+** RL training on paraphrased questions. |

### Step 5 -- RL checkpoint evaluation

After RL training, evaluate the resulting LoRA checkpoints.

| Script | Description |
|--------|-------------|
| `run_mia_aime_adra_rl_eval_full.sh` | **Full sweep**: loops over global steps, merges each LoRA checkpoint, generates samples, and runs lexical MIA eval. Use this to track how MIA metrics evolve across training. |
| `run_mia_aime_adra_rl_eval_quick.sh` | **Quick eval**: pulls a single base model and LoRA adapter (e.g. from HuggingFace), merges, generates, and evaluates in one shot. Use this to reproduce a single result without re-training. |

---

## Boilerplate scripts

The three scripts at `scripts/` are **dataset-agnostic** versions of the core evaluation steps. They contain the same logic as the AIME-specific scripts but with all dataset/model values replaced by `TODO` placeholders. 

You can use them to evaluate the RL lora models that we released or as starting points when adapting the pipeline to your own dataset.

### `run_mia_baselines.sh`

Runs standard MIA baseline attacks on a member/non-member split. Fill in:
- `MODEL` -- the fine-tuned model to audit
- `DATA_DIR` -- directory containing the prepared data
- `MEMBERS_FILE` / `NONMEMBERS_FILE` -- paths to the member/non-member JSONL files
- `REF_MODEL` -- reference model for the reference-based attack

### `run_mia_n-sampling_eval.sh`

Generates `n` samples from a model and evaluates them with lexical MIA metrics. Fill in:
- `BASE_PATH` -- root of this repository
- `data_path` -- path to the prepared training data parquet
- `model_dir` -- the model to evaluate (HuggingFace ID or local path)
- `eval_model_dir` -- output directory name for evaluation artifacts

### `run_mia_rl_eval_quick.sh`

End-to-end: merges a LoRA adapter into a base model, then generates and evaluates. Fill in:
- `BASE_PATH` -- root of this repository
- `data_path` -- path to the prepared training data parquet
- `base_model` -- the base (SFT) model
- `lora_adapter` -- the LoRA adapter (HuggingFace ID or local path)
- `merged_model_dir` -- local path to write the merged model
- `eval_model_dir` -- output directory name for evaluation artifacts
