<div align="center">

# Learning to Detect Language Model Training Data via Active Reconstruction

[**Paper**](TODO) • [**Data & Models**](https://huggingface.co/ADRA-RL)
</div>

We propose Active Data Reconstruction Attack (ADRA), a family of MIA that actively induces a model to reconstruct a given text through training. 

# Overview

This repository contains three main components:

- **[`adra/`](adra/)**: Core library for membership inference attacks and reconstruction evaluation. Implements standard MIA baselines (Loss, Zlib, Min-K, Min-K++, Reference), comprehensive reconstruction metrics (lexical, embedding, LLM-as-judge), LLM-based dataset paraphrasing for datasets, and controlled contamination & model distillation.

- **[`verl/`](verl/)**: RL training code based on [verl](https://github.com/verl-project/verl) for RL with GRPO, reconstruction rewards, and contrastive rewards.
  - **[`verl/examples/data_preprocess/`](verl/examples/data_preprocess/)**: Prepares candidate data pools (e.g. BookMIA, AIME) into RL-ready training data.
  - **[`verl/verl/utils/reward_score/`](verl/verl/utils/reward_score/)**: Reward functions including lexical reconstruction, embedding similarity, and LLM-as-judge rewards with contrastive reward formulation.

- **[`scripts/`](scripts/)**: scripts to process data, run baselines, launch ADRA (RL) training, and evaluate MIA & reconstruction performances.

For detailed setup and usage instructions, see the README files in each subdirectory.


# Set-up

All trainings & evaluations were done on a single node with 8 H200s. Hyperparameters in the scripts may need adjusting for your hardware.

### Prerequisites

- NVIDIA GPU(s) with CUDA 12.x compatible drivers
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- GCC and CUDA toolkit accessible via your system or module manager

### Installation

```bash
# 1. Create and activate a fresh conda env
conda create -n adra python=3.10
conda activate adra

# 2. Clone the repo
git clone https://github.com/oseyosey/MIA-RL.git
cd MIA-RL

# 3. Run the install script (review it first — you may need to update
#    module names, CUDA paths, and conda paths for your system)
bash adra_setup.sh
```

> **Note:** Before running `adra_setup.sh`, open it and update the system-specific lines at the top (GCC/CUDA module names, `CUDA_HOME` path, and conda path) to match your system / cluster. See [`requirements.txt`](requirements.txt) for the full list of pinned package versions.


# ADRA Usage

Below we walk through the AIME post-training pipeline as a quick-start example. See [`scripts/README.md`](scripts/README.md) for the full step-by-step guide and per-script documentation.

## Training

1. **Prepare data** -- Build the MIA training parquet (member/non-member splits, lexical reward profiles, optional MIA weighting for ADRA+):
   ```bash
   bash scripts/post-training/aime/prepare_aime_mia_data_lexical_adra.sh        # ADRA
   bash scripts/post-training/aime/prepare_aime_mia_data_lexical_adra-plus.sh    # ADRA+
   ```

2. **Launch RL training** (GRPO with lexical reward, Slurm):
   ```bash
   sbatch scripts/post-training/aime/submit_run_aime_adra_original_lora_h200_8.sh
   # or bash
   bash scripts/post-training/aime/submit_run_aime_adra_original_lora_h200_8.sh
   ```


Datasets and models are released at [huggingface.co/ADRA-RL](https://huggingface.co/ADRA-RL). You may also skip training and directly download the checkpoints for evaluation.

## Evaluation

1. **MIA baselines** -- Run standard attacks (loss, zlib, min-k, min-k++, gradnorm, ref) on the SFT model:
   ```bash
   bash scripts/post-training/aime/run_mia_aime_original_baselines.sh
   ```

2. **N-sampling eval** -- Generate `n` samples from the SFT model and compute lexical MIA metrics:
   ```bash
   bash scripts/post-training/aime/run_mia_aime_n-sampling_eval.sh
   ```

3. **RL checkpoint eval** -- Merge a LoRA checkpoint into the base model, generate, and evaluate:
   - **Full sweep** (loops over global steps): `run_mia_aime_adra_rl_eval_full.sh`
   - **Quick eval** (single HF checkpoint): `run_mia_aime_adra_rl_eval_quick.sh`

## Adapting to your own dataset

We provide three dataset-agnostic **boilerplate scripts** at `scripts/` that you can copy and fill in for a new dataset:

| Script | What it does |
|--------|--------------|
| [`run_mia_baselines.sh`](scripts/run_mia_baselines.sh) | Run MIA baseline attacks on any member/non-member split |
| [`run_mia_n-sampling_eval.sh`](scripts/run_mia_n-sampling_eval.sh) | Generate samples and compute lexical MIA metrics |
| [`run_mia_rl_eval_quick.sh`](scripts/run_mia_rl_eval_quick.sh) | End-to-end: merge LoRA, generate, and evaluate |

Each contains `TODO` placeholders for paths and model IDs. See [`scripts/README.md`](scripts/README.md) for details on what to fill in.

