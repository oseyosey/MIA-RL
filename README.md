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

All trainings & evaluations are done in a single node consisting of 8 H200s. As such, the hyperparamters set in the scripts may need to be updated depending on your computational constraint and set up. 


# ADRA Usage


## Training

Below, we will walk you through an example of AIME for post-training data detection. 


We release datasets and models at . You can download the datasets and models at https://huggingface.co/ADRA-RL  and run on our evaluation scripts for reproduction. 

## Evaluation

To evaluate, you 

Again, we will walk you through an example of AIME for post-training data detection. 


