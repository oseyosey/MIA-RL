#!/bin/bash
#SBATCH -J DOLMA3_ARXIV_ADRA_PLUS_ORIGINAL_LORA_H200_8  # Job name
#SBATCH -o slurm_out/DOLMA3_ARXIV_ADRA_PLUS_ORIGINAL_LORA_H200_8.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e slurm_out/DOLMA3_ARXIV_ADRA_PLUS_ORIGINAL_LORA_H200_8.e%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 8   # Total number of CPU cores requrested
#SBATCH --mem=1200gb    # CPU Memory pool for all cores
#SBATCH -t 24:00:00    # Run time (hh:mm:ss)
#SBATCH --requeue
#SBATCH --account=
#SBATCH --partition=gpu-h200 --gpus=8 --nodes=1   # Request 8 GPUs on a single node

nvidia-smi

# --- adra-v1 environment (Olmo3: CUDA_DEVICE_MAX_CONNECTIONS=1, VLLM_ALLREDUCE_USE_SYMM_MEM=0) ---
# module load gcc/13.4.0
# module load cuda/12.9.1
# export CUDA_HOME="/gpfs/software/cuda/12.9.1"
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

unset ROCR_VISIBLE_DEVICES
set -x


# Define list of prompt templates to iterate through
LEXICAL_METRIC_TEMPLATES=("trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max")  # Add your desired prompt template values here
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=7

n_gpus=8
n_cpus=80

# Set hyper-parameters
temperature=1.0
top_p=0.95
top_k=50
seed=1
rollout_n=32
lr=5e-5

# mia model loss we use
# mia_weight_tag="min_k++"

# Define output path and experiment name for rollout data directory
OUTPUT_PATH="./outputs"
PROJECT_NAME="verl_olmo3-7b-instruct_original_lexical_seed${seed}"
SFT_MODEL_PATH="allenai/Olmo-3-7B-Instruct"

# Use dynamic max_ngram for ngram coverage computation
export ADRA_USE_DYNAMIC_MAX_NGRAM=true
export ADRA_USE_PROCESS_POOL=true

# Seq length
PROMPT_LENGTH=2048
RESPONSE_LENGTH=1024

# Loop through each prompt template
for PROMPT_TEMPLATE in "${LEXICAL_METRIC_TEMPLATES[@]}"; do
    echo "Starting training with PROMPT_TEMPLATE: $PROMPT_TEMPLATE"

    # Set template-specific variables (TODO: set DATA_PATH)
    DATA_DIR="DATA_PATH/dolma3-arxiv_rl/dolma3-arxiv-mia-1k_64_rl_lexical_${PROMPT_TEMPLATE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${seed}_prefix_0.25_assist_0.25"
    EXP_NAME="verl_olmo3-7b-instruct_dolma3-arxiv_seed${seed}_original_${PROMPT_TEMPLATE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_temp${temperature}_topp${top_p}_topk${top_k}_rollout${rollout_n}_lora_prefix_0.25_assist_0.25"
    
    echo "Data directory: $DATA_DIR"
    echo "Experiment name: $EXP_NAME"
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Data directory $DATA_DIR does not exist. Skipping $PROMPT_TEMPLATE"
        continue
    fi
    
    # conda run -p /gscratch/h2lab/osey/envs/ddrl 
    python3 -m verl.trainer.main_ppo \
        ray_init.num_cpus=${n_cpus} \
        algorithm.adv_estimator=grpo \
        reward_model.reward_manager=batch \
        reward_model.reward_kwargs.truncate_prefix_ratio=0.25 \
        reward_model.reward_kwargs.num_workers=16 \
        data.assistant_prefix_key=assistant_prefix \
        data.train_files="$DATA_DIR"/train.parquet \
        data.val_files="$DATA_DIR"/train.parquet \
        data.train_batch_size=64 \
        data.max_prompt_length=${PROMPT_LENGTH} \
        data.max_response_length=${RESPONSE_LENGTH} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.lora_rank=64 \
        actor_rollout_ref.model.lora_alpha=128 \
        actor_rollout_ref.model.path="$SFT_MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=${lr} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.005 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=${rollout_n} \
        actor_rollout_ref.rollout.disable_log_stats=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXP_NAME} \
        trainer.n_gpus_per_node=${n_gpus} \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=-1 \
        trainer.total_epochs=20 "$@" \
        trainer.rollout_data_dir=${OUTPUT_PATH}/${EXP_NAME}/rollout_data \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k}
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
    else
        echo "Training failed for PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
    fi
    
    echo "Finished training with PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
    echo "----------------------------------------"
done

echo "All training runs completed!"