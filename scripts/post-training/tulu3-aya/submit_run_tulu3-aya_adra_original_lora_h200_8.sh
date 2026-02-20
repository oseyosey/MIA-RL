#!/bin/bash
#SBATCH -J TULU3_AYA_ADRA_ORIGINAL_LORA_H200_8  # Job name
#SBATCH -o slurm_out/TULU3_AYA_ADRA_ORIGINAL_LORA_H200_8.o%j
#SBATCH -e slurm_out/TULU3_AYA_ADRA_ORIGINAL_LORA_H200_8.e%j
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 8   # Total number of CPU cores requrested
#SBATCH --mem=1200gb    # CPU Memory pool for all cores
#SBATCH -t 24:00:00    # Run time (hh:mm:ss)
#SBATCH --requeue
#SBATCH --account=
#SBATCH --partition=gpu-h200 --gpus=8 --nodes=1

nvidia-smi

# Ensure CUDA toolkit (nvcc) is available on compute nodes.
# Uncomment this section if using adra-v1 environment
# module load gcc/13.4.0
# module load cuda/12.9.1
# export CUDA_HOME="/gpfs/software/cuda/12.9.1"
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

unset ROCR_VISIBLE_DEVICES # might need this for specific clusters

set -x

LEXICAL_METRIC_TEMPLATES=("unique_ngram_coverage_ref_ratio_1.50")
AUGMENT_SAMPLING_METHOD="random"
AUGMENT_NUM_SAMPLES=15

n_gpus=8
n_cpus=80

# Set hyper-parameters
temperature=1.0
top_p=0.95
top_k=50
seed=1
rollout_n=32
lr=5e-5

# Define output path and experiment name for rollout data directory
OUTPUT_PATH="./outputs"
PROJECT_NAME="verl_tulu3-aya_adra_original_lora_h200_8"
SFT_MODEL_PATH="allenai/Llama-3.1-Tulu-3-8B"

# export ADRA_USE_DYNAMIC_MAX_NGRAM=true
export ADRA_USE_PROCESS_POOL=true

PROMPT_LENGTH=1024
RESPONSE_LENGTH=512

# Loop through each prompt template
for PROMPT_TEMPLATE in "${LEXICAL_METRIC_TEMPLATES[@]}"; do
    echo "Starting training with PROMPT_TEMPLATE: $PROMPT_TEMPLATE"

    # TODO: Set DATA_DIR to the output of the prepare script
    DATA_DIR="DATA_PATH/tulu3-aya_rl/tulu3-aya_rl_lexical_${PROMPT_TEMPLATE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${seed}_prefix_0.25"
    EXP_NAME="verl_tulu3-aya_adra_original_lora_h200_8_${PROMPT_TEMPLATE}_augment_${AUGMENT_SAMPLING_METHOD}_${AUGMENT_NUM_SAMPLES}_seed${seed}_lr${lr}_temp${temperature}_topp${top_p}_topk${top_k}_rollout${rollout_n}_lora"

    echo "Data directory: $DATA_DIR"
    echo "Experiment name: $EXP_NAME"

    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Data directory $DATA_DIR does not exist. Skipping $PROMPT_TEMPLATE"
        continue
    fi

    python3 -m verl.trainer.main_ppo \
        ray_init.num_cpus=${n_cpus} \
        algorithm.adv_estimator=grpo \
        reward_model.reward_manager=batch \
        reward_model.reward_kwargs.truncate_prefix_ratio=0.25 \
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
        actor_rollout_ref.rollout.tensor_model_parallel_size=${n_gpus} \
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
        trainer.total_epochs=50 "$@" \
        trainer.rollout_data_dir=${OUTPUT_PATH}/${EXP_NAME}/rollout_data \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k}

    if [ $? -eq 0 ]; then
        echo "Training completed successfully for PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
    else
        echo "Training failed for PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
    fi

    echo "Finished training with PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
    echo "----------------------------------------"
done

echo "All training runs completed!"
