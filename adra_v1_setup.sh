#!/usr/bin/env bash
# ==============================================================================
# ADRA / MIA-RL Environment Setup
# Installs all dependencies into an existing conda env.
# See requirements.txt for a flat list of pinned packages.
# ==============================================================================

set -euo pipefail
set -x

# --- System modules ----------------------------------------------------------
#TODO: Adjust these to match your cluster's module system and available versions.
# You need a C/C++ compiler (GCC) and a CUDA toolkit that matches your GPU driver.
module load gcc/13.4.0       # Change to your cluster's GCC module
module load cuda/12.9.1      # Must match (or be compatible with) your GPU driver

# Point build tools at the CUDA installation.
#TODO: Update this path if your CUDA toolkit lives elsewhere.
export CUDA_HOME=/gpfs/software/cuda/12.9.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- Conda -------------------------------------------------------------------
#TODO: Update this to wherever your conda/miniconda is installed.
source /gpfs/projects/h2lab/osey/miniconda3/etc/profile.d/conda.sh

#TODO: Replace with your own env path or name.
conda activate /gpfs/projects/h2lab/osey/envs/adra

# --- vLLM (install first — it pins many transitive deps) ---------------------
pip install vllm==0.11.0

# --- VERL deps ---------------------------------------------------------------
pip install tensordict==0.10.0 torchdata==0.11.0

# --- HuggingFace + training ecosystem ----------------------------------------
pip install transformers==4.57.6 accelerate==1.12.0 datasets==4.5.0 peft==0.17.1 hf-transfer==0.1.9 \
    "numpy==2.2.6" "pyarrow>=15.0.0" pandas==2.3.3 \
    ray==2.53.0 codetiming==1.4.0 hydra-core==1.3.2 pylatexenc==2.10 qwen-vl-utils==0.0.14 wandb==0.24.0 dill==0.4.0 pybind11==3.0.1 liger-kernel==0.6.4 mathruler==0.1.0 \
    pytest==9.0.2 py-spy==0.4.1 pyext==0.7 pre-commit==4.5.1 ruff==0.14.14

# --- Serving / API deps (used by vLLM's API server) -------------------------
pip install "nvidia-ml-py==13.590.48" "fastapi==0.128.0" "optree==0.18.0" "pydantic>=2.9" "grpcio>=1.62.1"

# --- OpenCV (headless fix for clusters without display) ----------------------
pip install opencv-python==4.13.0.90
pip install opencv-fixer==0.2.5 && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

# --- Transformer Engine (requires NVIDIA's PyPI index) -----------------------
pip install --extra-index-url https://pypi.nvidia.com --pre transformer-engine==2.11.0

# --- VERL (local editable install, no deps — they're installed above) --------
cd verl
pip install --no-deps -e .
cd ..

# --- Flash Attention (needs CUDA compiler, skip build isolation) -------------
pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3

# --- DDRL, MIA, reward scoring, evaluation -----------------------------------
pip install trl==0.19.1 python-Levenshtein==0.27.3 rank_bm25==0.2.2 math_verify==0.9.0 scikit-learn==1.7.2 evaluate==0.4.6 seaborn==0.13.2 nvidia-ml-py==13.590.48 rich==14.2.0 peft==0.17.1 sentence-transformers==5.2.0

pip install omegaconf==2.3.0

# --- ADRA (local editable install, no deps — they're installed above) --------
pip install --no-deps -e .