# Remote LLM Judge Reward Function

Remote LLM-as-a-judge reward function that connects to a vLLM server (e.g., Qwen3-32B). Delegates GPU-accelerated inference to a separate server so reward workers need no GPU.

## Quick Start

### 1. Deploy the vLLM Server

```bash
# Using Python server (recommended)
cd adra/utils_rl/llm_judge_deploy/
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000

# Or using Docker
cd adra/utils_rl/llm_judge_deploy/
./run_vllm_docker.sh
```

### 2. Configure

```bash
export LLM_JUDGE_SERVER_URL="http://localhost:8000"
# For multi-node: use the actual IP from server output
# export LLM_JUDGE_SERVER_URL="http://172.16.0.60:8000"
```

### 3. Use in VERL Training

```yaml
data_source: llm_judge_remote_your_dataset
```

Or as a custom reward function:
```yaml
custom_reward_function:
  path: verl/utils/reward_score/llm_judge_remote.py
  name: compute_score
```

## Architecture

```
┌─────────────────────┐         HTTP/REST API      ┌──────────────────────┐
│   VERL Training     │ ─────────────────────────> │  vLLM Server         │
│   (Reward Worker)   │ <───────────────────────── │  Qwen3-32B           │
│   - No GPU needed   │      Text Generation       │  - GPU accelerated   │
└─────────────────────┘                            └──────────────────────┘
```

## Features

### Drop-in Replacement
Same API as `llm_judge.py`:
```python
score = compute_score(
    data_source="llm_judge_remote_test",
    solution_str="Your model output",
    ground_truth="Reference answer",
    extra_info={"problem": "What is 2+2?"}
)
```

### Batched Processing
```python
scores = compute_score_batched(
    data_sources=["test1", "test2", ...],
    solution_strs=["output1", "output2", ...],
    ground_truths=["reference1", "reference2", ...],
    extra_infos=[{"problem": "prob1"}, {"problem": "prob2"}, ...]
)
```

### Prompt Templates
Three built-in templates (see `llm_judge_prompts.py`):
```python
# V1 (default) - basic similarity with evaluation criteria, includes {PROBLEM}
extra_info = {"prompt_template": "default"}  # or "v1"

# V2.1 - comprehensive 5-criteria with surface/semantic resemblance
extra_info = {"prompt_template": "v2_1"}

# V3 - additive scoring with hard disqualifiers, 5 criteria x 0.20
extra_info = {"prompt_template": "v3"}
```

You can also pass a custom template string directly:
```python
extra_info = {
    "prompt_template": "Rate similarity... {REFERENCE_SOLUTION} vs {CANDIDATE_SOLUTION}\nREWARD: <score>"
}
```

### Thinking Mode
```python
extra_info = {"enable_thinking": True, "model_name": "Qwen/Qwen3-32B"}
```

## Configuration

### Environment Variables
- `LLM_JUDGE_SERVER_URL`: Default server URL
- `LLM_JUDGE_SERVER_API_KEY`: Optional authentication key
- `LLM_JUDGE_SERVER_TIMEOUT`: Request timeout in seconds (default: 60)

### Extra Info Parameters
```python
extra_info = {
    # Server
    "server_url": "http://custom-server:8000",
    "api_key": "your-api-key",
    "timeout": 90,

    # LLM generation
    "model_name": "Qwen/Qwen3-32B",
    "temperature": 0.7,
    "top_p": 0.8,
    "max_new_tokens": 512,
    "enable_thinking": False,
    "batch_size": 128,

    # Prompt
    "problem": "Math problem statement",
    "prompt_template": "default",  # "v1", "v2_1", "v3", or a template string

    # Reference filtering
    "target_gt": "specific_answer",
    "filter_gt_by_prompt_token": True,
    "prompt": "The answer is...",
}
```

## Health Check

```bash
curl http://your-server:8000/health
curl http://your-server:8000/v1/models
```
