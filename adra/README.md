# adra

Core library for membership inference attacks (MIA) and active data reconstruction evaluation on language models.

## Structure

```
adra/
├── mia/                  # MIA baselines: Loss, Reference, Zlib, Min-K, Min-K++
├── scripts/              # CLI entrypoints for running MIA, evaluating reconstructions, and producing checkpoints
├── utils/                # Result formatting, parquet inspection, AUROC extraction, and plotting
├── utils_rl/             # Reconstruction evaluation, checkpoint management, data mixing, LLM judge & embedding deployment
└── dataset_paraphrase/   # LLM-based dataset paraphrasing via LiteLLM for non-member generation
```

### `mia/` — MIA Baselines

Implements six standard membership inference attacks through a unified `BaseAttack` interface:

| Attack | Description |
|--------|-------------|
| **Loss** | Per-token log-likelihood of the target model |
| **Reference** | Likelihood ratio between target and a reference model |
| **Zlib** | Log-likelihood normalized by zlib compression length |
| **Min-K%** | Mean of the lowest-*k*% token probabilities |
| **Min-K++** | Mean-shifted, variance-normalized Min-K |
| **GradNorm** | Gradient norm of the loss w.r.t. model parameters |

Also provides dataset loaders with chat-template support (`datasets.py`), HuggingFace model adapters (`model_adapter.py`), and a single-command runner (`runner.py`).

### `scripts/` — CLI Entrypoints

- **`run_mia.py`** — Run any MIA baseline on a HuggingFace or local dataset and produce per-sample score JSONL files.
- **`evaluate_mia.py`** — Compute AUROC, PR-AUC, TPR@FPR, and bootstrapped confidence intervals from member/non-member score files.
- **`evaluate_reconstructions.py`** — Evaluate reconstruction quality from parquet outputs: lexical (Jaccard, LCS, n-gram coverage), embedding (FastText, Qwen3), BLEURT, and LLM-as-judge metrics. Supports MIA-weighted aggregation, budget forcing, and member/non-member splitting.
- **`generate_lexical_match_dataset.py`** / **`generate_embedding_match_dataset.py`** — Build match datasets for lexical or embedding similarity.
- **`produce_checkpoints/`** — Produce contaminated fine-tuning checkpoints (SFT/LoRA) for controlled MIA experiments.

### `utils_rl/` — RL & Evaluation Utilities

- **`reconstruction_evaluation.py`** — Comprehensive reconstruction evaluation engine. Computes lexical (Jaccard, LCS, LCS ratio, token overlap, n-gram coverage), embedding (FastText, Qwen3-Embedding), BLEURT, LLM-as-judge, and math-correctness metrics with parallel processing and MIA-weighted aggregation.
- **`checkpoint_manager.py`** — Lightweight wrapper around HF Transformers `Trainer` for fine-tuning causal LMs on small corpora (SFT and LoRA).
- **`data_mixer.py`** — Mix contamination datasets with general SFT data at controlled ratios for checkpoint production.
- **`llm_judge_local.py`** / **`llm_judge_remote.py`** — Local and remote (vLLM-served) LLM-as-judge scoring with batched inference.
- **`embedding_client.py`** / **`embedding_model_deploy/`** — Embedding model client and TEI deployment scripts.
- **`llm_judge_deploy/`** — vLLM server deployment scripts for remote LLM judge.
- **`ngram_coverage.py`** — Efficient n-gram coverage computation between candidate and reference texts.
- **`merge_lora.py`** — Merge LoRA adapters back into base models.
- **`process_data/`** — Dataset-specific preprocessing (WildChats, Olympiads, ArXiv, AIME, Aya, Tulu).

### `dataset_paraphrase/` — Paraphrased Data Generation for LLM-MIA dasets

LLM-based paraphrasing pipeline for generating non-member counterparts of training data. Uses LiteLLM for provider-agnostic API access (Gemini, OpenAI, etc.) with batch processing, retry logic, and configurable extraction. Includes dataset-specific paraphrasers for BookMIA, AIME, Olympiads, ArXiv, and WikiMIA.

## Installation

From the repo root:

```bash
pip install -e .
```
