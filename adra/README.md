# adra

Core package for membership inference attacks (MIA) and active data reconstruction on language models.

## Structure

```
adra/
├── mia/                  # Membership inference attacks (loss, reference, zlib, min-k, gradnorm)
├── scripts/              # CLI entrypoints for running MIA and evaluating reconstructions
├── utils/                # Helpers for inspecting parquet files and formatting MIA AUROC results
├── utils_rl/             # Fine-tuning checkpoint management, reconstruction evaluation metrics, data mixing
└── dataset_paraphrase/   # LLM-based dataset paraphrasing via LiteLLM (Gemini, etc.)
```

## Installation

From the repo root:

```bash
pip install -e .
```
