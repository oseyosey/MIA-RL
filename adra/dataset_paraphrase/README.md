# Dataset Paraphrase Module

A adra module for paraphrasing datasets using various LLM APIs through LiteLLM. This module supports multiple LLM providers with a focus on Google Gemini API, and allows for customizable paraphrasing of different fields in HuggingFace-compatible datasets.

## Features

- 🚀 **Multiple LLM Support**: Uses LiteLLM to support various LLM providers (Gemini, OpenAI, Anthropic, etc.)
- 📊 **HuggingFace Integration**: Works seamlessly with HuggingFace datasets
- 🔧 **Flexible Configuration**: Configure different prompts for different fields
- 🔄 **Retry Logic**: Built-in retry mechanism for API failures
- 💾 **Multiple Output Options**: Save locally or push to HuggingFace Hub
- 📝 **CLI and Python API**: Use via command line or integrate into your code

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Or install individually
pip install litellm datasets tqdm huggingface-hub
```

## Quick Start

### Using the CLI

1. **Basic usage with a single field:**
```bash
python -m adra.dataset_paraphrase.cli \
  --dataset "imdb" \
  --field "text:paraphrased_text:Rewrite this movie review in a different style: {input}" \
  --api-key YOUR_GOOGLE_API_KEY
```

2. **Multiple fields with different prompts:**
```bash
python -m adra.dataset_paraphrase.cli \
  --dataset "squad" \
  --field "question:simple_question:Simplify this question: {input}" \
  --field "context:summary:Summarize in 2 sentences: {input}:100:0.5" \
  --output-path "./paraphrased_squad" \
  --max-examples 1000
```

3. **Using a configuration file:**
```bash
python -m adra.dataset_paraphrase.cli --config config.json
```

### Using the Python API

```python
from adra.dataset_paraphrase import DatasetParaphraser, ParaphraseConfig

# Create configuration
config = ParaphraseConfig(
    model="gemini/gemini-2.5-flash",
    api_key="YOUR_GOOGLE_API_KEY",  # Or use environment variable
    dataset_path="imdb",
    dataset_split="train",
    max_examples=100
)

# Add field configuration
config.add_field_config(
    input_field="text",
    output_field="paraphrased_text",
    prompt="Rewrite this review: {input}",
    max_tokens=500,
    temperature=0.7
)

# Run paraphrasing
paraphraser = DatasetParaphraser(config)
dataset = paraphraser.run()
```

## Configuration

### Field Configuration Format

When using the CLI, field configurations follow this format:
```
input_field:output_field:prompt[:max_tokens[:temperature]]
```

- `input_field`: Name of the field to paraphrase
- `output_field`: Name of the field to store the result
- `prompt`: Prompt template (must include `{input}` placeholder)
- `max_tokens`: (Optional) Maximum tokens to generate
- `temperature`: (Optional) Generation temperature (0.0-1.0)

### Configuration File Format

```json
{
  "model": "gemini/gemini-2.5-flash",
  "api_key": "YOUR_API_KEY",
  "dataset_path": "squad",
  "dataset_split": "validation",
  "field_configs": [
    {
      "input_field": "question",
      "output_field": "paraphrased_question",
      "prompt": "Rewrite this question: {input}",
      "max_tokens": 100,
      "temperature": 0.7
    },
    {
      "input_field": "context",
      "output_field": "summary",
      "prompt": "Summarize: {input}",
      "max_tokens": 150,
      "temperature": 0.5
    }
  ],
  "output_path": "./output/paraphrased_squad",
  "push_to_hub": true,
  "hub_repo_id": "username/paraphrased-squad",
  "hub_private": true,
  "max_examples": 1000,
  "request_timeout": 60,
  "max_retries": 3
}
```

## Supported Models

The module supports all models available through LiteLLM. Some examples:

- **Google Gemini**: `gemini/gemini-2.5-flash`
- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`
- **And many more...**

See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for the full list.

## API Keys

Set your API key using environment variables:

```bash
# For Gemini via Google AI Studio
export GOOGLE_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

Or provide it directly in the configuration.

## Advanced Usage

### Custom Model Parameters

```python
config = ParaphraseConfig(
    model="gemini/gemini-2.5-flash",
    model_kwargs={
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1
    }
)
```

### Batch Processing

The module supports LiteLLM's batch completion for improved efficiency:

```python
config = ParaphraseConfig(
    # ... other config ...
    batch_size=5,           # Process 5 examples per API call
    enable_batching=True,   # Enable batch completion
    request_timeout=60,
    max_retries=3,
    retry_delay=1.0
)
```

**CLI usage with batching:**
```bash
python -m adra.dataset_paraphrase.cli \
  --dataset "imdb" \
  --field "text:paraphrased:Rewrite: {input}" \
  --enable-batching \
  --batch-size 5
```

**Benefits of batching:**
- **Improved throughput**: Process multiple examples in a single API call
- **Reduced latency**: Fewer round trips to the API
- **Better rate limit utilization**: More efficient use of API quotas
- **Automatic fallback**: Falls back to individual calls if batch fails

### Error Handling

The module includes robust error handling:
- Automatic retries with exponential backoff
- Graceful handling of API failures
- Detailed logging for debugging

## Examples

See `example.py` for complete examples including:
- Basic single-field paraphrasing
- Multiple fields with different prompts
- Saving and uploading to HuggingFace Hub
- Using custom model parameters

## Troubleshooting

1. **API Key Issues**: Make sure your API key is set correctly either via environment variable or in the configuration.

2. **Rate Limits**: If you encounter rate limits, try:
   - Reducing `max_examples`
   - Increasing `retry_delay`
   - Using a model with higher rate limits

3. **Memory Issues**: For large datasets, process in batches by setting `max_examples`.

4. **Field Not Found**: Ensure the input field names match exactly with the dataset schema.

## Contributing

Feel free to submit issues or pull requests for improvements!

## License

This module is part of the ADRA project and follows the same license.
