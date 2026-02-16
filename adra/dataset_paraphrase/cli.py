"""
Command-line interface for dataset paraphrasing.

This script provides a CLI to paraphrase datasets using LLM APIs.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

from .config import ParaphraseConfig, FieldConfig
from .paraphrase import DatasetParaphraser


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_field_config(field_str: str) -> Dict:
    """
    Parse a field configuration string.
    
    Expected format: "input_field:output_field:prompt[:max_tokens[:temperature]]"
    
    Args:
        field_str: The field configuration string
        
    Returns:
        Dictionary with field configuration
    """
    parts = field_str.split(":")
    if len(parts) < 3:
        raise ValueError(f"Invalid field config format: {field_str}. Expected at least 'input:output:prompt'")
    
    config = {
        "input_field": parts[0],
        "output_field": parts[1],
        "prompt": parts[2]
    }
    
    if len(parts) > 3 and parts[3]:
        config["max_tokens"] = int(parts[3])
    
    if len(parts) > 4 and parts[4]:
        config["temperature"] = float(parts[4])
    
    return config


def load_config_file(config_path: str) -> Dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Paraphrase datasets using LLM APIs through LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single field
  python -m adra.dataset_paraphrase.cli \\
    --dataset "squad" \\
    --field "question:paraphrased_question:Rewrite this question in a different way: {input}" \\
    --api-key YOUR_GOOGLE_API_KEY

  # Multiple fields with different prompts
  python -m adra.dataset_paraphrase.cli \\
    --dataset "squad" \\
    --field "question:paraphrased_question:Rewrite this question: {input}" \\
    --field "context:simplified_context:Simplify this text: {input}:500:0.5" \\
    --output-path "./paraphrased_squad" \\
    --push-to-hub --hub-repo-id "myusername/paraphrased-squad"

  # Using a configuration file
  python -m adra.dataset_paraphrase.cli --config config.json

Configuration file format:
{
  "dataset_path": "squad",
  "model": "gemini/gemini-2.5-flash",
  "api_key": "YOUR_API_KEY",
  "field_configs": [
    {
      "input_field": "question",
      "output_field": "paraphrased_question",
      "prompt": "Rewrite this question: {input}",
      "max_tokens": 100,
      "temperature": 0.7
    }
  ],
  "output_path": "./paraphrased_squad",
  "push_to_hub": true,
  "hub_repo_id": "myusername/paraphrased-squad"
}
        """
    )
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    
    # Model options
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash",
                        help="LiteLLM model identifier (default: gemini/gemini-2.5-flash)")
    parser.add_argument("--api-key", type=str, help="API key for the model provider")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset path")
    parser.add_argument("--dataset-name", type=str, help="Dataset configuration name")
    parser.add_argument("--dataset-split", type=str, default="train",
                        help="Dataset split to process (default: train)")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples to process")
    
    # Field configuration
    parser.add_argument("--field", action="append", dest="fields",
                        help="Field configuration in format 'input:output:prompt[:max_tokens[:temperature]]'")
    
    # Output options
    parser.add_argument("--output-path", type=str, help="Path to save the output dataset")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo-id", type=str, help="HuggingFace Hub repository ID")
    parser.add_argument("--hub-private", action="store_true", help="Make Hub repository private")
    
    # Processing options
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--enable-batching", action="store_true", 
                        help="Enable LiteLLM batch completion for improved efficiency")
    parser.add_argument("--request-timeout", type=int, default=60,
                        help="Timeout for LLM API requests (seconds)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for failed requests")
    
    # Additional model parameters
    parser.add_argument("--model-kwargs", type=str,
                        help="Additional model parameters as JSON string")
    
    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        import litellm
        litellm.set_verbose = True
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config_dict = load_config_file(args.config)
        config = ParaphraseConfig(**config_dict)
    else:
        # Build configuration from command-line arguments
        if not args.dataset:
            parser.error("--dataset is required when not using --config")
        
        if not args.fields:
            parser.error("At least one --field is required when not using --config")
        
        config = ParaphraseConfig(
            model=args.model,
            api_key=args.api_key,
            dataset_path=args.dataset,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            max_examples=args.max_examples,
            output_path=args.output_path,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
            hub_private=args.hub_private,
            batch_size=args.batch_size,
            enable_batching=args.enable_batching,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries
        )
        
        # Parse field configurations
        for field_str in args.fields:
            field_dict = parse_field_config(field_str)
            config.add_field_config(**field_dict)
        
        # Parse model kwargs if provided
        if args.model_kwargs:
            config.model_kwargs = json.loads(args.model_kwargs)
    
    # Set API key from environment if not provided - prefer GOOGLE_API_KEY
    if not config.api_key:
        if config.model.startswith("gemini/"):
            config.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not config.api_key:
                parser.error("GOOGLE_API_KEY (preferred) or GEMINI_API_KEY environment variable, or --api-key, is required for Gemini models")
    
    # Create and run paraphraser
    try:
        logger.info("Starting dataset paraphrasing...")
        paraphraser = DatasetParaphraser(config)
        dataset = paraphraser.run()
        logger.info(f"Successfully paraphrased {len(dataset)} examples")
        
        # Print sample output
        if len(dataset) > 0:
            logger.info("\nSample paraphrased example:")
            example = dataset[0]
            for field_config in config.field_configs:
                if field_config.output_field in example:
                    logger.info(f"\n{field_config.input_field}: {example.get(field_config.input_field, 'N/A')}")
                    logger.info(f"{field_config.output_field}: {example.get(field_config.output_field, 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error during paraphrasing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
