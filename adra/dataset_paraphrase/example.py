"""
Example script demonstrating how to use the dataset paraphrasing module.

This script shows various ways to configure and use the DatasetParaphraser.
"""

import os
import sys
from pathlib import Path

# Make the `ADRA` directory importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adra.dataset_paraphrase import DatasetParaphraser, ParaphraseConfig


def example_basic():
    """Basic example of paraphrasing a single field."""
    print("=== Basic Example ===")
    
    # Create configuration
    config = ParaphraseConfig(
        model="gemini/gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY"),  # Or provide directly
        dataset_path="imdb",  # Example with IMDB dataset
        dataset_split="train",
        max_examples=5  # Process only 10 examples for demo
    )
    # Set a default output path (absolute) for this example
    default_out = PROJECT_ROOT / "outputs" / "paraphrased" / "imdb_train_example"
    config.output_path = str(default_out)
    
    # Add a field configuration
    config.add_field_config(
        input_field="text",
        output_field="paraphrased_text",
        prompt="Rewrite the following movie review in a different style while keeping the same sentiment: \"{input}\". Output directly the paraphrased output after \"rewrite_output:\"",
        max_tokens=4096,
        temperature=0.7
    )
    
    # Create paraphraser and run
    paraphraser = DatasetParaphraser(config)
    dataset = paraphraser.run()
    
    # Show a sample - guard against None results
    original = dataset[0].get('text')
    paraphrased = dataset[0].get('paraphrased_text')
    if isinstance(original, str):
        print(f"\nOriginal: {original[:200]}...")
    else:
        print("\nOriginal: <missing>")
    if isinstance(paraphrased, str):
        print(f"\nParaphrased: {paraphrased[:200]}...")
    else:
        print("\nParaphrased: <None>")
    print(f"\nSaved to: {config.output_path}")


def example_multiple_fields():
    """Example of paraphrasing multiple fields with different prompts."""
    print("\n=== Multiple Fields Example ===")
    
    config = ParaphraseConfig(
        model="gemini/gemini-2.5-flash",
        dataset_path="squad",  # Question-answering dataset
        dataset_split="validation",
        max_examples=5
    )
    
    # Add multiple field configurations
    config.add_field_config(
        input_field="question",
        output_field="simple_question",
        prompt="Simplify this question for a child: {input}",
        temperature=0.5
    )
    
    config.add_field_config(
        input_field="question",
        output_field="formal_question",
        prompt="Rewrite this question in a more formal, academic style: {input}",
        temperature=0.3
    )
    
    config.add_field_config(
        input_field="context",
        output_field="summary",
        prompt="Provide a brief summary of this text in 2-3 sentences: {input}",
        max_tokens=100,
        temperature=0.5
    )
    
    # Run paraphrasing
    paraphraser = DatasetParaphraser(config)
    dataset = paraphraser.run()
    
    # Show results
    example = dataset[0]
    print(f"\nOriginal question: {example['question']}")
    print(f"Simple version: {example['simple_question']}")
    print(f"Formal version: {example['formal_question']}")
    print(f"\nContext summary: {example['summary']}")


def example_with_output():
    """Example showing how to save and push to Hub."""
    print("\n=== Output Example ===")
    
    config = ParaphraseConfig(
        model="gemini/gemini-2.5-flash",
        dataset_path="glue",
        dataset_name="sst2",  # Sentiment classification dataset
        dataset_split="train",
        max_examples=100,
        output_path="./paraphrased_sst2",  # Save locally
        push_to_hub=False,  # Set to True to push to Hub
        hub_repo_id="your-username/paraphrased-sst2",  # Replace with your repo
        hub_private=True
    )
    
    config.add_field_config(
        input_field="sentence",
        output_field="paraphrased_sentence",
        prompt="Rewrite this sentence with different words but keep exactly the same meaning: {input}",
        temperature=0.6
    )
    
    paraphraser = DatasetParaphraser(config)
    dataset = paraphraser.run()
    
    print(f"\nProcessed {len(dataset)} examples")
    print(f"Dataset saved to: {config.output_path}")


def example_custom_model_params():
    """Example using custom model parameters."""
    print("\n=== Custom Model Parameters Example ===")
    
    config = ParaphraseConfig(
        model="gemini/gemini-2.5-flash",
        dataset_path="ag_news",
        dataset_split="train",
        max_examples=20,
        model_kwargs={
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1
        }
    )
    
    config.add_field_config(
        input_field="text",
        output_field="headline",
        prompt="Convert this news article into a catchy headline (max 10 words): {input}",
        max_tokens=30,
        temperature=0.8
    )
    
    paraphraser = DatasetParaphraser(config)
    dataset = paraphraser.run()
    
    # Show some headlines
    for i in range(3):
        print(f"\nArticle {i+1} headline: {dataset[i]['headline']}")


if __name__ == "__main__":
    # Make sure to set your API key
    # export GOOGLE_API_KEY="your-api-key-here"
    
    # Run examples
    try:
        example_basic()
        # example_multiple_fields()
        # example_with_output()
        # example_custom_model_params()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set your GOOGLE_API_KEY environment variable")
