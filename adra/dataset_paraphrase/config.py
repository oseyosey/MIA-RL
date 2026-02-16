"""
Configuration module for dataset paraphrasing.

This module defines the configuration structure for paraphrasing datasets
using LLM APIs through LiteLLM.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class FieldConfig:
    """Configuration for paraphrasing a specific field in the dataset."""
    
    input_field: str
    """The name of the field to be paraphrased."""
    
    output_field: str
    """The name of the field where the paraphrased content will be stored."""
    
    prompt: str
    """The prompt template to use for this field. Use {input} as placeholder for the field content."""
    
    max_tokens: Optional[int] = None
    """Maximum tokens to generate for this field."""
    
    temperature: float = 0.7
    """Temperature for generation (0.0 to 1.0)."""

    output_label: Optional[str] = None
    """Optional label used in model output, e.g., 'rewrite_output'. If provided, extractor will use it."""

    strip_quotes: bool = True
    """If True, remove wrapping quotes around the extracted output."""
    
    thinking_enabled: bool = False
    """If True, enable thinking mode for models that support it (e.g., Gemini)."""
    
    reasoning_effort: Optional[str] = None
    """Reasoning effort level for models that support it. Options: 'disable', 'low', 'medium', 'high'."""


@dataclass
class ParaphraseConfig:
    """Main configuration for dataset paraphrasing."""
    
    # Model configuration
    model: str = "gemini/gemini-2.5-flash"
    """The LiteLLM model identifier to use."""
    
    api_key: Optional[str] = None
    """API key for the model provider. If None, will look for environment variable."""
    
    # Dataset configuration
    dataset_path: str = ""
    """HuggingFace dataset path or local path to load."""
    
    dataset_name: Optional[str] = None
    """Dataset configuration name if applicable."""
    
    dataset_split: str = "train"
    """Which split of the dataset to process."""
    
    # Field configurations
    field_configs: List[FieldConfig] = field(default_factory=list)
    """List of field configurations for paraphrasing."""
    
    # Processing configuration
    batch_size: int = 1
    """Number of examples to process in a single batch API call. Higher values improve efficiency but use more memory."""
    
    enable_batching: bool = False
    """Whether to use LiteLLM's batch_completion for processing multiple examples at once."""
    
    max_examples: Optional[int] = None
    """Maximum number of examples to process. If None, process all."""
    
    # Output configuration
    output_path: Optional[str] = None
    """Path to save the output dataset. If None, will not save locally."""
    
    push_to_hub: bool = False
    """Whether to push the resulting dataset to HuggingFace Hub."""
    
    hub_repo_id: Optional[str] = None
    """HuggingFace Hub repository ID to push to."""
    
    hub_private: bool = True
    """Whether the Hub repository should be private."""
    
    # LiteLLM specific configuration
    request_timeout: int = 60
    """Timeout for LLM API requests in seconds."""
    
    max_retries: int = 10
    """Maximum number of retries for failed requests."""
    
    retry_delay: float = 10.0
    """Delay between retries in seconds."""
    
    # Additional model parameters
    model_kwargs: Dict[str, any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the model."""
    
    # Manual overrides
    manual_paraphrases: Dict = field(default_factory=dict)
    """Dictionary of manual paraphrase overrides keyed by (book_id, snippet_id) or similar identifiers."""
    
    def add_field_config(
        self,
        input_field: str,
        output_field: str,
        prompt: str,
        **kwargs
    ) -> None:
        """
        Add a field configuration for paraphrasing.
        
        Args:
            input_field: The name of the field to be paraphrased
            output_field: The name of the field where the paraphrased content will be stored
            prompt: The prompt template to use for this field
            **kwargs: Additional arguments for FieldConfig
        """
        field_config = FieldConfig(
            input_field=input_field,
            output_field=output_field,
            prompt=prompt,
            **kwargs
        )
        self.field_configs.append(field_config)
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided")
        
        if not self.field_configs:
            raise ValueError("At least one field configuration must be provided")
        
        if self.push_to_hub and not self.hub_repo_id:
            raise ValueError("hub_repo_id must be provided when push_to_hub is True")
        
        # Check for duplicate output fields
        output_fields = [fc.output_field for fc in self.field_configs]
        if len(output_fields) != len(set(output_fields)):
            raise ValueError("Output fields must be unique")
        
        # Validate prompts contain {input} placeholder
        for fc in self.field_configs:
            if "{input}" not in fc.prompt:
                raise ValueError(f"Prompt for field {fc.input_field} must contain {{input}} placeholder")
