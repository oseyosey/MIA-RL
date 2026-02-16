"""
Main module for dataset paraphrasing using LiteLLM.

This module provides the DatasetParaphraser class that handles loading datasets,
calling LLM APIs, and saving the paraphrased results.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from tqdm import tqdm

import litellm
from litellm import batch_completion
from datasets import Dataset, DatasetDict, load_dataset
import re

from .config import ParaphraseConfig, FieldConfig


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetParaphraser:
    """Main class for paraphrasing datasets using LLM APIs."""
    
    def __init__(self, config: ParaphraseConfig):
        """
        Initialize the DatasetParaphraser.
        
        Args:
            config: Configuration object for paraphrasing
        """
        self.config = config
        self.config.validate()
        
        # Set up API key. Prefer GOOGLE_API_KEY if present, mirror to GEMINI_API_KEY for LiteLLM.
        if self.config.api_key:
            os.environ.setdefault("GOOGLE_API_KEY", self.config.api_key)
        if self.config.model.startswith("gemini/"):
            # LiteLLM expects GEMINI_API_KEY; mirror GOOGLE_API_KEY -> GEMINI_API_KEY
            if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
        
        # Configure LiteLLM
        litellm.set_verbose = False  # Set to True for debugging
        
        # Load dataset
        self.dataset = None
        self.processed_dataset = None

        # Backward-compatible model aliasing
        if self.config.model in ("gemini/gemini-pro", "gemini-pro"):
            logger.warning(
                "Model '%s' is deprecated or unavailable. Switching to 'gemini/gemini-2.5-flash'.",
                self.config.model,
            )
            self.config.model = "gemini/gemini-2.5-flash"
    
    def load_dataset(self) -> Dataset:
        """
        Load the dataset from HuggingFace or local path.
        
        Returns:
            The loaded dataset
        """
        logger.info(f"Loading dataset from {self.config.dataset_path}")
        
        if self.config.dataset_name:
            dataset = load_dataset(
                self.config.dataset_path,
                self.config.dataset_name,
                split=self.config.dataset_split
            )
        else:
            dataset = load_dataset(
                self.config.dataset_path,
                split=self.config.dataset_split
            )
        
        # Limit the number of examples if specified
        if self.config.max_examples:
            dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
        
        self.dataset = dataset
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    
    def _call_llm(self, prompt: str, field_config: FieldConfig) -> str:
        """
        Call the LLM API with retry logic, including null response handling.
        
        Args:
            prompt: The complete prompt to send to the LLM
            field_config: Configuration for this specific field
            
        Returns:
            The generated response text
        """
        retry_count = 0
        last_error = None
        null_response_count = 0
        max_null_retries = 10  # Additional retries specifically for null responses
        
        while retry_count < self.config.max_retries:
            try:
                # Prepare the completion arguments
                completion_args = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": field_config.temperature,
                    "timeout": self.config.request_timeout,
                    **self.config.model_kwargs
                }
                
                if field_config.max_tokens:
                    completion_args["max_tokens"] = field_config.max_tokens
                
                # Add thinking mode and reasoning effort for supported models
                if field_config.thinking_enabled and field_config.reasoning_effort and field_config.reasoning_effort != "disable":
                    if self.config.model.startswith("gemini/"):
                        completion_args["reasoning_effort"] = field_config.reasoning_effort
                elif field_config.reasoning_effort == "disable":
                    # If reasoning effort is explicitly disabled, ensure thinking is off
                    field_config.thinking_enabled = False
                
                # Call LiteLLM
                response = litellm.completion(**completion_args)
                
                # Extract the response text
                response_text = response.choices[0].message.content
                
                # Extract reasoning content if available and thinking is enabled
                reasoning_content = None
                if field_config.thinking_enabled and hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning_content = response.choices[0].message.reasoning_content
                
                # Check for content filtering
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                if finish_reason == 'content_filter':
                    logger.warning(f"Content blocked by API safety filter (finish_reason=content_filter). This content cannot be paraphrased.")
                    # Return a special marker to indicate content filtering
                    return None, None
                
                # Check for null/empty response
                if response_text is None or (isinstance(response_text, str) and len(response_text.strip()) == 0):
                    null_response_count += 1
                    if null_response_count <= max_null_retries:
                        logger.warning(f"Received null/empty response (attempt {null_response_count}/{max_null_retries}), retrying...")
                        time.sleep(self.config.retry_delay)
                        continue  # Retry without incrementing main retry counter
                    else:
                        logger.error(f"Received null/empty response {max_null_retries} times, treating as failure")
                        raise ValueError(f"Model returned null/empty response after {max_null_retries} attempts")
                
                return response_text, reasoning_content
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.config.max_retries:
                    logger.warning(f"Error calling LLM (attempt {retry_count}/{self.config.max_retries}): {e}")
                    time.sleep(self.config.retry_delay * retry_count)  # Exponential backoff
                else:
                    logger.error(f"Failed after {self.config.max_retries} retries: {e}")
                    raise last_error
    
    def _call_llm_batch(self, prompts: List[str], field_config: FieldConfig) -> List[Optional[str]]:
        """
        Call the LLM API with batch processing for multiple prompts.
        
        Args:
            prompts: List of prompts to send to the LLM
            field_config: Configuration for this specific field
            
        Returns:
            List of generated response texts (same order as input prompts)
        """
        if not prompts:
            return []
        
        retry_count = 0
        last_error = None
        
        while retry_count < self.config.max_retries:
            try:
                # Prepare batch messages
                batch_messages = []
                for prompt in prompts:
                    batch_messages.append([{"role": "user", "content": prompt}])
                
                # Prepare the completion arguments
                completion_args = {
                    "model": self.config.model,
                    "messages": batch_messages,
                    "temperature": field_config.temperature,
                    "timeout": self.config.request_timeout,
                    **self.config.model_kwargs
                }
                
                if field_config.max_tokens:
                    completion_args["max_tokens"] = field_config.max_tokens
                
                # Add thinking mode and reasoning effort for supported models
                if field_config.thinking_enabled and field_config.reasoning_effort and field_config.reasoning_effort != "disable":
                    if self.config.model.startswith("gemini/"):
                        completion_args["reasoning_effort"] = field_config.reasoning_effort
                elif field_config.reasoning_effort == "disable":
                    # If reasoning effort is explicitly disabled, ensure thinking is off
                    field_config.thinking_enabled = False
                
                # Call LiteLLM batch completion
                responses = batch_completion(**completion_args)
                
                # Extract response texts and reasoning content, handle null responses
                response_texts = []
                reasoning_contents = []
                null_indices = []
                
                for i, response in enumerate(responses):
                    if response and hasattr(response, 'choices') and len(response.choices) > 0:
                        response_text = response.choices[0].message.content
                        
                        # Extract reasoning content if available and thinking is enabled
                        reasoning_content = None
                        if field_config.thinking_enabled and hasattr(response.choices[0].message, 'reasoning_content'):
                            reasoning_content = response.choices[0].message.reasoning_content
                        
                        if response_text is None or (isinstance(response_text, str) and len(response_text.strip()) == 0):
                            response_texts.append(None)
                            reasoning_contents.append(None)
                            null_indices.append(i)
                        else:
                            response_texts.append(response_text)
                            reasoning_contents.append(reasoning_content)
                    else:
                        response_texts.append(None)
                        reasoning_contents.append(None)
                        null_indices.append(i)
                
                # If we have null responses, retry them in batches
                if null_indices:
                    response_texts, reasoning_contents = self._retry_null_responses_batch(
                        prompts, response_texts, reasoning_contents, null_indices, field_config
                    )
                
                return response_texts, reasoning_contents
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.config.max_retries:
                    logger.warning(f"Error in batch LLM call (attempt {retry_count}/{self.config.max_retries}): {e}")
                    time.sleep(self.config.retry_delay * retry_count)  # Exponential backoff
                else:
                    logger.error(f"Batch call failed after {self.config.max_retries} retries: {e}")
                    # Fall back to individual calls
                    logger.info("Falling back to individual API calls...")
                    fallback_results = [self._call_llm_single_with_fallback(prompt, field_config) for prompt in prompts]
                    response_texts = [result[0] if result else None for result in fallback_results]
                    reasoning_contents = [result[1] if result else None for result in fallback_results]
                    return response_texts, reasoning_contents
    
    def _call_llm_single_with_fallback(self, prompt: str, field_config: FieldConfig) -> Optional[tuple]:
        """Single LLM call with error handling that returns (response, reasoning) or None on failure."""
        try:
            return self._call_llm(prompt, field_config)
        except Exception as e:
            logger.error(f"Single LLM call failed: {e}")
            return None
    
    def _retry_null_responses_batch(
        self, 
        prompts: List[str], 
        response_texts: List[Optional[str]], 
        reasoning_contents: List[Optional[str]], 
        null_indices: List[int], 
        field_config: FieldConfig,
        max_batch_retries: int = 10
    ) -> tuple:
        """
        Retry null responses using batch processing with multiple retry attempts.
        
        Args:
            prompts: Original list of all prompts
            response_texts: Current response texts (with some nulls)
            reasoning_contents: Current reasoning contents (with some nulls)
            null_indices: Indices of null responses to retry
            field_config: Configuration for this specific field
            max_batch_retries: Maximum number of batch retry attempts (default: 5)
            
        Returns:
            Updated (response_texts, reasoning_contents) tuple
        """
        current_null_indices = null_indices.copy()
        retry_attempt = 0
        
        while current_null_indices and retry_attempt < max_batch_retries:
            retry_attempt += 1
            logger.warning(
                f"Batch retry attempt {retry_attempt}/{max_batch_retries}: "
                f"Retrying {len(current_null_indices)} null responses in batch"
            )
            
            # Collect prompts for current null indices
            retry_prompts = [prompts[idx] for idx in current_null_indices]
            
            try:
                # Prepare batch messages
                batch_messages = []
                for prompt in retry_prompts:
                    batch_messages.append([{"role": "user", "content": prompt}])
                
                # Prepare the completion arguments
                completion_args = {
                    "model": self.config.model,
                    "messages": batch_messages,
                    "temperature": field_config.temperature,
                    "timeout": self.config.request_timeout,
                    **self.config.model_kwargs
                }
                
                if field_config.max_tokens:
                    completion_args["max_tokens"] = field_config.max_tokens
                
                # Add thinking mode and reasoning effort for supported models
                if field_config.thinking_enabled and field_config.reasoning_effort and field_config.reasoning_effort != "disable":
                    if self.config.model.startswith("gemini/"):
                        completion_args["reasoning_effort"] = field_config.reasoning_effort
                
                # Call LiteLLM batch completion
                retry_responses = batch_completion(**completion_args)
                
                # Process retry responses and identify remaining nulls
                new_null_indices = []
                
                for i, (retry_idx, response) in enumerate(zip(current_null_indices, retry_responses)):
                    if response and hasattr(response, 'choices') and len(response.choices) > 0:
                        response_text = response.choices[0].message.content
                        
                        # Extract reasoning content if available and thinking is enabled
                        reasoning_content = None
                        if field_config.thinking_enabled and hasattr(response.choices[0].message, 'reasoning_content'):
                            reasoning_content = response.choices[0].message.reasoning_content
                        
                        if response_text is None or (isinstance(response_text, str) and len(response_text.strip()) == 0):
                            # Still null after retry
                            new_null_indices.append(retry_idx)
                        else:
                            # Successfully got a response
                            response_texts[retry_idx] = response_text
                            reasoning_contents[retry_idx] = reasoning_content
                    else:
                        # Response is still null
                        new_null_indices.append(retry_idx)
                
                # Log progress
                successful_count = len(current_null_indices) - len(new_null_indices)
                logger.info(
                    f"Batch retry {retry_attempt}: {successful_count}/{len(current_null_indices)} responses recovered, "
                    f"{len(new_null_indices)} still null"
                )
                
                # Update for next iteration
                current_null_indices = new_null_indices
                
                # Small delay before next retry
                if current_null_indices and retry_attempt < max_batch_retries:
                    time.sleep(self.config.retry_delay)
                    
            except Exception as e:
                logger.error(f"Batch retry attempt {retry_attempt} failed: {e}")
                # On batch failure, try individual calls as last resort
                if retry_attempt == max_batch_retries:
                    logger.warning("Max batch retries reached, falling back to individual retries for remaining nulls")
                    for null_idx in current_null_indices:
                        try:
                            retry_response, retry_reasoning = self._call_llm(prompts[null_idx], field_config)
                            response_texts[null_idx] = retry_response
                            reasoning_contents[null_idx] = retry_reasoning
                        except Exception as e2:
                            logger.error(f"Individual retry failed for prompt {null_idx}: {e2}")
                            response_texts[null_idx] = None
                            reasoning_contents[null_idx] = None
                    break
                else:
                    # Continue to next batch retry attempt
                    time.sleep(self.config.retry_delay * retry_attempt)
        
        # Final report
        if current_null_indices:
            logger.warning(
                f"After {retry_attempt} batch retry attempts, {len(current_null_indices)} responses remain null"
            )
        
        return response_texts, reasoning_contents
    
    def _extract_output(self, raw_text: Optional[str], field_config: FieldConfig) -> Optional[str]:
        """
        Extract the desired output from raw LLM text.
        
        Simple extraction logic:
        1. Look for output_label (e.g., "rewrite_output:") and take everything after it
        2. Check if content is wrapped in quotes and optionally remove them
        3. Fallback to entire text if no label found
        """
        if not raw_text:
            return None

        text = raw_text.strip()
        if not text:
            return None

        # Strategy 1: Look for specific output label and take everything after it
        if field_config.output_label:
            label_pattern = re.escape(field_config.output_label)
            
            # Look for "label:" and take everything after it (to end of text)
            pattern = rf"{label_pattern}\s*:\s*(.*)$"
            match = re.search(pattern, text, flags=re.DOTALL)
            
            if match:
                content = match.group(1).strip()
                if content:
                    extracted = self._clean_extracted_text(content, field_config)
                    return extracted

        # Strategy 2: Look for any "*_output:" pattern and take everything after it
        generic_pattern = r"(\w*_?output)\s*:\s*(.*)$"
        match = re.search(generic_pattern, text, flags=re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(2).strip()
            if content:
                return self._clean_extracted_text(content, field_config)

        # Strategy 3: No label found, return entire text
        logger.debug("No output label found, using entire text")
        result = self._clean_extracted_text(text, field_config)
        return result

    def _clean_extracted_text(self, text: str, field_config: FieldConfig) -> str:
        """Clean and process extracted text."""
        if not text:
            return text
            
        text = text.strip()
        
        # Only strip outer quotes if configured and if the entire content is wrapped
        # if field_config.strip_quotes:
        # Check if the entire content is wrapped in quotes
        if ((text.startswith('"') and text.endswith('"')) or 
            (text.startswith("'") and text.endswith("'"))):
            # Only remove if it's a complete wrap (not partial quotes within content)
            inner_text = text[1:-1].strip()
            if inner_text:  # Make sure we don't end up with empty content
                text = inner_text
    
        return text.strip()

    def _paraphrase_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Paraphrase a single example according to field configurations.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            The example with added paraphrased fields
        """
        result = example.copy()
        
        for field_config in self.config.field_configs:
            # Get the input text
            if field_config.input_field not in example:
                logger.warning(f"Field {field_config.input_field} not found in example, skipping")
                continue
            
            input_text = example[field_config.input_field]
            
            # Handle None or non-string input
            if input_text is None:
                logger.warning(f"Field {field_config.input_field} is None, skipping")
                continue
            
            if not isinstance(input_text, str):
                logger.warning(f"Field {field_config.input_field} is not a string, converting to string")
                input_text = str(input_text)
            
            # Check for manual paraphrase override
            manual_key = None
            if self.config.manual_paraphrases and 'book_id' in example and 'snippet_id' in example:
                manual_key = (example['book_id'], example['snippet_id'])
                if manual_key in self.config.manual_paraphrases:
                    manual_entry = self.config.manual_paraphrases[manual_key]
                    # Check if this field has a manual override
                    if field_config.output_field in manual_entry:
                        logger.info(f"Using manual paraphrase for book_id={example['book_id']}, snippet_id={example['snippet_id']}, field={field_config.output_field}")
                        # Use the manual paraphrase directly
                        result[field_config.output_field] = manual_entry[field_config.output_field]
                        result[f"{field_config.output_field}_raw"] = f"MANUAL_OVERRIDE: {manual_entry[field_config.output_field]}"
                        continue
            
            # Format the prompt using string replacement to avoid format string issues
            # No need to escape braces since we're not using .format()
            prompt = field_config.prompt.replace('{input}', input_text)
            
            try:
                # Call the LLM
                raw_text, reasoning_content = self._call_llm(prompt, field_config)
                
                # Store raw response in separate field
                raw_field_name = f"{field_config.output_field}_raw"
                result[raw_field_name] = raw_text
                
                # Store reasoning content if available
                if reasoning_content is not None:
                    reasoning_field_name = f"{field_config.output_field}_reasoning"
                    result[reasoning_field_name] = reasoning_content
                
                # Extract desired content
                extracted = self._extract_output(raw_text, field_config)
                result[field_config.output_field] = extracted
                
                # Log extraction info for debugging
                if extracted is None or (isinstance(extracted, str) and len(extracted.strip()) == 0):
                    logger.warning(f"Extraction resulted in empty content for field {field_config.input_field}")
                    logger.debug(f"Raw text was: {raw_text[:200]}..." if raw_text else "None")
                    
            except Exception as e:
                logger.error(f"Failed to paraphrase field {field_config.input_field}: {e}")
                result[field_config.output_field] = None
                result[f"{field_config.output_field}_raw"] = None
                if field_config.thinking_enabled:
                    result[f"{field_config.output_field}_reasoning"] = None
        
        return result
    
    def _paraphrase_batch(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Paraphrase a batch of examples using batch completion.
        
        Args:
            examples: List of examples from the dataset
            
        Returns:
            List of examples with added paraphrased fields
        """
        if not examples:
            return []
        
        results = [example.copy() for example in examples]
        
        # Process each field configuration
        for field_config in self.config.field_configs:
            # Collect prompts for this field across all examples in the batch
            prompts = []
            valid_indices = []  # Track which examples have the required field
            
            for i, example in enumerate(examples):
                if field_config.input_field not in example:
                    logger.warning(f"Field {field_config.input_field} not found in example {i}, skipping")
                    continue
                
                input_text = example[field_config.input_field]
                
                # Handle None or non-string input
                if input_text is None:
                    logger.warning(f"Field {field_config.input_field} is None in example {i}, skipping")
                    continue
                
                if not isinstance(input_text, str):
                    logger.warning(f"Field {field_config.input_field} is not a string in example {i}, converting to string")
                    input_text = str(input_text)
                
                # Format the prompt using string replacement to avoid format string issues
                # No need to escape braces since we're not using .format()
                prompt = field_config.prompt.replace('{input}', input_text)
                prompts.append(prompt)
                valid_indices.append(i)
            
            if not prompts:
                logger.warning(f"No valid prompts found for field {field_config.input_field} in batch")
                continue
            
            try:
                # Call batch LLM API
                raw_responses, reasoning_responses = self._call_llm_batch(prompts, field_config)
                
                # Process responses and update results
                for prompt_idx, example_idx in enumerate(valid_indices):
                    raw_text = raw_responses[prompt_idx] if prompt_idx < len(raw_responses) else None
                    reasoning_content = reasoning_responses[prompt_idx] if prompt_idx < len(reasoning_responses) else None
                    
                    # Store raw response
                    raw_field_name = f"{field_config.output_field}_raw"
                    results[example_idx][raw_field_name] = raw_text
                    
                    # Store reasoning content if available
                    if reasoning_content is not None:
                        reasoning_field_name = f"{field_config.output_field}_reasoning"
                        results[example_idx][reasoning_field_name] = reasoning_content
                    
                    # Extract desired content
                    extracted = self._extract_output(raw_text, field_config)
                    results[example_idx][field_config.output_field] = extracted
                    
                    # Log extraction info for debugging
                    if extracted is None or (isinstance(extracted, str) and len(extracted.strip()) == 0):
                        logger.warning(f"Batch extraction resulted in empty content for field {field_config.input_field}, example {example_idx}")
                        logger.debug(f"Raw text was: {raw_text[:200]}..." if raw_text else "None")
                        
            except Exception as e:
                logger.error(f"Failed to paraphrase field {field_config.input_field} in batch: {e}")
                # Set all fields to None for this batch
                for example_idx in valid_indices:
                    results[example_idx][field_config.output_field] = None
                    results[example_idx][f"{field_config.output_field}_raw"] = None
                    if field_config.thinking_enabled:
                        results[example_idx][f"{field_config.output_field}_reasoning"] = None
        
        return results
    
    def paraphrase(self) -> Dataset:
        """
        Paraphrase the entire dataset.
        
        Returns:
            The paraphrased dataset
        """
        if self.dataset is None:
            self.load_dataset()
        
        logger.info("Starting paraphrasing process...")
        
        paraphrased_examples = []
        
        if self.config.enable_batching and self.config.batch_size > 1:
            # Batched processing
            logger.info(f"Using batched processing with batch size {self.config.batch_size}")
            
            # Process in batches
            for i in tqdm(range(0, len(self.dataset), self.config.batch_size), desc="Processing batches"):
                batch_end = min(i + self.config.batch_size, len(self.dataset))
                batch_examples = [self.dataset[j] for j in range(i, batch_end)]
                
                batch_results = self._paraphrase_batch(batch_examples)
                paraphrased_examples.extend(batch_results)
        else:
            # Sequential processing (original behavior)
            logger.info("Using sequential processing")
            for example in tqdm(self.dataset, desc="Paraphrasing"):
                paraphrased_example = self._paraphrase_example(example)
                paraphrased_examples.append(paraphrased_example)
        
        # Create new dataset
        self.processed_dataset = Dataset.from_list(paraphrased_examples)
        logger.info(f"Paraphrasing complete. Processed {len(self.processed_dataset)} examples")
        
        return self.processed_dataset
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the paraphrased dataset locally.
        
        Args:
            path: Path to save the dataset. If None, uses config.output_path
        """
        if self.processed_dataset is None:
            raise ValueError("No processed dataset to save. Run paraphrase() first.")
        
        save_path = path or self.config.output_path
        if save_path is None:
            logger.warning("No save path specified, skipping local save")
            return
        
        logger.info(f"Saving dataset to {save_path}")
        self.processed_dataset.save_to_disk(save_path)
        logger.info("Dataset saved successfully")
    
    def push_to_hub(self, repo_id: Optional[str] = None, private: Optional[bool] = None) -> None:
        """
        Push the paraphrased dataset to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID on HuggingFace Hub. If None, uses config.hub_repo_id
            private: Whether the repository should be private. If None, uses config.hub_private
        """
        if self.processed_dataset is None:
            raise ValueError("No processed dataset to push. Run paraphrase() first.")
        
        repo_id = repo_id or self.config.hub_repo_id
        if repo_id is None:
            raise ValueError("No repository ID specified")
        
        private = private if private is not None else self.config.hub_private
        
        logger.info(f"Pushing dataset to HuggingFace Hub: {repo_id}")
        self.processed_dataset.push_to_hub(repo_id, private=private)
        logger.info("Dataset pushed successfully")
    
    def run(self) -> Dataset:
        """
        Run the complete paraphrasing pipeline.
        
        This method:
        1. Loads the dataset
        2. Paraphrases all examples
        3. Saves locally if configured
        4. Pushes to Hub if configured
        
        Returns:
            The paraphrased dataset
        """
        # Load and paraphrase
        self.load_dataset()
        self.paraphrase()
        
        # Save locally if configured
        if self.config.output_path:
            self.save()
        
        # Push to Hub if configured
        if self.config.push_to_hub:
            self.push_to_hub()
        
        return self.processed_dataset
