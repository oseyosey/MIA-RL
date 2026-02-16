"""
Client utilities for vLLM server hosting LLM judge models.

This module provides a robust client for communicating with a vLLM server
hosting language models like Qwen3-32B for judge evaluation.

Features:
- Connection pooling for efficient HTTP reuse
- Retry logic with exponential backoff
- Batch request optimization
- OpenAI-compatible API support
- Health check endpoints
- Graceful error handling
"""

from __future__ import annotations

import os
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache
import warnings
import numpy as np

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    warnings.warn(
        "requests library not installed. Remote LLM judge client will not work. "
        "Install with: pip install requests",
        RuntimeWarning
    )

try:
    import aiohttp
    import asyncio
    _HAS_ASYNC = True
except ImportError:
    _HAS_ASYNC = False

logger = logging.getLogger(__name__)

max_workers_default = 128    # Optimized for H200 with 1000-token sequences
batch_size_per_worker_default = 4  # Balanced efficiency for longer sequences

class LLMJudgeClient:
    """Optimized client for vLLM server hosting LLM judge models.
    
    Features:
    - Reduced concurrency (64 workers vs 512) for better resource management
    - batch_size_per_worker optimization to process multiple prompts per worker
    - Connection pooling and retry logic for reliability
    - 30-50% expected throughput improvement over high-concurrency approach
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        connection_pool_size: int = 1028,
        enable_cache: bool = True,
        cache_size: int = 1024,
    ):
        """
        Initialize the LLM judge client.
        
        Args:
            server_url: URL of the vLLM server. Falls back to LLM_JUDGE_SERVER_URL env var.
            api_key: Optional API key for authentication. Falls back to LLM_JUDGE_SERVER_API_KEY.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_backoff_factor: Backoff factor for retries.
            connection_pool_size: Size of the HTTP connection pool.
            enable_cache: Whether to cache generation results.
            cache_size: Maximum number of cached generations.
        """
        if not _HAS_REQUESTS:
            raise ImportError("requests library required for remote LLM judge client")
        
        # Server configuration
        self.server_url = server_url or os.getenv("LLM_JUDGE_SERVER_URL")
        if not self.server_url:
            raise ValueError(
                "No LLM judge server URL provided. Set LLM_JUDGE_SERVER_URL or pass server_url parameter."
            )
        
        # Clean up URL
        self.server_url = self.server_url.rstrip("/")
        
        # Authentication
        self.api_key = api_key or os.getenv("LLM_JUDGE_SERVER_API_KEY")
        
        # Client configuration
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        
        # Setup session with connection pooling and retries
        self.session = requests.Session()
        
        # Disable proxy usage - connect directly to vLLM server
        self.session.proxies = {'http': None, 'https': None}
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        adapter = HTTPAdapter(
            pool_connections=connection_pool_size,
            pool_maxsize=connection_pool_size,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Cache configuration
        self.enable_cache = enable_cache
        if enable_cache:
            # Create LRU cache for generations
            self._generate_cached = lru_cache(maxsize=cache_size)(self._generate_uncached)
        
        # Server info cache
        self._server_info = None
        self._server_info_time = 0
        self._server_info_ttl = 300  # 5 minutes
        
        logger.info(f"Initialized LLM judge client for {self.server_url}")
    
    def health_check(self) -> bool:
        """
        Check if the LLM judge server is healthy.
        
        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                headers=self.headers,
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get server information including model details.
        
        Returns:
            Server info dict or None if request fails.
        """
        # Check cache
        if self._server_info and (time.time() - self._server_info_time) < self._server_info_ttl:
            return self._server_info
        
        try:
            response = self.session.get(
                f"{self.server_url}/v1/models",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            self._server_info = response.json()
            self._server_info_time = time.time()
            return self._server_info
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return None
    
    def generate_responses(
        self,
        prompts: List[str],
        model: str = "Qwen/Qwen3-32B",
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        batch_size: Optional[int] = None,
        batch_size_per_worker: int = batch_size_per_worker_default,
        **generation_kwargs
    ) -> Optional[List[str]]:
        """
        Generate responses for a list of prompts.
        
        Args:
            prompts: List of prompts to generate responses for.
            model: Model name to use for generation.
            temperature: Generation temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.
            enable_thinking: Whether to enable thinking mode (Qwen3 specific).
            batch_size: Optional batch size for processing large lists.
            **generation_kwargs: Additional generation parameters.
        
        Returns:
            List of generated responses or None if failed.
        """
        if not prompts:
            return []
        
        # If caching is enabled and we have a single prompt, try cache
        if self.enable_cache and len(prompts) == 1:
            cache_key = (prompts[0], model, temperature, top_p, max_tokens, enable_thinking)
            try:
                # Use the cached version
                response = self._generate_cached(*cache_key, **generation_kwargs)
                return [response] if response is not None else None
            except Exception:
                # If caching fails, continue with normal flow
                pass
        
        # For multiple prompts or cache miss, make direct request
        if batch_size and len(prompts) > batch_size:
            # Process in batches
            responses = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                batch_responses = self._generate_batch(
                    batch, model, temperature, top_p, max_tokens, enable_thinking, batch_size_per_worker, **generation_kwargs
                )
                if batch_responses is None:
                    return None
                responses.extend(batch_responses)
            return responses
        else:
            # Single request
            return self._generate_batch(
                prompts, model, temperature, top_p, max_tokens, enable_thinking, batch_size_per_worker, **generation_kwargs
            )
    
    def _generate_batch(
        self,
        prompts: List[str],
        model: str = "Qwen/Qwen3-32B",
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        batch_size_per_worker: int = batch_size_per_worker_default,
        **generation_kwargs
    ) -> Optional[List[str]]:
        """
        Internal method to generate responses for a batch of prompts.
        
        Uses optimized concurrent processing with batch_size_per_worker to reduce
        network overhead while maintaining high throughput.
        
        Args:
            batch_size_per_worker: Number of prompts to process per worker thread
        
        Returns:
            List of response strings or None if failed.
        """
        try:
            import concurrent.futures
            import threading
            
            # Chunk prompts for optimized concurrent processing
            prompt_chunks = []
            for i in range(0, len(prompts), batch_size_per_worker):
                chunk = prompts[i:i + batch_size_per_worker]
                chunk_start_idx = i
                prompt_chunks.append((chunk_start_idx, chunk))
            
            # Pre-allocate responses to maintain order
            responses = [None] * len(prompts)
            
            def send_batch_request(chunk_start_idx, prompt_chunk):
                """Send a batch of prompts in a single request and return results."""
                try:
                    chunk_responses = []
                    
                    # Process each prompt in the chunk with individual requests
                    # Note: vLLM OpenAI API doesn't support true multi-prompt batching
                    for local_idx, prompt in enumerate(prompt_chunk):
                        global_idx = chunk_start_idx + local_idx
                        
                        # Format as chat messages
                        messages = [{"role": "user", "content": prompt}]
                        
                        # Prepare request payload for OpenAI-compatible API
                        payload = {
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": max_tokens,
                            "stream": False,
                            **generation_kwargs
                        }
                        
                        # Add thinking mode for Qwen3 if supported
                        if "qwen" in model.lower():
                            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
                        
                        # Make request
                        response = self.session.post(
                            f"{self.server_url}/v1/chat/completions",
                            json=payload,
                            headers=self.headers,
                            timeout=self.timeout
                        )
                        
                        response.raise_for_status()
                        
                        # Parse response
                        result = response.json()
                        
                        # Extract generated text
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            chunk_responses.append((global_idx, content))
                        else:
                            logger.error(f"Unexpected response format for prompt {global_idx}: {result}")
                            chunk_responses.append((global_idx, ""))
                    
                    return chunk_responses
                        
                except Exception as e:
                    logger.error(f"Error processing chunk starting at {chunk_start_idx}: {e}")
                    # Return empty responses for this chunk
                    return [(chunk_start_idx + i, "") for i in range(len(prompt_chunk))]
            
            # Send chunk requests concurrently with reduced worker count
            num_chunks = len(prompt_chunks)
            max_workers = min(num_chunks, max_workers_default)
            logger.info(f"Processing {len(prompts)} prompts in {num_chunks} chunks using {max_workers} workers (batch_size_per_worker={batch_size_per_worker})")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunk requests
                future_to_chunk = {
                    executor.submit(send_batch_request, chunk_start_idx, chunk): chunk_start_idx
                    for chunk_start_idx, chunk in prompt_chunks
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_responses = future.result()
                        for prompt_idx, content in chunk_responses:
                            responses[prompt_idx] = content
                    except Exception as e:
                        chunk_start_idx = future_to_chunk[future]
                        logger.error(f"Chunk request starting at {chunk_start_idx} failed: {e}")
                        # Fill in empty responses for failed chunk
                        chunk_size = len(prompt_chunks[chunk_start_idx // batch_size_per_worker][1])
                        for i in range(chunk_size):
                            if responses[chunk_start_idx + i] is None:
                                responses[chunk_start_idx + i] = ""
            
            # Convert None values to empty strings (safety check)
            responses = [r if r is not None else "" for r in responses]
            return responses
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to LLM judge server at {self.server_url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in generate_batch: {e}")
            return None
    
    def _generate_uncached(
        self,
        prompt: str,
        model: str = "Qwen/Qwen3-32B",
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        **generation_kwargs
    ) -> Optional[str]:
        """
        Generate response for a single prompt (uncached version).
        
        Returns:
            Generated response string or None if failed.
        """
        responses = self._generate_batch(
            [prompt], model, temperature, top_p, max_tokens, enable_thinking, batch_size_per_worker_default, **generation_kwargs
        )
        return responses[0] if responses else None
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncLLMJudgeClient:
    """Asynchronous client for vLLM server hosting LLM judge models."""
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        connector_limit: int = 100,
    ):
        """
        Initialize the async LLM judge client.
        
        Args:
            server_url: URL of the vLLM server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            connector_limit: Maximum number of connections.
        """
        if not _HAS_ASYNC:
            raise ImportError(
                "aiohttp required for async client. Install with: pip install aiohttp"
            )
        
        self.server_url = (server_url or os.getenv("LLM_JUDGE_SERVER_URL", "")).rstrip("/")
        if not self.server_url:
            raise ValueError("No LLM judge server URL provided")
        
        self.api_key = api_key or os.getenv("LLM_JUDGE_SERVER_API_KEY")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        
        # Headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Connector with connection limit
        self.connector = aiohttp.TCPConnector(limit=connector_limit)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            headers=self.headers,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def generate_responses(
        self,
        prompts: List[str],
        model: str = "Qwen/Qwen3-32B",
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        **generation_kwargs
    ) -> Optional[List[str]]:
        """
        Asynchronously generate responses for prompts.
        
        Returns:
            List of generated responses or None if failed.
        """

        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Create concurrent tasks for all prompts
        async def process_prompt(prompt):
            for attempt in range(self.max_retries):
                try:
                    # Format as chat messages
                    messages = [{"role": "user", "content": prompt}]
                    
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "stream": False,
                        **generation_kwargs
                    }
                    
                    # Add thinking mode for Qwen3 if supported
                    if "qwen" in model.lower():
                        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
                    
                    async with self.session.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=payload
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        
                        # Extract generated text
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            return content
                        else:
                            return ""
                        
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"All retry attempts failed for prompt: {e}")
                        return ""
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            return ""

        # Process all prompts concurrently
        tasks = [process_prompt(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        return responses


# Convenience functions
_default_client = None

def get_default_client() -> LLMJudgeClient:
    """Get or create the default LLM judge client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMJudgeClient()
    return _default_client


def generate_responses(
    prompts: List[str],
    model: str = "Qwen/Qwen3-32B",
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 512,
    enable_thinking: bool = False,
    server_url: Optional[str] = None,
    **generation_kwargs
) -> Optional[List[str]]:
    """
    Convenience function to generate responses using default or specified server.
    
    Args:
        prompts: List of prompts to generate responses for.
        model: Model name to use for generation.
        temperature: Generation temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens to generate.
        enable_thinking: Whether to enable thinking mode.
        server_url: Optional server URL (uses default if not specified).
        **generation_kwargs: Additional generation parameters.
    
    Returns:
        List of generated responses or None if failed.
    """
    if server_url:
        # Create temporary client for specific server
        with LLMJudgeClient(server_url=server_url) as client:
            return client.generate_responses(
                prompts, model, temperature, top_p, max_tokens, enable_thinking, **generation_kwargs
            )
    else:
        # Use default client
        client = get_default_client()
        return client.generate_responses(
            prompts, model, temperature, top_p, max_tokens, enable_thinking, **generation_kwargs
        )
