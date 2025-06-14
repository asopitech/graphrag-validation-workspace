# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Usage extraction for AWS Bedrock responses."""

from __future__ import annotations

from typing import Any

from ..types import LLMUsageMetrics


class BedrockUsageExtractor:
    """Extract usage metrics from AWS Bedrock responses."""

    def extract_usage(self, response: Any) -> LLMUsageMetrics:
        """Extract usage metrics from Bedrock response."""
        usage = LLMUsageMetrics()
        
        if isinstance(response, dict):
            # Handle different response formats
            if "usage" in response:
                # Standard Bedrock usage format
                bedrock_usage = response["usage"]
                usage.input_tokens = bedrock_usage.get("input_tokens", 0)
                usage.output_tokens = bedrock_usage.get("output_tokens", 0)
                usage.total_tokens = usage.input_tokens + usage.output_tokens
                
            elif "inputTextTokenCount" in response:
                # Titan embedding format
                usage.input_tokens = response.get("inputTextTokenCount", 0)
                usage.output_tokens = 0  # Embeddings don't have output tokens
                usage.total_tokens = usage.input_tokens
                
            elif hasattr(response, 'usage'):
                # Object with usage attribute
                bedrock_usage = response.usage
                if hasattr(bedrock_usage, 'input_tokens'):
                    usage.input_tokens = bedrock_usage.input_tokens
                if hasattr(bedrock_usage, 'output_tokens'):
                    usage.output_tokens = bedrock_usage.output_tokens
                usage.total_tokens = usage.input_tokens + usage.output_tokens
                
        return usage


class BedrockTokenEstimator:
    """Estimate token usage for AWS Bedrock requests."""
    
    def __init__(self, model_type: str = "anthropic"):
        """Create a new BedrockTokenEstimator."""
        self._model_type = model_type
    
    def estimate_tokens(self, prompt: str, kwargs: dict[str, Any] | None = None) -> int:
        """Estimate token count for a prompt."""
        if not isinstance(prompt, str):
            return 100  # Default fallback
            
        # Different estimation strategies by model type
        if self._model_type == "anthropic":
            return self._estimate_anthropic_tokens(prompt, kwargs)
        elif self._model_type == "titan_embed":
            return self._estimate_titan_embed_tokens(prompt)
        else:
            return self._estimate_generic_tokens(prompt)
    
    def _estimate_anthropic_tokens(self, prompt: str, kwargs: dict[str, Any] | None = None) -> int:
        """Estimate tokens for Anthropic Claude models."""
        # Anthropic rough estimation: ~4 characters per token
        base_tokens = len(prompt) // 4
        
        # Add tokens for system prompt if present
        if kwargs:
            system_prompt = kwargs.get("system")
            if system_prompt:
                base_tokens += len(str(system_prompt)) // 4
                
            # Add tokens for history
            history = kwargs.get("history") or []
            for entry in history:
                if isinstance(entry, dict) and "content" in entry:
                    content = entry["content"]
                    if isinstance(content, str):
                        base_tokens += len(content) // 4
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                base_tokens += len(item["text"]) // 4
        
        # Add overhead for message formatting
        return max(base_tokens + 50, 10)  # Minimum 10 tokens
    
    def _estimate_titan_embed_tokens(self, text: str) -> int:
        """Estimate tokens for Titan embedding models."""
        # Titan embedding rough estimation: ~4 characters per token
        return max(len(text) // 4, 1)  # Minimum 1 token
    
    def _estimate_generic_tokens(self, text: str) -> int:
        """Generic token estimation."""
        # Conservative estimation: ~3 characters per token
        return max(len(text) // 3, 10)  # Minimum 10 tokens