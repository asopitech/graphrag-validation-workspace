# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Factory functions for creating enhanced Bedrock models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .bedrock_models_enhanced import (
    EnhancedBedrockAnthropicChatLLM,
    EnhancedBedrockEmbeddingLLM,
)
from .config import (
    ANTHROPIC_CLAUDE_3_5_SONNET,
    ANTHROPIC_CLAUDE_3_HAIKU,
    TITAN_EMBED_TEXT_V2,
    BedrockAnthropicConfig,
    BedrockEmbeddingConfig,
)
from .events import LLMEvents

if TYPE_CHECKING:
    pass


def create_bedrock_anthropic_llm(
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    *,
    config: BedrockAnthropicConfig | None = None,
    events: LLMEvents | None = None,
    **kwargs,
) -> EnhancedBedrockAnthropicChatLLM:
    """Create an enhanced Bedrock Anthropic Chat LLM.
    
    Args:
        model: Model identifier
        config: Optional configuration override
        events: Optional event handler
        **kwargs: Additional configuration parameters
        
    Returns:
        Enhanced Bedrock Anthropic Chat LLM instance
    """
    if config is None:
        # Use predefined config or create new one
        if model == "anthropic.claude-3-5-sonnet-20241022-v2:0":
            config = ANTHROPIC_CLAUDE_3_5_SONNET
        elif model == "anthropic.claude-3-haiku-20240307-v1:0":
            config = ANTHROPIC_CLAUDE_3_HAIKU
        else:
            config = BedrockAnthropicConfig(model=model)
    
    # Override config with kwargs
    if kwargs:
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        config = BedrockAnthropicConfig(**config_dict)
    
    return EnhancedBedrockAnthropicChatLLM(
        config=config,
        events=events,
    )


def create_bedrock_embedding_llm(
    model: str = "amazon.titan-embed-text-v2:0",
    *,
    config: BedrockEmbeddingConfig | None = None,
    events: LLMEvents | None = None,
    **kwargs,
) -> EnhancedBedrockEmbeddingLLM:
    """Create an enhanced Bedrock Embedding LLM.
    
    Args:
        model: Model identifier
        config: Optional configuration override
        events: Optional event handler
        **kwargs: Additional configuration parameters
        
    Returns:
        Enhanced Bedrock Embedding LLM instance
    """
    if config is None:
        # Use predefined config or create new one
        if model == "amazon.titan-embed-text-v2:0":
            config = TITAN_EMBED_TEXT_V2
        else:
            config = BedrockEmbeddingConfig(model=model)
    
    # Override config with kwargs
    if kwargs:
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        config = BedrockEmbeddingConfig(**config_dict)
    
    return EnhancedBedrockEmbeddingLLM(
        config=config,
        events=events,
    )


# Convenience factory functions with specific configurations
def create_claude_3_5_sonnet(
    *,
    rpm_limit: int | None = 1000,
    tpm_limit: int | None = 100000,
    max_retries: int = 10,
    events: LLMEvents | None = None,
    **kwargs,
) -> EnhancedBedrockAnthropicChatLLM:
    """Create Claude 3.5 Sonnet with production settings."""
    return create_bedrock_anthropic_llm(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        max_retries=max_retries,
        events=events,
        **kwargs,
    )


def create_claude_3_haiku(
    *,
    rpm_limit: int | None = 2000,
    tpm_limit: int | None = 200000,
    max_retries: int = 10,
    events: LLMEvents | None = None,
    **kwargs,
) -> EnhancedBedrockAnthropicChatLLM:
    """Create Claude 3 Haiku with production settings."""
    return create_bedrock_anthropic_llm(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        max_retries=max_retries,
        events=events,
        **kwargs,
    )


def create_titan_embed_v2(
    *,
    dimensions: int = 1024,
    rpm_limit: int | None = 2000,
    tpm_limit: int | None = 500000,
    events: LLMEvents | None = None,
    **kwargs,
) -> EnhancedBedrockEmbeddingLLM:
    """Create Titan Embed Text V2 with production settings."""
    return create_bedrock_embedding_llm(
        model="amazon.titan-embed-text-v2:0",
        dimensions=dimensions,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        events=events,
        **kwargs,
    )