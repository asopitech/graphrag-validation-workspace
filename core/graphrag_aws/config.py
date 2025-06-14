# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Configuration system for GraphRAG AWS Bedrock."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import RetryStrategy


@dataclass
class BedrockBaseConfig:
    """Base configuration for AWS Bedrock models."""
    
    # AWS Configuration
    region: str = "us-east-1"
    endpoint_url: str | None = None
    
    # Model Configuration
    model: str = ""
    
    # Rate Limiting Configuration
    rpm_limit: int | None = None  # Requests per minute
    tpm_limit: int | None = None  # Tokens per minute
    burst_mode: bool = True
    
    # Retry Configuration
    max_retries: int = 10
    max_retry_wait: float = 60.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Cache Configuration
    cache_enabled: bool = False
    cache_ttl: int = 3600  # Cache TTL in seconds
    
    # Event Configuration
    track_usage: bool = True
    log_level: str = "INFO"


@dataclass
class BedrockAnthropicConfig(BedrockBaseConfig):
    """Configuration for Anthropic Claude models via Bedrock."""
    
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Anthropic-specific parameters
    anthropic_version: str = "bedrock-2023-05-31"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str | None = None
    
    # Default rate limits for Anthropic models
    rpm_limit: int | None = 1000  # Conservative default
    tpm_limit: int | None = 100000  # Conservative default


@dataclass
class BedrockEmbeddingConfig(BedrockBaseConfig):
    """Configuration for embedding models via Bedrock."""
    
    model: str = "amazon.titan-embed-text-v2:0"
    
    # Embedding-specific parameters
    dimensions: int | None = None  # For V2 models
    normalize: bool = True
    embedding_types: list[str] = field(default_factory=lambda: ["float"])
    
    # Default rate limits for embedding models
    rpm_limit: int | None = 2000
    tpm_limit: int | None = 500000  # Higher for embeddings


@dataclass
class BedrockRateLimitBehavior:
    """Rate limit behavior configuration."""
    
    # How to handle rate limit errors
    on_rate_limit: str = "sleep"  # "sleep", "limit", "none"
    
    # Backoff configuration
    backoff_factor: float = 2.0
    max_backoff: float = 300.0  # 5 minutes max
    jitter: bool = True


# Predefined configurations for common models
ANTHROPIC_CLAUDE_3_5_SONNET = BedrockAnthropicConfig(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    max_tokens=4096,
    rpm_limit=1000,
    tpm_limit=100000,
)

ANTHROPIC_CLAUDE_3_HAIKU = BedrockAnthropicConfig(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    max_tokens=4096,
    rpm_limit=2000,
    tpm_limit=200000,
)

TITAN_EMBED_TEXT_V2 = BedrockEmbeddingConfig(
    model="amazon.titan-embed-text-v2:0",
    dimensions=1024,
    normalize=True,
    rpm_limit=2000,
    tpm_limit=500000,
)

TITAN_EMBED_TEXT_V1 = BedrockEmbeddingConfig(
    model="amazon.titan-embed-text-v1",
    normalize=False,
    rpm_limit=2000,
    tpm_limit=300000,
)