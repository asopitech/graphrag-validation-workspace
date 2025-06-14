# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Type definitions for AWS Bedrock GraphRAG implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar
from enum import Enum

# Generic type variables
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput") 
THistoryEntry = TypeVar("THistoryEntry")
TModelParameters = TypeVar("TModelParameters")
TJsonModel = TypeVar("TJsonModel")


class RetryStrategy(str, Enum):
    """The retry strategy to use for the LLM service."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    """Use exponential backoff for retries (e.g. exponential factor + randomized max_wait)."""

    RANDOM_WAIT = "random_wait"
    """Use random wait times between [0, max_retry_wait] for retries."""

    INCREMENTAL_WAIT = "incremental_wait"
    """Use incremental wait times between [0, max_retry_wait] for retries."""


@dataclass
class LLMUsageMetrics:
    """Usage metrics for LLM calls."""
    
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_input_tokens: int = 0
    
    @property
    def tokens_diff(self) -> int:
        """Calculate difference between actual and estimated tokens."""
        return max(0, self.total_tokens - self.estimated_input_tokens)


@dataclass
class LLMRetryMetrics:
    """Retry metrics for LLM calls."""
    
    total_time: float = 0.0
    num_retries: int = 0
    call_times: list[float] = field(default_factory=list)


@dataclass
class LLMMetrics:
    """Combined metrics for LLM operations."""
    
    usage: LLMUsageMetrics = field(default_factory=LLMUsageMetrics)
    retry: LLMRetryMetrics = field(default_factory=LLMRetryMetrics)


@dataclass
class LLMInput:
    """Input parameters for LLM calls."""
    
    history: list[THistoryEntry] | None = None
    json: bool = False
    json_model: TJsonModel | None = None
    variables: dict[str, Any] | None = None
    name: str | None = None
    model_parameters: TModelParameters | None = None


@dataclass
class LLMOutput:
    """Output from LLM calls."""
    
    output: TOutput
    history: list[THistoryEntry] | None = None
    metrics: LLMMetrics = field(default_factory=LLMMetrics)
    cache_hit: bool = False
    
    @property
    def full_response(self) -> Any:
        """Get the full response object if available."""
        if hasattr(self.output, 'full_response'):
            return self.output.full_response
        return None


@dataclass
class Manifest:
    """Resource usage manifest for limiting."""
    
    request_tokens: int = 1
    post_request_tokens: int = 0


@dataclass 
class LimitReconciliation:
    """Reconciliation data for rate limits."""
    
    limit: int | None = None
    remaining: int | None = None
    reset_time: float | None = None


@dataclass
class LimitUpdate:
    """Update notification for limit changes."""
    
    old_value: float
    new_value: float