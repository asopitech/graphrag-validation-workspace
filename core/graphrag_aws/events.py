# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Event system for LLM operations (inspired by fnllm)."""

from __future__ import annotations

import logging
from typing import Any

from .types import LLMMetrics, LimitReconciliation, LimitUpdate, Manifest

logger = logging.getLogger(__name__)


class LLMEvents:
    """Event handler for LLM operations."""

    async def on_execute_llm(self) -> None:
        """Called when LLM execution starts."""
        logger.debug("LLM execution started")

    async def on_error(
        self, error: BaseException, stack_trace: str, context: dict[str, Any]
    ) -> None:
        """Called when an error occurs."""
        logger.error(f"LLM error: {error}", extra={"stack_trace": stack_trace, "context": context})

    async def on_usage(self, usage: Any) -> None:
        """Called when usage metrics are available."""
        logger.debug(f"Usage metrics: {usage}")

    async def on_success(self, metrics: LLMMetrics) -> None:
        """Called on successful completion."""
        logger.debug(f"LLM call completed successfully: {metrics}")

    async def on_try(self, attempt_number: int) -> None:
        """Called on each retry attempt."""
        logger.debug(f"LLM retry attempt #{attempt_number}")

    async def on_retryable_error(self, error: BaseException, attempt_number: int) -> None:
        """Called when a retryable error occurs."""
        logger.warning(f"Retryable error on attempt #{attempt_number}: {error}")

    async def on_non_retryable_error(self, error: BaseException, attempt_number: int) -> None:
        """Called when a non-retryable error occurs."""
        logger.error(f"Non-retryable error on attempt #{attempt_number}: {error}")

    async def on_recover_from_error(self, attempt_number: int) -> None:
        """Called when recovering from an error."""
        logger.info(f"Recovered from error after {attempt_number} attempts")

    # Rate limiting events
    async def on_limit_acquired(self, manifest: Manifest) -> None:
        """Called when rate limit is acquired."""
        logger.debug(f"Rate limit acquired: {manifest}")

    async def on_limit_released(self, manifest: Manifest) -> None:
        """Called when rate limit is released."""
        logger.debug(f"Rate limit released: {manifest}")

    async def on_limit_reconcile(self, reconciliation: LimitReconciliation) -> None:
        """Called when rate limit is reconciled."""
        logger.debug(f"Rate limit reconciled: {reconciliation}")

    async def on_post_limit(self, manifest: Manifest) -> None:
        """Called for post-request limiting."""
        logger.debug(f"Post-request limit applied: {manifest}")