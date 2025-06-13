# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Rate limiting service (ported from fnllm)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import Unpack

from ..limiting import Limiter
from ..types import LLMInput, LLMOutput, Manifest, TInput, TJsonModel, TModelParameters
from .decorator import LLMDecorator, THistoryEntry, TOutput

if TYPE_CHECKING:
    from ..events import LLMEvents

TokenEstimator = Callable[
    [TInput, LLMInput], int
]


class RateLimiter(
    LLMDecorator[TOutput, THistoryEntry],
    Generic[TInput, TOutput, THistoryEntry, TModelParameters],
):
    """A rate limiter decorator for LLM operations."""

    def __init__(
        self,
        limiter: Limiter,
        *,
        events: LLMEvents,
        estimator: TokenEstimator[TInput] | None = None,
    ):
        """Create a new RateLimiter."""
        self._limiter = limiter
        self._events = events
        self._estimator = estimator or self._default_estimator

    def _default_estimator(self, prompt: TInput, args: LLMInput) -> int:
        """Default token estimator (rough approximation)."""
        if isinstance(prompt, str):
            # Rough approximation: 1 token ≈ 4 characters
            return len(prompt) // 4
        return 100  # Default fallback

    async def _handle_post_request_limiting(
        self,
        result: LLMOutput[TOutput, TJsonModel, THistoryEntry],
    ) -> None:
        """Handle post-request rate limiting adjustments."""
        reconciliation = await self._limiter.reconcile(result)
        if reconciliation is not None:
            await self._events.on_limit_reconcile(reconciliation)
        else:
            # If we didn't get explicit reconciliation, try to adjust based on actual usage
            diff = result.metrics.usage.tokens_diff
            if diff > 0:
                manifest = Manifest(post_request_tokens=diff)
                # Consume the token difference
                async with self._limiter.use(manifest):
                    await self._events.on_post_limit(manifest)

    def decorate(
        self,
        delegate: Callable[
            ..., Awaitable[LLMOutput[TOutput, TJsonModel, THistoryEntry]]
        ],
    ) -> Callable[..., Awaitable[LLMOutput[TOutput, TJsonModel, THistoryEntry]]]:
        """Execute the LLM with configured rate limits."""

        async def invoke(prompt: TInput, **args: Unpack[LLMInput]):
            estimated_input_tokens = self._estimator(prompt, args)

            manifest = Manifest(request_tokens=estimated_input_tokens)
            try:
                async with self._limiter.use(manifest):
                    await self._events.on_limit_acquired(manifest)
                    result = await delegate(prompt, **args)
            finally:
                await self._events.on_limit_released(manifest)

            result.metrics.usage.estimated_input_tokens = estimated_input_tokens
            await self._handle_post_request_limiting(result)

            return result

        return invoke