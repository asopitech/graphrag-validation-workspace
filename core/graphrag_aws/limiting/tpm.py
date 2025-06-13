# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""TPM (Tokens Per Minute) limiter module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiolimiter import AsyncLimiter

from .base import Limiter
from .rpm import LimitReconciler, update_limiter
from ..types import LimitUpdate

if TYPE_CHECKING:
    from ..types import LLMOutput, Manifest


class TPMLimiter(Limiter):
    """TPM (Tokens Per Minute) limiter class definition."""

    def __init__(
        self,
        limiter: AsyncLimiter,
        reconciler: LimitReconciler | None = None,
    ) -> None:
        """Create a new TPMLimiter."""
        self._limiter = limiter
        self._reconciler = reconciler

    async def acquire(self, manifest: Manifest) -> None:
        """Acquire tokens based on estimated usage."""
        tokens_needed = manifest.request_tokens
        if tokens_needed > 0:
            await self._limiter.acquire(tokens_needed)

    async def release(self, manifest: Manifest) -> None:
        """Do nothing for TPM limiter."""

    async def reconcile(self, output: LLMOutput[Any, Any, Any]) -> LimitUpdate | None:
        """Reconcile token usage based on actual consumption."""
        if self._reconciler is not None:
            reconciliation = self._reconciler(output)
            if (
                reconciliation.limit is not None
                and reconciliation.remaining is not None
            ):
                old = update_limiter(self._limiter, reconciliation)
                return LimitUpdate(
                    old_value=old, new_value=reconciliation.remaining or 0
                )
        return None

    @classmethod
    def from_tpm(
        cls,
        tokens_per_minute: int,
        reconciler: LimitReconciler | None = None,
    ) -> TPMLimiter:
        """Create a new TPMLimiter from tokens per minute."""
        return cls(
            AsyncLimiter(tokens_per_minute, time_period=60),
            reconciler=reconciler,
        )