# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""RPM limiter module (ported from fnllm)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from aiolimiter import AsyncLimiter

from .base import Limiter
from ..types import LimitReconciliation, LimitUpdate

if TYPE_CHECKING:
    from ..types import LLMOutput, Manifest


LimitReconciler = Callable[[Any], LimitReconciliation]


def update_limiter(limiter: AsyncLimiter, reconciliation: LimitReconciliation) -> float:
    """Update limiter with reconciliation data."""
    if reconciliation.remaining is not None:
        old_value = limiter.rate
        # Update the limiter's rate based on remaining capacity
        new_rate = min(limiter.rate, reconciliation.remaining)
        limiter._rate = new_rate  # Direct assignment as AsyncLimiter doesn't have public setter
        return old_value
    return 0.0


class RPMLimiter(Limiter):
    """RPM limiter class definition."""

    def __init__(
        self,
        limiter: AsyncLimiter,
        reconciler: LimitReconciler | None = None,
        *,
        rps: bool,
    ) -> None:
        """Create a new RPMLimiter."""
        self._limiter = limiter
        self._reconciler = reconciler
        self._rps = rps

    async def acquire(self, manifest: Manifest) -> None:
        """Acquire a new request."""
        if manifest.request_tokens > 0:
            await self._limiter.acquire()

    async def release(self, manifest: Manifest) -> None:
        """Do nothing."""

    async def reconcile(self, output: LLMOutput[Any, Any, Any]) -> LimitUpdate | None:
        """Reconcile rate limit based on actual usage."""
        if self._reconciler is not None:
            reconciliation = self._reconciler(output)
            if (
                reconciliation.limit is not None
                and reconciliation.remaining is not None
            ):
                if self._rps:
                    # If the limiter is in RPS mode, convert to per-second rate
                    reconciliation.remaining = _rpm_to_rps(reconciliation.remaining)
                    reconciliation.limit = _rpm_to_rps(reconciliation.limit)
                old = update_limiter(self._limiter, reconciliation)
                return LimitUpdate(
                    old_value=old, new_value=reconciliation.remaining or 0
                )
        return None

    @classmethod
    def from_rpm(
        cls,
        requests_per_minute: int,
        burst_mode: bool = True,
        reconciler: LimitReconciler | None = None,
    ) -> RPMLimiter:
        """Create a new RPMLimiter from requests per minute."""
        if burst_mode:
            return cls(
                AsyncLimiter(requests_per_minute, time_period=60),
                reconciler=reconciler,
                rps=False,
            )

        rps = _rpm_to_rps(requests_per_minute)
        return cls(
            AsyncLimiter(rps, 1),
            reconciler=reconciler,
            rps=True,
        )


def _rpm_to_rps(rpm: float) -> float:
    """Convert requests per minute to requests per second."""
    return rpm / 60.0