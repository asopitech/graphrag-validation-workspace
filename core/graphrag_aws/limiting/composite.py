# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Composite limiter for combining multiple limiters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Limiter
from ..types import LimitUpdate

if TYPE_CHECKING:
    from ..types import LLMOutput, Manifest


class CompositeLimiter(Limiter):
    """Composite limiter that combines multiple limiters."""

    def __init__(self, limiters: list[Limiter]) -> None:
        """Create a new CompositeLimiter."""
        self._limiters = limiters

    async def acquire(self, manifest: Manifest) -> None:
        """Acquire from all limiters."""
        for limiter in self._limiters:
            await limiter.acquire(manifest)

    async def release(self, manifest: Manifest) -> None:
        """Release from all limiters."""
        for limiter in self._limiters:
            await limiter.release(manifest)

    async def reconcile(self, output: LLMOutput[Any, Any, Any]) -> LimitUpdate | None:
        """Reconcile all limiters."""
        updates = []
        for limiter in self._limiters:
            update = await limiter.reconcile(output)
            if update:
                updates.append(update)
        
        # Return the first significant update
        return updates[0] if updates else None