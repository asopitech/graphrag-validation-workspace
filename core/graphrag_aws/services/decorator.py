# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Decorator interface for LLM services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from ..types import LLMOutput, THistoryEntry, TJsonModel

TOutput = TypeVar("TOutput")


class LLMDecorator(ABC, Generic[TOutput, THistoryEntry]):
    """Base class for LLM decorators."""

    @abstractmethod
    def decorate(
        self,
        delegate: Callable[
            ..., Awaitable[LLMOutput[TOutput, TJsonModel, THistoryEntry]]
        ],
    ) -> Callable[..., Awaitable[LLMOutput[TOutput, TJsonModel, THistoryEntry]]]:
        """Decorate the delegate function."""