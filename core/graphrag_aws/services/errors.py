# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Error definitions for GraphRAG AWS Bedrock services."""


class RetriesExhaustedError(Exception):
    """Raised when retries are exhausted."""

    def __init__(self, name: str, max_retries: int):
        """Create a new RetriesExhaustedError."""
        self.name = name
        self.max_retries = max_retries
        super().__init__(f"Retries exhausted for {name} after {max_retries} attempts")


class InvalidLLMResultError(Exception):
    """Raised when LLM result is invalid."""

    def __init__(self, message: str):
        """Create a new InvalidLLMResultError."""
        super().__init__(message)