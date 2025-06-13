# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""AWS Bedrock error handling and retry logic."""

from __future__ import annotations

import asyncio
from typing import Final

from botocore.exceptions import ClientError

from .errors import InvalidLLMResultError

# AWS Bedrock specific retryable errors
BEDROCK_RETRYABLE_ERRORS: Final[list[type[Exception]]] = [
    ClientError,  # AWS service errors (includes throttling)
    ConnectionError,  # Network connectivity issues
    TimeoutError,  # Request timeouts
    InvalidLLMResultError,  # Invalid responses
]

# Specific AWS error codes that are retryable
RETRYABLE_ERROR_CODES: Final[set[str]] = {
    "ThrottlingException",
    "ServiceUnavailableException", 
    "InternalServerException",
    "TemporaryFailure",
    "RequestTimeoutException",
    "TooManyRequestsException",
    "ModelTimeoutException",
    "ModelErrorException",  # Sometimes retryable
}

# Error codes that should never be retried
NON_RETRYABLE_ERROR_CODES: Final[set[str]] = {
    "ValidationException",
    "AccessDeniedException", 
    "ResourceNotFoundException",
    "ModelNotReadyException",
    "InferenceUnavailableException",
    "UnauthorizedException",
    "ForbiddenException",
}


def is_retryable_bedrock_error(error: BaseException) -> bool:
    """Check if a Bedrock error is retryable."""
    if isinstance(error, ClientError):
        error_code = error.response.get("Error", {}).get("Code", "")
        
        # Check specific error codes
        if error_code in NON_RETRYABLE_ERROR_CODES:
            return False
        if error_code in RETRYABLE_ERROR_CODES:
            return True
            
        # Check HTTP status codes
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        # 5xx errors are generally retryable, 4xx are not (except 429)
        if status_code == 429:  # Too Many Requests
            return True
        if 500 <= status_code < 600:
            return True
        if 400 <= status_code < 500:
            return False
            
    # Other error types
    return isinstance(error, (ConnectionError, TimeoutError, InvalidLLMResultError))


class BedrockRetryableErrorHandler:
    """Handler for AWS Bedrock retryable errors."""

    def __init__(self, behavior: str = "sleep") -> None:
        """Create a new BedrockRetryableErrorHandler."""
        self._behavior = behavior  # "sleep", "limit", "none"

    async def __call__(self, error: BaseException) -> None:
        """Handle the retryable error."""
        if isinstance(error, ClientError):
            await self._handle_client_error(error)
        elif isinstance(error, (ConnectionError, TimeoutError)):
            await self._handle_network_error(error)

    async def _handle_client_error(self, error: ClientError) -> None:
        """Handle AWS ClientError."""
        error_code = error.response.get("Error", {}).get("Code", "")
        
        # Extract retry-after header if present
        retry_after = None
        response_metadata = error.response.get("ResponseMetadata", {})
        http_headers = response_metadata.get("HTTPHeaders", {})
        
        if "retry-after" in http_headers:
            try:
                retry_after = float(http_headers["retry-after"])
            except (ValueError, TypeError):
                pass
                
        # Handle throttling with exponential backoff
        if error_code in {"ThrottlingException", "TooManyRequestsException"}:
            await self._handle_throttling(retry_after)
        elif error_code in {"ServiceUnavailableException", "InternalServerException"}:
            await self._handle_service_error(retry_after)

    async def _handle_throttling(self, retry_after: float | None) -> None:
        """Handle throttling errors."""
        if self._behavior == "sleep":
            # Use provided retry-after or default backoff
            sleep_time = retry_after or 1.0
            await asyncio.sleep(sleep_time)
        # For "limit" behavior, the rate limiter will handle the backoff
        # For "none" behavior, do nothing

    async def _handle_service_error(self, retry_after: float | None) -> None:
        """Handle service unavailable errors."""
        if self._behavior == "sleep":
            # Use provided retry-after or shorter default for service errors
            sleep_time = retry_after or 0.5
            await asyncio.sleep(sleep_time)

    async def _handle_network_error(self, error: Exception) -> None:
        """Handle network-related errors."""
        if self._behavior == "sleep":
            # Short sleep for network errors
            await asyncio.sleep(0.1)


def extract_retry_after(error: ClientError) -> float | None:
    """Extract retry-after value from AWS error response."""
    try:
        response_metadata = error.response.get("ResponseMetadata", {})
        headers = response_metadata.get("HTTPHeaders", {})
        
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            return float(retry_after)
    except (KeyError, ValueError, TypeError):
        pass
    
    return None