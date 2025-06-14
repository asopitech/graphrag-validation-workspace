# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Enhanced AWS Bedrock models with fnllm-inspired features."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .config import BedrockAnthropicConfig, BedrockEmbeddingConfig
from .events import LLMEvents
from .limiting import CompositeLimiter, RPMLimiter, TPMLimiter
from .services.bedrock_errors import (
    BEDROCK_RETRYABLE_ERRORS,
    BedrockRetryableErrorHandler,
    is_retryable_bedrock_error,
)
from .services.rate_limiter import RateLimiter
from .services.retryer import Retryer
from .services.usage_extractor import BedrockTokenEstimator, BedrockUsageExtractor
from .types import LLMInput, LLMOutput, LLMUsageMetrics, Manifest, RetryStrategy

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

logger = logging.getLogger(__name__)


class BedrockAPIError(Exception):
    """AWS Bedrock API specific error."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class EnhancedBedrockAnthropicChatLLM:
    """Enhanced AWS Bedrock Anthropic Chat LLM with fnllm-inspired features."""

    def __init__(
        self,
        config: BedrockAnthropicConfig,
        *,
        events: LLMEvents | None = None,
    ) -> None:
        """Initialize the enhanced Bedrock Anthropic Chat model."""
        self.config = config
        self.model_id = config.model
        self.endpoint_url = config.endpoint_url or os.environ.get("BEDROCK_ENDPOINT")
        self.region = config.region
        
        # Initialize events
        self._events = events or LLMEvents()
        
        # Initialize AWS client
        try:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
            )
        except NoCredentialsError as e:
            error_msg = "AWS credentials not found. Please configure AWS credentials."
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e

        # Initialize services
        self._usage_extractor = BedrockUsageExtractor()
        self._token_estimator = BedrockTokenEstimator("anthropic")
        self._error_handler = BedrockRetryableErrorHandler()
        
        # Setup rate limiting
        self._rate_limiter = self._create_rate_limiter() if self._should_rate_limit() else None
        
        # Setup retry mechanism
        self._retryer = self._create_retryer() if self._should_retry() else None

    def _should_rate_limit(self) -> bool:
        """Check if rate limiting should be enabled."""
        return self.config.rpm_limit is not None or self.config.tpm_limit is not None

    def _should_retry(self) -> bool:
        """Check if retry should be enabled."""
        return self.config.max_retries > 0

    def _create_rate_limiter(self) -> RateLimiter | None:
        """Create rate limiter based on configuration."""
        limiters = []
        
        if self.config.rpm_limit:
            rpm_limiter = RPMLimiter.from_rpm(
                self.config.rpm_limit,
                burst_mode=self.config.burst_mode
            )
            limiters.append(rpm_limiter)
            
        if self.config.tpm_limit:
            tpm_limiter = TPMLimiter.from_tpm(self.config.tpm_limit)
            limiters.append(tpm_limiter)
        
        if limiters:
            composite_limiter = CompositeLimiter(limiters)
            return RateLimiter(
                composite_limiter,
                events=self._events,
                estimator=self._token_estimator.estimate_tokens,
            )
        
        return None

    def _create_retryer(self) -> Retryer | None:
        """Create retryer based on configuration."""
        if self.config.max_retries <= 0:
            return None
            
        # Use Bedrock retryable errors directly (they are already exception classes)
        retryable_errors = BEDROCK_RETRYABLE_ERRORS
        
        return Retryer(
            retryable_errors=retryable_errors,
            tag=f"BedrockAnthropic-{self.model_id}",
            max_retries=self.config.max_retries,
            max_retry_wait=self.config.max_retry_wait,
            events=self._events,
            retry_strategy=self.config.retry_strategy,
            retryable_error_handler=self._error_handler,
        )

    async def _execute_llm(self, prompt: str, kwargs: LLMInput) -> dict[str, Any]:
        """Execute the raw LLM call."""
        try:
            # Build messages according to Anthropic API specification
            messages = []
            
            # Add history if provided
            if kwargs.history:
                for h in kwargs.history:
                    if isinstance(h, dict) and "role" in h and "content" in h:
                        content = h["content"]
                        if isinstance(content, str):
                            content = [{"type": "text", "text": content}]
                        messages.append({"role": h["role"], "content": content})
            
            # Add current prompt
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            })
            
            # Build request body
            body = {
                "anthropic_version": self.config.anthropic_version,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            
            # Add system prompt if configured
            if self.config.system_prompt:
                body["system"] = self.config.system_prompt
                
            # Make API call
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            
            # Check response status
            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise BedrockAPIError(
                    f"Bedrock API returned status {response['ResponseMetadata']['HTTPStatusCode']}"
                )
            
            # Parse response
            result = response["body"].read().decode()
            result_json = json.loads(result)
            
            return result_json
            
        except ClientError as e:
            # Re-raise ClientError directly for retry mechanism
            logger.error(f"AWS Bedrock ClientError: {e.response.get('Error', {}).get('Message', str(e))}")
            raise
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Bedrock API response: {str(e)}"
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e

    async def _decorated_call(self, prompt: str, **kwargs) -> LLMOutput[dict, Any, Any]:
        """Execute with all decorators applied."""
        await self._events.on_execute_llm()
        
        # Convert kwargs to LLMInput
        llm_input = LLMInput(
            history=kwargs.get("history"),
            json=kwargs.get("json", False),
            variables=kwargs.get("variables"),
            name=kwargs.get("name"),
        )
        
        output = await self._execute_llm(prompt, llm_input)
        
        # Create LLM output
        result = LLMOutput(output=output)
        
        # Extract usage metrics
        usage = self._usage_extractor.extract_usage(output)
        result.metrics.usage = usage
        
        return result

    async def achat(self, prompt: str, history: list | None = None, **kwargs) -> Any:
        """Generate chat response with all enhancements."""
        # Prepare kwargs
        call_kwargs = {
            "history": history,
            **kwargs
        }
        
        # Apply decorators in order: rate_limiter -> retryer -> core
        target_func = self._decorated_call
        
        # Apply retryer if configured
        if self._retryer:
            target_func = self._retryer.decorate(target_func)
            
        # Apply rate limiter if configured
        if self._rate_limiter:
            target_func = self._rate_limiter.decorate(target_func)
        
        # Execute with all decorators
        result = await target_func(prompt, **call_kwargs)
        
        # Extract content for backwards compatibility
        content = ""
        if isinstance(result.output, dict) and "content" in result.output:
            for content_block in result.output["content"]:
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")
        
        return content

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> Any:
        """Synchronous version of achat."""
        import asyncio
        return asyncio.run(self.achat(prompt, history, **kwargs))

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat response."""
        # TODO: Implement streaming with invoke_model_with_response_stream
        response = await self.achat(prompt, history, **kwargs)
        yield response

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """Synchronous streaming chat."""
        import asyncio
        
        async def _async_gen():
            async for chunk in self.achat_stream(prompt, history, **kwargs):
                yield chunk
                
        return asyncio.run(_async_gen())


class EnhancedBedrockEmbeddingLLM:
    """Enhanced AWS Bedrock Embedding LLM with fnllm-inspired features."""

    def __init__(
        self,
        config: BedrockEmbeddingConfig,
        *,
        events: LLMEvents | None = None,
    ) -> None:
        """Initialize the enhanced Bedrock Embedding model."""
        self.config = config
        self.model_id = config.model
        self.endpoint_url = config.endpoint_url or os.environ.get("BEDROCK_ENDPOINT")
        self.region = config.region
        
        # Initialize events and services
        self._events = events or LLMEvents()
        self._usage_extractor = BedrockUsageExtractor()
        self._token_estimator = BedrockTokenEstimator("titan_embed")
        
        # Initialize AWS client
        try:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
            )
        except NoCredentialsError as e:
            error_msg = "AWS credentials not found. Please configure AWS credentials."
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e

        # Setup rate limiting for embeddings
        self._rate_limiter = self._create_rate_limiter() if self._should_rate_limit() else None

    def _should_rate_limit(self) -> bool:
        """Check if rate limiting should be enabled."""
        return self.config.rpm_limit is not None or self.config.tpm_limit is not None

    def _create_rate_limiter(self) -> RateLimiter | None:
        """Create rate limiter for embedding model."""
        limiters = []
        
        if self.config.rpm_limit:
            rpm_limiter = RPMLimiter.from_rpm(self.config.rpm_limit)
            limiters.append(rpm_limiter)
            
        if self.config.tpm_limit:
            tpm_limiter = TPMLimiter.from_tpm(self.config.tpm_limit)
            limiters.append(tpm_limiter)
        
        if limiters:
            composite_limiter = CompositeLimiter(limiters)
            return RateLimiter(
                composite_limiter,
                events=self._events,
                estimator=self._token_estimator.estimate_tokens,
            )
        
        return None

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """Generate embedding with rate limiting."""
        # Apply rate limiting if configured
        if self._rate_limiter:
            async def _embed_call(text_input: str, **call_kwargs) -> LLMOutput[list[float], Any, Any]:
                embedding = await self._execute_embed(text_input, call_kwargs)
                return LLMOutput(output=embedding)
            
            target_func = self._rate_limiter.decorate(_embed_call)
            result = await target_func(text, **kwargs)
            return result.output
        else:
            return await self._execute_embed(text, kwargs)

    async def _execute_embed(self, text: str, kwargs: dict[str, Any]) -> list[float]:
        """Execute embedding generation."""
        try:
            # Build request body for Titan models
            body = {"inputText": text}
            
            # Add V2 model specific parameters
            if self.config.dimensions:
                body["dimensions"] = self.config.dimensions
            if "normalize" in kwargs or self.config.normalize:
                body["normalize"] = kwargs.get("normalize", self.config.normalize)
            if self.config.embedding_types:
                body["embeddingTypes"] = self.config.embedding_types
                
            # Make API call
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            
            # Parse response
            result = response["body"].read().decode()
            result_json = json.loads(result)
            
            # Extract embedding
            if "embedding" in result_json:
                return result_json["embedding"]
            elif "embeddingsByType" in result_json and "float" in result_json["embeddingsByType"]:
                return result_json["embeddingsByType"]["float"]
            else:
                raise BedrockAPIError("No embedding found in response")
                
        except ClientError as e:
            # Re-raise ClientError directly for retry mechanism
            logger.error(f"AWS Bedrock ClientError: {e.response.get('Error', {}).get('Message', str(e))}")
            raise

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        for text in text_list:
            embedding = await self.aembed(text, **kwargs)
            embeddings.append(embedding)
        return embeddings

    def embed(self, text: str, **kwargs) -> list[float]:
        """Synchronous version of aembed."""
        import asyncio
        return asyncio.run(self.aembed(text, **kwargs))

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """Synchronous version of aembed_batch."""
        import asyncio
        return asyncio.run(self.aembed_batch(text_list, **kwargs))