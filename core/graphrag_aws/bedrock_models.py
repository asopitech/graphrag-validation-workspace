# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""AWS Bedrock LLM provider definitions with corrected API specifications."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.response.base import ModelResponse

logger = logging.getLogger(__name__)


class BedrockAPIError(Exception):
    """AWS Bedrock API specific error."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class BedrockAnthropicChatLLM:
    """AWS Bedrock Anthropic Claude Chat Model provider with correct API specification."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize the Bedrock Anthropic Chat model.
        
        Args:
            name: Model name identifier
            config: Language model configuration
            callbacks: Optional workflow callbacks
            cache: Optional pipeline cache
        """
        self.config = config
        self.model_id = config.model
        self.endpoint_url = config.api_base or os.environ.get("BEDROCK_ENDPOINT")
        self.region = os.environ.get("AWS_REGION", "us-east-1")
        
        # Initialize boto3 client with error handling
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

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """Generate a chat response using Anthropic Claude via AWS Bedrock.
        
        Args:
            prompt: User input text
            history: Optional conversation history 
            **kwargs: Additional model parameters
            
        Returns:
            ModelResponse with generated content
        """
        from graphrag.language_model.response.base import BaseModelOutput, BaseModelResponse
        
        try:
            # Build messages according to Anthropic API specification
            messages = []
            
            # Add history if provided
            if history:
                for h in history:
                    if isinstance(h, dict) and "role" in h and "content" in h:
                        # Ensure content is in proper format
                        content = h["content"]
                        if isinstance(content, str):
                            content = [{"type": "text", "text": content}]
                        messages.append({"role": h["role"], "content": content})
            
            # Add current prompt
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            })
            
            # Build request body according to Anthropic Bedrock specification
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get("max_tokens", 4096),
                "messages": messages,
            }
            
            # Add optional parameters
            if "temperature" in kwargs:
                body["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                body["top_p"] = kwargs["top_p"]
            if "system" in kwargs:
                body["system"] = kwargs["system"]
                
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
            
            # Extract content according to Anthropic response format
            content = ""
            if "content" in result_json:
                for content_block in result_json["content"]:
                    if content_block.get("type") == "text":
                        content += content_block.get("text", "")
            
            return BaseModelResponse(
                output=BaseModelOutput(
                    content=content,
                    full_response=result_json
                )
            )
            
        except ClientError as e:
            error_msg = f"AWS Bedrock ClientError: {e.response.get('Error', {}).get('Message', str(e))}"
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Bedrock API response: {str(e)}"
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in Bedrock Anthropic model: {str(e)}"
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e

    def chat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """Synchronous version of achat."""
        import asyncio
        return asyncio.run(self.achat(prompt, history, **kwargs))

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat response (placeholder implementation)."""
        # TODO: Implement streaming with invoke_model_with_response_stream
        response = await self.achat(prompt, history, **kwargs)
        yield response.output.content

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """Synchronous streaming chat (placeholder implementation)."""
        import asyncio
        
        async def _async_gen():
            async for chunk in self.achat_stream(prompt, history, **kwargs):
                yield chunk
                
        return asyncio.run(_async_gen())


class BedrockEmbeddingLLM:
    """AWS Bedrock Embedding Model provider for Amazon Titan models."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize the Bedrock Embedding model.
        
        Args:
            name: Model name identifier
            config: Language model configuration
            callbacks: Optional workflow callbacks
            cache: Optional pipeline cache
        """
        self.config = config
        self.model_id = config.model
        self.endpoint_url = config.api_base or os.environ.get("BEDROCK_ENDPOINT")
        self.region = os.environ.get("AWS_REGION", "us-east-1")
        
        # Initialize boto3 client with error handling
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

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings for a batch of text strings.
        
        Args:
            text_list: List of text strings to embed
            **kwargs: Additional model parameters
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in text_list:
            embedding = await self.aembed(text, **kwargs)
            embeddings.append(embedding)
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            **kwargs: Additional model parameters
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Build request body for Amazon Titan Embed Text models
            body = {
                "inputText": text,
            }
            
            # Add optional parameters for V2 models
            if "dimensions" in kwargs:
                body["dimensions"] = kwargs["dimensions"]
            if "normalize" in kwargs:
                body["normalize"] = kwargs["normalize"]
            if "embeddingTypes" in kwargs:
                body["embeddingTypes"] = kwargs["embeddingTypes"]
                
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
            
            # Extract embedding according to Titan response format
            if "embedding" in result_json:
                return result_json["embedding"]
            elif "embeddingsByType" in result_json and "float" in result_json["embeddingsByType"]:
                return result_json["embeddingsByType"]["float"]
            else:
                raise BedrockAPIError("No embedding found in response")
                
        except ClientError as e:
            error_msg = f"AWS Bedrock ClientError: {e.response.get('Error', {}).get('Message', str(e))}"
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Bedrock API response: {str(e)}"
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in Bedrock Embedding model: {str(e)}"
            logger.error(error_msg)
            raise BedrockAPIError(error_msg) from e

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """Synchronous version of aembed_batch."""
        import asyncio
        return asyncio.run(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """Synchronous version of aembed."""
        import asyncio
        return asyncio.run(self.aembed(text, **kwargs))


class BedrockChatLLM:
    """AWS Bedrock generic Chat Model provider (throws exception for unsupported models)."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize generic Bedrock chat model."""
        self.config = config
        self.model_id = config.model

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """Raise exception for unsupported model."""
        raise ValueError(
            f"BedrockChatLLM: Unsupported or invalid model ID: {self.model_id}. "
            f"Please use BedrockAnthropicChatLLM for Anthropic models."
        )

    def chat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """Raise exception for unsupported model."""
        raise ValueError(
            f"BedrockChatLLM: Unsupported or invalid model ID: {self.model_id}. "
            f"Please use BedrockAnthropicChatLLM for Anthropic models."
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Raise exception for unsupported model."""
        raise ValueError(
            f"BedrockChatLLM: Unsupported or invalid model ID: {self.model_id}"
        )
        yield  # Make this a generator

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """Raise exception for unsupported model."""
        raise ValueError(
            f"BedrockChatLLM: Unsupported or invalid model ID: {self.model_id}"
        )
        yield  # Make this a generator