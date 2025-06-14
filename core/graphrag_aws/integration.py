# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Integration layer for GraphRAG compatibility."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .bedrock_models_enhanced import (
    EnhancedBedrockAnthropicChatLLM,
    EnhancedBedrockEmbeddingLLM,
)
from .config import BedrockAnthropicConfig, BedrockEmbeddingConfig
from .events import LLMEvents

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GraphRAGEnhancedBedrockChatLLM:
    """GraphRAG-compatible wrapper for enhanced Bedrock Chat LLM."""

    def __init__(
        self,
        *,
        name: str,
        config: Any,  # LanguageModelConfig from GraphRAG
        callbacks: Any | None = None,  # WorkflowCallbacks
        cache: Any | None = None,  # PipelineCache
    ) -> None:
        """Initialize GraphRAG-compatible Bedrock Chat LLM."""
        self.name = name
        self.graphrag_config = config
        self.callbacks = callbacks
        self.cache = cache

        # Convert GraphRAG config to enhanced config
        enhanced_config = self._create_enhanced_config(config)
        
        # Create enhanced LLM with events
        events = LLMEvents()
        self._enhanced_llm = EnhancedBedrockAnthropicChatLLM(
            enhanced_config, events=events
        )

    def _create_enhanced_config(self, graphrag_config: Any) -> BedrockAnthropicConfig:
        """Convert GraphRAG config to enhanced Bedrock config."""
        # Extract settings from GraphRAG config
        model = getattr(graphrag_config, 'model', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        api_base = getattr(graphrag_config, 'api_base', None)
        
        # Get additional parameters from config or use defaults
        max_tokens = getattr(graphrag_config, 'max_tokens', 4096)
        temperature = getattr(graphrag_config, 'temperature', 0.7)
        
        # Create enhanced config with production settings
        return BedrockAnthropicConfig(
            model=model,
            endpoint_url=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            # Production-ready settings
            rpm_limit=1000,  # Conservative default
            tpm_limit=100000,
            max_retries=10,
            burst_mode=True,
        )

    async def achat(self, prompt: str, history: list | None = None, **kwargs) -> Any:
        """GraphRAG-compatible chat interface."""
        try:
            # Call enhanced implementation
            result = await self._enhanced_llm.achat(prompt, history, **kwargs)
            
            # Log to GraphRAG callbacks if available
            if self.callbacks:
                await self._log_to_callbacks(prompt, result)
                
            return result
            
        except Exception as e:
            logger.error(f"Enhanced Bedrock Chat error: {str(e)}")
            if self.callbacks:
                await self._log_error_to_callbacks(str(e))
            raise

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> Any:
        """Synchronous GraphRAG-compatible chat interface."""
        import asyncio
        return asyncio.run(self.achat(prompt, history, **kwargs))

    async def _log_to_callbacks(self, prompt: str, result: Any) -> None:
        """Log successful operation to GraphRAG callbacks."""
        if hasattr(self.callbacks, 'on_llm_request'):
            await self.callbacks.on_llm_request(
                name=self.name,
                prompt=prompt,
                response=str(result)
            )

    async def _log_error_to_callbacks(self, error_msg: str) -> None:
        """Log error to GraphRAG callbacks."""
        if hasattr(self.callbacks, 'on_error'):
            await self.callbacks.on_error(
                name=self.name,
                error=error_msg
            )


class GraphRAGEnhancedBedrockEmbeddingLLM:
    """GraphRAG-compatible wrapper for enhanced Bedrock Embedding LLM."""

    def __init__(
        self,
        *,
        name: str,
        config: Any,  # LanguageModelConfig from GraphRAG
        callbacks: Any | None = None,  # WorkflowCallbacks
        cache: Any | None = None,  # PipelineCache
    ) -> None:
        """Initialize GraphRAG-compatible Bedrock Embedding LLM."""
        self.name = name
        self.graphrag_config = config
        self.callbacks = callbacks
        self.cache = cache

        # Convert GraphRAG config to enhanced config
        enhanced_config = self._create_enhanced_config(config)
        
        # Create enhanced LLM
        events = LLMEvents()
        self._enhanced_llm = EnhancedBedrockEmbeddingLLM(
            enhanced_config, events=events
        )

    def _create_enhanced_config(self, graphrag_config: Any) -> BedrockEmbeddingConfig:
        """Convert GraphRAG config to enhanced Bedrock embedding config."""
        # Extract settings from GraphRAG config
        model = getattr(graphrag_config, 'model', 'amazon.titan-embed-text-v2:0')
        api_base = getattr(graphrag_config, 'api_base', None)
        
        # Create enhanced config with production settings
        return BedrockEmbeddingConfig(
            model=model,
            endpoint_url=api_base,
            dimensions=1024,  # Default for Titan V2
            # Production-ready settings
            rpm_limit=2000,
            tpm_limit=500000,
            normalize=True,
        )

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """GraphRAG-compatible batch embedding interface."""
        try:
            # Call enhanced implementation
            result = await self._enhanced_llm.aembed_batch(text_list, **kwargs)
            
            # Log to GraphRAG callbacks if available
            if self.callbacks:
                await self._log_to_callbacks(len(text_list), len(result))
                
            return result
            
        except Exception as e:
            logger.error(f"Enhanced Bedrock Embedding error: {str(e)}")
            if self.callbacks:
                await self._log_error_to_callbacks(str(e))
            raise

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """Single text embedding."""
        result = await self._enhanced_llm.aembed(text, **kwargs)
        return result

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """Synchronous batch embedding."""
        import asyncio
        return asyncio.run(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """Synchronous single embedding."""
        import asyncio
        return asyncio.run(self.aembed(text, **kwargs))

    async def _log_to_callbacks(self, input_count: int, output_count: int) -> None:
        """Log successful operation to GraphRAG callbacks."""
        if hasattr(self.callbacks, 'on_embedding_request'):
            await self.callbacks.on_embedding_request(
                name=self.name,
                input_count=input_count,
                output_count=output_count
            )

    async def _log_error_to_callbacks(self, error_msg: str) -> None:
        """Log error to GraphRAG callbacks."""
        if hasattr(self.callbacks, 'on_error'):
            await self.callbacks.on_error(
                name=self.name,
                error=error_msg
            )