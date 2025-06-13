# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Model factory for AWS Bedrock integration (without Nova models)."""

from collections.abc import Callable
from typing import Any, ClassVar

from graphrag.config.enums import ModelType
from graphrag.language_model.protocol import ChatModel, EmbeddingModel

# Import corrected Bedrock models
from .bedrock_models import (
    BedrockAnthropicChatLLM,
    BedrockChatLLM,
    BedrockEmbeddingLLM,
)


class BedrockModelFactory:
    """Factory for creating Bedrock Model instances."""

    _chat_registry: ClassVar[dict[str, Callable[..., ChatModel]]] = {}
    _embedding_registry: ClassVar[dict[str, Callable[..., EmbeddingModel]]] = {}

    @classmethod
    def register_chat(cls, model_type: str, creator: Callable[..., ChatModel]) -> None:
        """Register a ChatModel implementation."""
        cls._chat_registry[model_type] = creator

    @classmethod
    def register_embedding(
        cls, model_type: str, creator: Callable[..., EmbeddingModel]
    ) -> None:
        """Register an EmbeddingModel implementation."""
        cls._embedding_registry[model_type] = creator

    @classmethod
    def create_chat_model(cls, model_type: str, **kwargs: Any) -> ChatModel:
        """Create a ChatModel instance."""
        if model_type not in cls._chat_registry:
            msg = f"ChatModel implementation '{model_type}' is not registered."
            raise ValueError(msg)
        return cls._chat_registry[model_type](**kwargs)

    @classmethod
    def create_embedding_model(cls, model_type: str, **kwargs: Any) -> EmbeddingModel:
        """Create an EmbeddingModel instance."""
        if model_type not in cls._embedding_registry:
            msg = f"EmbeddingModel implementation '{model_type}' is not registered."
            raise ValueError(msg)
        return cls._embedding_registry[model_type](**kwargs)

    @classmethod
    def get_chat_models(cls) -> list[str]:
        """Get the registered ChatModel implementations."""
        return list(cls._chat_registry.keys())

    @classmethod
    def get_embedding_models(cls) -> list[str]:
        """Get the registered EmbeddingModel implementations."""
        return list(cls._embedding_registry.keys())

    @classmethod
    def is_supported_chat_model(cls, model_type: str) -> bool:
        """Check if the given model type is supported."""
        return model_type in cls._chat_registry

    @classmethod
    def is_supported_embedding_model(cls, model_type: str) -> bool:
        """Check if the given model type is supported."""
        return model_type in cls._embedding_registry


# Register Bedrock implementations (excluding Nova models)
BedrockModelFactory.register_chat(
    ModelType.BedrockChat, lambda **kwargs: BedrockChatLLM(**kwargs)
)
BedrockModelFactory.register_chat(
    ModelType.BedrockAnthropicChat, lambda **kwargs: BedrockAnthropicChatLLM(**kwargs)
)
# Note: BedrockNovaChat removed as per requirements

BedrockModelFactory.register_embedding(
    ModelType.BedrockTextEmbeddingV2, lambda **kwargs: BedrockEmbeddingLLM(**kwargs)
)