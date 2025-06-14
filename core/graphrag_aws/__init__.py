# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""GraphRAG AWS Bedrock integration with fnllm-inspired features."""

from .bedrock_models_enhanced import (
    BedrockAPIError,
    EnhancedBedrockAnthropicChatLLM,
    EnhancedBedrockEmbeddingLLM,
)
from .config import (
    ANTHROPIC_CLAUDE_3_5_SONNET,
    ANTHROPIC_CLAUDE_3_HAIKU,
    TITAN_EMBED_TEXT_V2,
    TITAN_EMBED_TEXT_V1,
    BedrockAnthropicConfig,
    BedrockBaseConfig,
    BedrockEmbeddingConfig,
)
from .events import LLMEvents
from .factories import (
    create_bedrock_anthropic_llm,
    create_bedrock_embedding_llm,
    create_claude_3_5_sonnet,
    create_claude_3_haiku,
    create_titan_embed_v2,
)
from .types import RetryStrategy

__all__ = [
    # Models
    "EnhancedBedrockAnthropicChatLLM",
    "EnhancedBedrockEmbeddingLLM",
    "BedrockAPIError",
    
    # Configuration
    "BedrockBaseConfig",
    "BedrockAnthropicConfig", 
    "BedrockEmbeddingConfig",
    "ANTHROPIC_CLAUDE_3_5_SONNET",
    "ANTHROPIC_CLAUDE_3_HAIKU",
    "TITAN_EMBED_TEXT_V2",
    "TITAN_EMBED_TEXT_V1",
    "RetryStrategy",
    
    # Events
    "LLMEvents",
    
    # Factories
    "create_bedrock_anthropic_llm",
    "create_bedrock_embedding_llm",
    "create_claude_3_5_sonnet",
    "create_claude_3_haiku",
    "create_titan_embed_v2",
]