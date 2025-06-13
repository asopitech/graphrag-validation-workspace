# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for enhanced AWS Bedrock models."""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from graphrag_aws import (
    BedrockAPIError,
    BedrockAnthropicConfig,
    BedrockEmbeddingConfig,
    EnhancedBedrockAnthropicChatLLM,
    EnhancedBedrockEmbeddingLLM,
    LLMEvents,
    RetryStrategy,
    create_claude_3_5_sonnet,
    create_titan_embed_v2,
)


class TestEnhancedBedrockAnthropicChatLLM:
    """Test enhanced Anthropic Chat LLM."""

    @pytest.fixture
    def config(self):
        """Basic configuration."""
        return BedrockAnthropicConfig(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            rpm_limit=100,
            tpm_limit=10000,
            max_retries=3,
        )

    @pytest.fixture
    def mock_boto3_client(self):
        """Mock boto3 bedrock-runtime client."""
        with patch("boto3.client") as mock_client:
            yield mock_client

    def test_init_with_rate_limiting(self, config, mock_boto3_client):
        """Test initialization with rate limiting enabled."""
        mock_client_instance = Mock()
        mock_boto3_client.return_value = mock_client_instance
        
        llm = EnhancedBedrockAnthropicChatLLM(config)
        
        assert llm.config == config
        assert llm._rate_limiter is not None
        assert llm._retryer is not None
        assert llm._should_rate_limit() is True
        assert llm._should_retry() is True

    def test_init_without_rate_limiting(self, mock_boto3_client):
        """Test initialization without rate limiting."""
        config = BedrockAnthropicConfig(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            rpm_limit=None,
            tpm_limit=None,
            max_retries=0,
        )
        
        mock_client_instance = Mock()
        mock_boto3_client.return_value = mock_client_instance
        
        llm = EnhancedBedrockAnthropicChatLLM(config)
        
        assert llm._rate_limiter is None
        assert llm._retryer is None
        assert llm._should_rate_limit() is False
        assert llm._should_retry() is False

    @pytest.mark.asyncio
    async def test_achat_success(self, config, mock_boto3_client):
        """Test successful chat completion."""
        # Setup mock response
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "content": [
                {"type": "text", "text": "Hello! How can I help you today?"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 15}
        }
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        llm = EnhancedBedrockAnthropicChatLLM(config)
        
        # Test chat
        result = await llm.achat("Hello")
        
        assert result == "Hello! How can I help you today?"
        
        # Verify correct API call
        mock_client_instance.invoke_model.assert_called_once()
        call_args = mock_client_instance.invoke_model.call_args
        
        body = json.loads(call_args[1]["body"])
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["max_tokens"] == 4096
        assert body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]

    @pytest.mark.asyncio
    async def test_achat_with_history_and_rate_limiting(self, config, mock_boto3_client):
        """Test chat with history and rate limiting."""
        # Mock rate limiting by simulating delayed response
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "content": [{"type": "text", "text": "I understand the context."}],
            "usage": {"input_tokens": 25, "output_tokens": 10}
        }
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        llm = EnhancedBedrockAnthropicChatLLM(config)
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        result = await llm.achat("New question", history=history)
        
        assert result == "I understand the context."
        
        # Verify history is correctly formatted
        call_args = mock_client_instance.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        
        expected_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Previous question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Previous answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "New question"}]}
        ]
        
        assert body["messages"] == expected_messages

    @pytest.mark.asyncio
    async def test_retry_on_throttling(self, config, mock_boto3_client):
        """Test retry behavior on throttling error."""
        # Configure for fast retries in test
        config.max_retries = 2
        config.max_retry_wait = 0.1
        config.retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
        
        # Mock throttling error followed by success
        throttling_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel"
        )
        
        success_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "content": [{"type": "text", "text": "Success after retry"}],
            "usage": {"input_tokens": 10, "output_tokens": 8}
        }
        
        success_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.side_effect = [throttling_error, success_response]
        mock_boto3_client.return_value = mock_client_instance
        
        llm = EnhancedBedrockAnthropicChatLLM(config)
        
        result = await llm.achat("Test prompt")
        
        assert result == "Success after retry"
        assert mock_client_instance.invoke_model.call_count == 2


class TestEnhancedBedrockEmbeddingLLM:
    """Test enhanced Embedding LLM."""

    @pytest.fixture
    def config(self):
        """Basic embedding configuration."""
        return BedrockEmbeddingConfig(
            model="amazon.titan-embed-text-v2:0",
            dimensions=1024,
            rpm_limit=500,
            tpm_limit=50000,
        )

    @pytest.fixture
    def mock_boto3_client(self):
        """Mock boto3 bedrock-runtime client."""
        with patch("boto3.client") as mock_client:
            yield mock_client

    @pytest.mark.asyncio
    async def test_aembed_success(self, config, mock_boto3_client):
        """Test successful embedding generation."""
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "inputTextTokenCount": 5
        }
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        llm = EnhancedBedrockEmbeddingLLM(config)
        
        result = await llm.aembed("Test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        
        # Verify API call with V2 parameters
        call_args = mock_client_instance.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert body["inputText"] == "Test text"
        assert body["dimensions"] == 1024
        assert body["normalize"] is True

    @pytest.mark.asyncio
    async def test_aembed_batch_with_rate_limiting(self, config, mock_boto3_client):
        """Test batch embedding with rate limiting."""
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "inputTextTokenCount": 5
        }
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        llm = EnhancedBedrockEmbeddingLLM(config)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        result = await llm.aembed_batch(texts)
        
        assert len(result) == 3
        assert all(emb == [0.1, 0.2, 0.3, 0.4] for emb in result)
        assert mock_client_instance.invoke_model.call_count == 3


class TestFactoryFunctions:
    """Test factory functions."""

    @patch("boto3.client")
    def test_create_claude_3_5_sonnet(self, mock_boto3_client):
        """Test Claude 3.5 Sonnet factory."""
        mock_client_instance = Mock()
        mock_boto3_client.return_value = mock_client_instance
        
        llm = create_claude_3_5_sonnet(
            rpm_limit=500,
            tpm_limit=50000,
            max_retries=5,
        )
        
        assert isinstance(llm, EnhancedBedrockAnthropicChatLLM)
        assert llm.config.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert llm.config.rpm_limit == 500
        assert llm.config.tpm_limit == 50000
        assert llm.config.max_retries == 5

    @patch("boto3.client")
    def test_create_titan_embed_v2(self, mock_boto3_client):
        """Test Titan Embed V2 factory."""
        mock_client_instance = Mock()
        mock_boto3_client.return_value = mock_client_instance
        
        llm = create_titan_embed_v2(
            dimensions=512,
            rpm_limit=1000,
        )
        
        assert isinstance(llm, EnhancedBedrockEmbeddingLLM)
        assert llm.config.model == "amazon.titan-embed-text-v2:0"
        assert llm.config.dimensions == 512
        assert llm.config.rpm_limit == 1000

    @patch("boto3.client")
    def test_factory_with_custom_events(self, mock_boto3_client):
        """Test factory with custom event handler."""
        mock_client_instance = Mock()
        mock_boto3_client.return_value = mock_client_instance
        
        events = LLMEvents()
        llm = create_claude_3_5_sonnet(events=events)
        
        assert llm._events is events