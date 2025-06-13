# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for AWS Bedrock models."""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from graphrag_aws.bedrock_models import (
    BedrockAnthropicChatLLM,
    BedrockAPIError,
    BedrockChatLLM,
    BedrockEmbeddingLLM,
)


class TestBedrockAnthropicChatLLM:
    """Test cases for BedrockAnthropicChatLLM."""

    @pytest.fixture
    def mock_config(self):
        """Mock language model configuration."""
        config = Mock()
        config.model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        config.api_base = None
        return config

    @pytest.fixture
    def mock_boto3_client(self):
        """Mock boto3 bedrock-runtime client."""
        with patch("boto3.client") as mock_client:
            yield mock_client

    def test_init_success(self, mock_config, mock_boto3_client):
        """Test successful initialization."""
        mock_client_instance = Mock()
        mock_boto3_client.return_value = mock_client_instance
        
        model = BedrockAnthropicChatLLM(
            name="test",
            config=mock_config,
        )
        
        assert model.config == mock_config
        assert model.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert model.region == "us-east-1"  # default
        mock_boto3_client.assert_called_once()

    def test_init_no_credentials(self, mock_config, mock_boto3_client):
        """Test initialization failure due to missing credentials."""
        mock_boto3_client.side_effect = NoCredentialsError()
        
        with pytest.raises(BedrockAPIError, match="AWS credentials not found"):
            BedrockAnthropicChatLLM(
                name="test",
                config=mock_config,
            )

    @pytest.mark.asyncio
    async def test_achat_success(self, mock_config, mock_boto3_client):
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
        
        model = BedrockAnthropicChatLLM(name="test", config=mock_config)
        
        # Test chat
        result = await model.achat("Hello")
        
        assert result.output.content == "Hello! How can I help you today?"
        assert result.output.full_response == response_data
        
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
    async def test_achat_with_history(self, mock_config, mock_boto3_client):
        """Test chat with conversation history."""
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "content": [{"type": "text", "text": "Response"}]
        }
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        model = BedrockAnthropicChatLLM(name="test", config=mock_config)
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        await model.achat("New question", history=history)
        
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
    async def test_achat_client_error(self, mock_config, mock_boto3_client):
        """Test handling of AWS ClientError."""
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid model"}},
            "InvokeModel"
        )
        mock_boto3_client.return_value = mock_client_instance
        
        model = BedrockAnthropicChatLLM(name="test", config=mock_config)
        
        with pytest.raises(BedrockAPIError, match="AWS Bedrock ClientError"):
            await model.achat("Hello")

    @pytest.mark.asyncio
    async def test_achat_http_error(self, mock_config, mock_boto3_client):
        """Test handling of HTTP error response."""
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 400},
            "body": Mock()
        }
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        model = BedrockAnthropicChatLLM(name="test", config=mock_config)
        
        with pytest.raises(BedrockAPIError, match="Bedrock API returned status 400"):
            await model.achat("Hello")

    def test_chat_sync(self, mock_config, mock_boto3_client):
        """Test synchronous chat method."""
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "content": [{"type": "text", "text": "Sync response"}]
        }
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        model = BedrockAnthropicChatLLM(name="test", config=mock_config)
        
        result = model.chat("Hello")
        assert result.output.content == "Sync response"


class TestBedrockEmbeddingLLM:
    """Test cases for BedrockEmbeddingLLM."""

    @pytest.fixture
    def mock_config(self):
        """Mock embedding model configuration."""
        config = Mock()
        config.model = "amazon.titan-embed-text-v2:0"
        config.api_base = None
        return config

    @pytest.fixture
    def mock_boto3_client(self):
        """Mock boto3 bedrock-runtime client."""
        with patch("boto3.client") as mock_client:
            yield mock_client

    @pytest.mark.asyncio
    async def test_aembed_success(self, mock_config, mock_boto3_client):
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
        
        model = BedrockEmbeddingLLM(name="test", config=mock_config)
        
        result = await model.aembed("Test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        
        # Verify API call
        call_args = mock_client_instance.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert body["inputText"] == "Test text"

    @pytest.mark.asyncio
    async def test_aembed_v2_with_options(self, mock_config, mock_boto3_client):
        """Test embedding with V2 options."""
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {
            "embeddingsByType": {
                "float": [0.1, 0.2, 0.3, 0.4],
                "binary": [1, 0, 1, 0]
            },
            "inputTextTokenCount": 5
        }
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        model = BedrockEmbeddingLLM(name="test", config=mock_config)
        
        result = await model.aembed(
            "Test text",
            dimensions=512,
            normalize=True,
            embeddingTypes=["float"]
        )
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        
        # Verify API call with options
        call_args = mock_client_instance.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert body["inputText"] == "Test text"
        assert body["dimensions"] == 512
        assert body["normalize"] is True
        assert body["embeddingTypes"] == ["float"]

    @pytest.mark.asyncio
    async def test_aembed_batch(self, mock_config, mock_boto3_client):
        """Test batch embedding generation."""
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
        
        model = BedrockEmbeddingLLM(name="test", config=mock_config)
        
        result = await model.aembed_batch(["Text 1", "Text 2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_aembed_no_embedding_error(self, mock_config, mock_boto3_client):
        """Test error when no embedding found in response."""
        mock_response = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "body": Mock()
        }
        
        response_data = {"inputTextTokenCount": 5}  # No embedding field
        
        mock_response["body"].read.return_value.decode.return_value = json.dumps(response_data)
        
        mock_client_instance = Mock()
        mock_client_instance.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client_instance
        
        model = BedrockEmbeddingLLM(name="test", config=mock_config)
        
        with pytest.raises(BedrockAPIError, match="No embedding found in response"):
            await model.aembed("Test text")

    def test_embed_sync(self, mock_config, mock_boto3_client):
        """Test synchronous embedding method."""
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
        
        model = BedrockEmbeddingLLM(name="test", config=mock_config)
        
        result = model.embed("Test text")
        assert result == [0.1, 0.2, 0.3, 0.4]


class TestBedrockChatLLM:
    """Test cases for BedrockChatLLM (fallback)."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.model = "unsupported.model:0"
        return config

    def test_achat_raises_error(self, mock_config):
        """Test that unsupported model raises error."""
        model = BedrockChatLLM(name="test", config=mock_config)
        
        with pytest.raises(ValueError, match="Unsupported or invalid model ID"):
            import asyncio
            asyncio.run(model.achat("Hello"))

    def test_chat_raises_error(self, mock_config):
        """Test that synchronous chat raises error."""
        model = BedrockChatLLM(name="test", config=mock_config)
        
        with pytest.raises(ValueError, match="Unsupported or invalid model ID"):
            model.chat("Hello")