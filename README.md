# 💬 GraphRAG Chat Application Workspace

**Enhanced AWS Bedrock Chat Interface with Real-time Configuration**

## 📋 Overview

This workspace is dedicated to the chat application component of the GraphRAG AWS Bedrock integration. It provides an interactive Streamlit-based interface for testing and demonstrating the enhanced AWS Bedrock implementation with real-time configuration and monitoring capabilities.

## 🏗️ Architecture

```
workspace-chat/
├── chat-app/                     # Streamlit chat application
│   ├── streamlit_app.py         # Main application
│   └── run_app.sh               # Launch script
├── core/                        # Enhanced Bedrock implementation
│   └── graphrag_aws/            # Core package
│       ├── bedrock_models_enhanced.py
│       ├── limiting/            # Rate limiting
│       ├── services/            # Core services
│       ├── config.py            # Configuration
│       ├── factories.py         # Factory functions
│       └── integration.py       # GraphRAG compatibility
├── deployment/                  # Deployment configurations
├── upstream/                    # Upstream dependencies
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to workspace-chat
cd workspace-chat

# Setup Python environment (recommended: Python 3.11.11)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. AWS Configuration

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# Or use AWS profile
export AWS_PROFILE=your_aws_profile
```

### 3. Launch Chat Application

```bash
# Quick start with launch script
cd chat-app
./run_app.sh

# Or manual launch
streamlit run streamlit_app.py --server.port=8501
```

**Access:** http://localhost:8501

## ✨ Features

### 🎛️ Real-time Configuration
- **Model Selection**: Choose between Claude 3.5 Sonnet and Claude 3 Haiku
- **Rate Limiting**: Adjust RPM/TPM limits in real-time
- **Retry Settings**: Configure max retries and burst mode
- **Model Parameters**: Temperature, max tokens, and more

### 📊 Live Monitoring
- **Request Metrics**: Track total requests and tokens used
- **Retry Statistics**: Monitor retry attempts and patterns
- **Rate Limit Status**: Real-time rate limiting information
- **Performance Metrics**: Response times and throughput

### 💬 Interactive Chat
- **Conversation History**: Persistent chat sessions
- **Message Export**: Download chat sessions as JSON
- **Clear History**: Reset conversation state
- **Error Handling**: Graceful error display and recovery

### 🔧 Advanced Settings
- **Burst Mode**: Handle traffic spikes
- **Custom Endpoints**: Support for VPC endpoints
- **Logging Levels**: Adjustable verbosity
- **Debug Mode**: Detailed error information

## 🎯 Usage Examples

### Basic Chat Session

1. **Launch Application**: `./run_app.sh`
2. **Configure Model**: Select Claude 3.5 Sonnet from sidebar
3. **Set Rate Limits**: Adjust RPM to 1000, TPM to 100,000
4. **Initialize LLM**: Click "Apply Configuration"
5. **Start Chatting**: Type your message in the chat input

### Performance Testing

1. **Configure for High Load**:
   - Model: Claude 3 Haiku (faster)
   - RPM Limit: 2000
   - Burst Mode: Enabled
   - Max Retries: 5

2. **Monitor Metrics**: Watch real-time statistics
3. **Test Rate Limiting**: Send rapid requests to trigger limits
4. **Observe Retries**: Monitor automatic retry behavior

### Configuration Examples

**High Performance Setup**:
```
Model: Claude 3 Haiku
RPM Limit: 2000
TPM Limit: 200000
Temperature: 0.3
Max Retries: 5
Burst Mode: Enabled
```

**High Quality Setup**:
```
Model: Claude 3.5 Sonnet
RPM Limit: 1000
TPM Limit: 100000
Temperature: 0.7
Max Retries: 10
Burst Mode: Enabled
```

## 🔧 Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Optional: Custom endpoint
BEDROCK_ENDPOINT=https://bedrock-runtime.us-east-1.amazonaws.com

# Application Settings
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
```

### Streamlit Configuration

The application automatically creates a `.streamlit/config.toml` file with optimized settings:

```toml
[server]
port = 8501
headless = true
enableCORS = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
```

## 🚨 Troubleshooting

### Common Issues

**AWS Credentials Error**
```
⚠️ AWS credentials not found. Please set AWS_ACCESS_KEY_ID, 
AWS_SECRET_ACCESS_KEY, and AWS_REGION environment variables.
```
**Solution**: Set environment variables or AWS profile

**LLM Initialization Error**
```
❌ Error: AWS credentials not found. Please configure AWS credentials.
```
**Solution**: Check AWS configuration and permissions

**Rate Limiting Issues**
- Reduce RPM/TPM limits for new AWS accounts
- Enable burst mode for temporary spikes
- Monitor retry metrics for optimization

**Import Errors**
```
❌ Error creating LLM: No module named 'graphrag_aws'
```
**Solution**: Ensure you're in the correct directory and dependencies are installed

### Performance Optimization

**For High Throughput**:
- Use Claude 3 Haiku (faster, cheaper)
- Set higher RPM/TPM limits
- Enable burst mode
- Reduce temperature for consistency

**For High Quality**:
- Use Claude 3.5 Sonnet
- Increase max retries
- Lower temperature (0.1-0.3)
- Monitor token usage

## 📊 Metrics and Monitoring

The application provides real-time metrics:

- **Requests**: Total number of API calls
- **Tokens**: Token usage across all requests
- **Retries**: Number of retry attempts
- **Rate Limits**: Rate limiting events

### Export Capabilities

- **Chat Export**: Download conversations as JSON
- **Metrics Export**: Performance data export
- **Configuration Export**: Save current settings

## 🔄 Development Workflow

### Local Development

1. **Edit Application**: Modify `streamlit_app.py`
2. **Test Changes**: Streamlit auto-reloads on file changes
3. **Debug Issues**: Use browser developer tools and Streamlit logs
4. **Test Performance**: Use different configurations

### Adding Features

1. **New Metrics**: Extend `CustomEvents` class
2. **UI Components**: Add Streamlit widgets
3. **Configuration Options**: Update sidebar controls
4. **Export Features**: Enhance download functionality

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Enhanced Implementation Details](../workspace-core/README_ENHANCED.md)

## 📄 License

Copyright (c) 2025 Microsoft Corporation. Licensed under the MIT License.

---

**🎉 Ready for Interactive Testing!** This chat application provides a comprehensive interface for testing and demonstrating the enhanced AWS Bedrock integration with real-time configuration and monitoring capabilities.