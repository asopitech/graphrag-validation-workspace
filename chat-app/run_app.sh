#!/bin/bash
# Enhanced AWS Bedrock Chat App Launcher

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Enhanced AWS Bedrock Chat App${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo -e "${YELLOW}Please run the following commands from the workspace-core directory:${NC}"
    echo -e "${YELLOW}  python -m venv .venv${NC}"
    echo -e "${YELLOW}  source .venv/bin/activate${NC}"
    echo -e "${YELLOW}  pip install -r requirements.txt${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${BLUE}📍 Python version: ${PYTHON_VERSION}${NC}"

# Activate virtual environment
echo -e "${YELLOW}🔄 Activating virtual environment...${NC}"
source ../.venv/bin/activate

# Check required packages
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
python -c "import streamlit, boto3, pydantic, aiolimiter, tenacity" 2>/dev/null || {
    echo -e "${RED}❌ Required packages not installed${NC}"
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r ../requirements.txt
}

# Set environment variables if .env exists
if [ -f "../.env" ]; then
    echo -e "${YELLOW}📋 Loading environment variables from .env${NC}"
    export $(cat ../.env | grep -v '^#' | xargs)
fi

# Check AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ] && [ -z "$AWS_PROFILE" ]; then
    echo -e "${YELLOW}⚠️  AWS credentials not found in environment${NC}"
    echo -e "${YELLOW}Please set one of the following:${NC}"
    echo -e "${YELLOW}  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION${NC}"
    echo -e "${YELLOW}  - AWS_PROFILE${NC}"
    echo -e "${YELLOW}The app will still run but AWS calls will fail${NC}"
else
    echo -e "${GREEN}✅ AWS credentials found${NC}"
fi

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create Streamlit config
cat > .streamlit/config.toml << EOF
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOF

echo -e "${GREEN}🎯 Starting Enhanced AWS Bedrock Chat App...${NC}"
echo -e "${BLUE}📱 App will be available at: http://localhost:8501${NC}"
echo -e "${YELLOW}💡 Press Ctrl+C to stop the app${NC}"
echo ""

# Start Streamlit app
streamlit run streamlit_app.py --server.port=8501