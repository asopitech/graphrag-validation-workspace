#!/usr/bin/env python3
# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Streamlit chat application demonstrating enhanced AWS Bedrock integration."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Add core module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from graphrag_aws import (
    create_claude_3_5_sonnet,
    create_titan_embed_v2,
    LLMEvents,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced AWS Bedrock Chat",
    page_icon="=�",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CustomEvents(LLMEvents):
    """Custom event handler for Streamlit integration."""
    
    def __init__(self):
        super().__init__()
        self.metrics = {
            "requests": 0,
            "tokens": 0,
            "retries": 0,
            "rate_limits": 0
        }
    
    async def on_execute_llm(self):
        """Called when LLM execution starts."""
        self.metrics["requests"] += 1
        if "metrics_container" in st.session_state:
            st.session_state.metrics_container.metric(
                "Total Requests", self.metrics["requests"]
            )
    
    async def on_usage(self, usage):
        """Called when usage metrics are available."""
        self.metrics["tokens"] += getattr(usage, 'total_tokens', 0)
        if "metrics_container" in st.session_state:
            st.session_state.metrics_container.metric(
                "Total Tokens", self.metrics["tokens"]
            )
    
    async def on_retryable_error(self, error, attempt):
        """Called on retryable errors."""
        self.metrics["retries"] += 1
        if "metrics_container" in st.session_state:
            st.session_state.metrics_container.metric(
                "Retries", self.metrics["retries"]
            )
    
    async def on_limit_acquired(self, manifest):
        """Called when rate limit is acquired."""
        self.metrics["rate_limits"] += 1
        if "metrics_container" in st.session_state:
            st.session_state.metrics_container.metric(
                "Rate Limits", self.metrics["rate_limits"]
            )


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "llm" not in st.session_state:
        events = CustomEvents()
        st.session_state.events = events
        st.session_state.llm = None
    
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None


def create_sidebar():
    """Create sidebar with configuration options."""
    st.sidebar.title("=' Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Chat Model",
        [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ],
        key="model_choice"
    )
    
    # Rate limiting settings
    st.sidebar.subheader("Rate Limiting")
    rpm_limit = st.sidebar.slider("RPM Limit", 10, 2000, 1000, key="rpm_limit")
    tpm_limit = st.sidebar.slider("TPM Limit", 1000, 200000, 100000, key="tpm_limit")
    
    # Retry settings
    st.sidebar.subheader("Retry Settings")
    max_retries = st.sidebar.slider("Max Retries", 0, 20, 10, key="max_retries")
    burst_mode = st.sidebar.checkbox("Burst Mode", True, key="burst_mode")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, key="temperature")
    max_tokens = st.sidebar.slider("Max Tokens", 512, 8192, 4096, key="max_tokens")
    
    return {
        "model": model_choice,
        "rpm_limit": rpm_limit,
        "tpm_limit": tpm_limit,
        "max_retries": max_retries,
        "burst_mode": burst_mode,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def create_llm(config, events):
    """Create LLM with given configuration."""
    try:
        if "claude-3-5-sonnet" in config["model"]:
            llm = create_claude_3_5_sonnet(
                rpm_limit=config["rpm_limit"],
                tpm_limit=config["tpm_limit"],
                max_retries=config["max_retries"],
                burst_mode=config["burst_mode"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                events=events,
            )
        else:  # Haiku
            from graphrag_aws import create_claude_3_haiku
            llm = create_claude_3_haiku(
                rpm_limit=config["rpm_limit"],
                tpm_limit=config["tpm_limit"],
                max_retries=config["max_retries"],
                burst_mode=config["burst_mode"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                events=events,
            )
        
        return llm, None
        
    except Exception as e:
        return None, str(e)


def display_metrics():
    """Display real-time metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.session_state.metrics_container = st.empty()
        st.metric("Requests", 0)
    
    with col2:
        st.metric("Tokens", 0)
    
    with col3:
        st.metric("Retries", 0)
    
    with col4:
        st.metric("Rate Limits", 0)


async def generate_response(llm, prompt, history):
    """Generate response using enhanced LLM."""
    try:
        # Convert Streamlit messages to LLM format
        llm_history = []
        for msg in history:
            llm_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Generate response
        response = await llm.achat(prompt, history=llm_history)
        return response, None
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return None, str(e)


def main():
    """Main Streamlit application."""
    st.title("=� Enhanced AWS Bedrock Chat")
    st.markdown("*Powered by fnllm-inspired features: Rate Limiting, Retries, and Advanced Error Handling*")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    config = create_sidebar()
    
    # Display metrics
    display_metrics()
    
    # Create LLM button
    if st.sidebar.button("= Apply Configuration", type="primary"):
        with st.spinner("Initializing enhanced LLM..."):
            llm, error = create_llm(config, st.session_state.events)
            if llm:
                st.session_state.llm = llm
                st.sidebar.success(" LLM initialized successfully!")
            else:
                st.sidebar.error(f"L Error: {error}")
    
    # AWS credentials check
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        st.warning("� AWS credentials not found. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION environment variables.")
    
    # Chat interface
    st.subheader("=� Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if not st.session_state.llm:
            st.error("Please configure and initialize the LLM first using the sidebar.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                # Run async function
                response, error = asyncio.run(generate_response(
                    st.session_state.llm, 
                    prompt, 
                    st.session_state.messages[:-1]  # Exclude current user message
                ))
                
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    error_msg = f"L Error generating response: {error}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Configuration display
    with st.expander("= Current Configuration", expanded=False):
        st.json(config)
    
    # Chat history management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("=� Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("=� Export Chat"):
            chat_export = {
                "messages": st.session_state.messages,
                "config": config,
                "timestamp": str(pd.Timestamp.now())
            }
            st.download_button(
                "=� Download Chat JSON",
                json.dumps(chat_export, indent=2),
                "chat_export.json",
                "application/json"
            )


if __name__ == "__main__":
    main()