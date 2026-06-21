import streamlit as st
import pandas as pd
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR, TEXT_COLOR, ACCENT_COLOR
)

# Safe imports to prevent crashing if user hasn't pip installed yet
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    LANGCHAIN_READY = True
except ImportError:
    LANGCHAIN_READY = False

def _inject_css():
    st.markdown(f"""
    <style>
      .chat-header {{ color: {PRIMARY_COLOR}; font-weight: 700; margin-top: 1rem; margin-bottom: 0.5rem; border-bottom: 1px solid {PRIMARY_COLOR}33; padding-bottom: 0.3rem; }}
      .chat-bubble-user {{ background: {PRIMARY_COLOR}22; border: 1px solid {PRIMARY_COLOR}55; border-radius: 12px; padding: 0.7rem 1rem; margin: 0.4rem 0; color: {TEXT_COLOR}; }}
      .chat-bubble-answer {{ background: {SURFACE_COLOR}; border: 1px solid #2a2a45; border-left: 4px solid {ACCENT_COLOR}; border-radius: 12px; padding: 0.9rem 1.1rem; margin: 0.4rem 0; color: {TEXT_COLOR}; }}
    </style>
    """, unsafe_allow_html=True)

def show():
    _inject_css()
    st.title("💬 Ask Your Data")
    st.caption("Phase 14 — Autonomous Data Analyst powered by LangChain & Gemini")

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload data first.")
        return

    if not LANGCHAIN_READY:
        st.error("Missing AI dependencies. Please run: `pip install langchain langchain-google-genai langchain-experimental tabulate`")
        return

    # Check for Gemini API Key in secrets, fallback to UI input if missing
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in `.streamlit/secrets.toml`. Please configure it to enable the agent.")
        st.code('GEMINI_API_KEY = "your-key-here"', language="toml")
        return

    st.info("ℹ️ Your dataset is being analyzed locally by a LangChain Agent powered by Gemini 2.5 Flash. It can perform multi-step reasoning, handle complex groupings, and detect trends autonomously.")

    st.markdown('<div class="chat-header">Agent Chat Console</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "agent_chat_history" not in st.session_state:
        st.session_state.agent_chat_history = [{"role": "assistant", "content": "Hello! I am your autonomous data analyst. Ask me anything complex about your dataset."}]

    # Render history
    for msg in st.session_state.agent_chat_history:
        css_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-answer"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    # Chat Input
    prompt = st.chat_input("Ask a question about your data...")

    if prompt:
        # Append and show user message
        st.session_state.agent_chat_history.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-bubble-user">🧑 {prompt}</div>', unsafe_allow_html=True)

        # Agent execution
        with st.spinner("🤖 Analyzing data and writing execution code..."):
            try:
                # Initialize Gemini Model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    google_api_key=api_key, 
                    temperature=0.0
                )
                
                # Create the Pandas execution agent
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type="tool-calling",
                    allow_dangerous_code=True, # Required by LangChain to run pandas queries locally
                    handle_parsing_errors=True
                )
                
                response = agent.invoke({"input": prompt})
                answer = response.get("output", "I could not generate an answer.")
                
                st.markdown(f'<div class="chat-bubble-answer">🤖 {answer}</div>', unsafe_allow_html=True)
                st.session_state.agent_chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ **Agent Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.agent_chat_history.append({"role": "assistant", "content": error_msg})
                
        if st.session_state.agent_chat_history:
            if st.button("🗑️ Clear conversation"):
                st.session_state.agent_chat_history = [{"role": "assistant", "content": "Hello! I am your autonomous data analyst. Ask me anything complex about your dataset."}]
                st.rerun()