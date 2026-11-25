import os
from datetime import datetime

import streamlit as st
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools.search import tavily_search
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Page Setup & Sidebar Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 Deep Research Agent")

with st.sidebar:
    st.header("Configuration")
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        type="password",
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        value=os.environ.get("TAVILY_API_KEY", ""),
        type="password",
    )
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.files = {}
        st.session_state.report_order = []
        st.rerun()

# ---------------------------------------------------------------------------
# Build & Cache the Deep Research Agent
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    current_date = datetime.now().strftime("%Y-%m-%d")

    RESEARCH_WORKFLOW_INSTRUCTIONS = f"""You are a thorough Deep Research Orchestrator. Today's date is {current_date}.
Use your todo list tool to plan multi-step research, delegate focused sub-topics to your research-agent subagent, and save comprehensive findings to files when helpful."""

    RESEARCHER_INSTRUCTIONS = f"""You are a focused Research Sub-Agent. Today's date is {current_date}.
Conduct targeted web searches using tavily_search and return well-cited, factual findings."""

    research_sub_agent = {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent. Give one topic at a time.",
        "system_prompt": RESEARCHER_INSTRUCTIONS,
        "tools": [tavily_search],
    }

    model = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        temperature=0.1,
    )

    return create_deep_agent(
        model=model,
        tools=[tavily_search],
        subagents=[research_sub_agent],
        middleware=[TodoListMiddleware()],
        system_prompt=RESEARCH_WORKFLOW_INSTRUCTIONS,
    )


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files" not in st.session_state:
    st.session_state.files = {}
if "report_order" not in st.session_state:
    st.session_state.report_order = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("todos"):
            with st.expander("📋 Completed Research Plan", expanded=False):
                st.markdown(msg["todos"])
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat Input & Live Middleware Streaming
# ---------------------------------------------------------------------------
if prompt := st.chat_input("What would you like me to research?"):
    if not os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        st.error("Please enter both your OpenRouter and Tavily API keys in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = get_agent()

    with st.chat_message("assistant"):
        # Live collapsible container for TodoListMiddleware & Tool Calls
        status_box = st.status("🧠 Planning & executing research...", expanded=True)
        todo_container = status_box.empty()
        activity_container = status_box.empty()
        answer_container = st.empty()

        latest_todos_md = "_Waiting for TodoListMiddleware plan..._"
        todo_container.markdown(latest_todos_md)
        activity_log = []
        final_response = ""

        # Convert session history into LangChain messages
        lc_messages = [
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        # Stream both state snapshots ("values") and token chunks ("messages")
        for mode, chunk in agent.stream(
            {"messages": lc_messages},
            stream_mode=["values", "messages"],
        ):
            # 1. Capture live TodoListMiddleware state & virtual files
            if mode == "values":
                if "todos" in chunk and chunk["todos"]:
                    lines = ["**📋 TodoListMiddleware Plan:**"]
                    for t in chunk["todos"]:
                        status = t.get("status", "pending")
                        icon = (
                            "✅"
                            if status == "completed"
                            else "⏳"
                            if status == "in_progress"
                            else "⬜"
                        )
                        lines.append(f"{icon} `{status.upper()}` — {t.get('content', '')}")
                    latest_todos_md = "\n\n".join(lines)
                    todo_container.markdown(latest_todos_md)

                if "files" in chunk and chunk["files"]:
                    for filename in chunk["files"]:
                        if filename not in st.session_state.report_order:
                            st.session_state.report_order.append(filename)
                    st.session_state.files.update(chunk["files"])

            # 2. Capture tool calls (tavily_search / subagents) and stream final text
            elif mode == "messages":
                msg, metadata = chunk

                # Log tool invocations
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "tool")
                        if name != "write_todos":
                            activity_log.append(f"🔧 Calling **`{name}`**")
                            activity_container.markdown("\n\n".join(activity_log[-6:]))

                # Stream assistant text tokens from the main model node
                if (
                    isinstance(msg, AIMessage)
                    and msg.content
                    and not getattr(msg, "tool_calls", None)
                    and metadata.get("langgraph_node") == "model"
                ):
                    final_response += msg.content
                    answer_container.markdown(final_response + "▌")

        status_box.update(label="✅ Research complete", state="complete", expanded=False)
        answer_container.markdown(final_response)

        # Save assistant turn to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_response,
                "todos": latest_todos_md,
            }
        )

# ---------------------------------------------------------------------------
# Display the Most Recent Markdown Report Saved by the Deep Agent
# ---------------------------------------------------------------------------
if st.session_state.files:
    assets = [
        name
        for name in st.session_state.report_order
        if name in st.session_state.files
    ]
    if assets:
        assets.sort(
            key=lambda name: (
                getattr(st.session_state.files[name], "updated_at", None)
                or datetime.min
            ),
            reverse=True,
        )
        latest_name = assets[0]
        content = st.session_state.files[latest_name].content
        with st.sidebar:
            st.divider()
            st.subheader("📁 Latest Research Report")
            file_text = (
                content
                if isinstance(content, str)
                else content.decode("utf-8", errors="replace")
            )
            st.download_button(
                label=f"Download {Path(latest_name).name}",
                data=file_text,
                file_name=Path(latest_name).name,
                mime="text/markdown",
            )
