import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from tools.search import tavily_search

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

    with st.expander("Langfuse Observability", expanded=False):
        lf_public_key = st.text_input(
            "Langfuse Public Key",
            value=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            type="password",
        )
        lf_secret_key = st.text_input(
            "Langfuse Secret Key",
            value=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            type="password",
        )
        lf_base_url = st.text_input(
            "Langfuse Base URL",
            value=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
        if lf_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = lf_public_key
        if lf_secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = lf_secret_key
        if lf_base_url:
            os.environ["LANGFUSE_BASE_URL"] = lf_base_url

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.files = {}
        st.session_state.report_order = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ---------------------------------------------------------------------------
# Build & Cache the Deep Research Agent
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    current_date = datetime.now().strftime("%Y-%m-%d")

    RESEARCH_WORKFLOW_INSTRUCTIONS = f"""You are a thorough Deep Research Orchestrator. Today's date is {current_date}.
    # Research Workflow

Follow this workflow for all research requests:

1. **Plan**: Create a todo list with write_todos to break down the research into focused tasks
2. **Save the request**: Use write_file() to save the user's research question to `/research_request.md`
3. **Research**: Delegate research tasks to sub-agents using the task() tool - ALWAYS use sub-agents for research, never conduct research yourself
4. **Synthesize**: Review all sub-agent findings and consolidate citations (each unique URL gets one number across all findings)
5. **Write Report**: Write a comprehensive final report to `/final_report.md` (see Report Writing Guidelines below)
6. **Verify**: Read `/research_request.md` and confirm you've addressed all aspects with proper citations and structure

## Research Planning Guidelines
- Batch similar research tasks into a single TODO to minimize overhead
- For simple fact-finding questions, use 1 sub-agent
- For comparisons or multi-faceted topics, delegate to multiple parallel sub-agents
- Each sub-agent should research one specific aspect and return findings

## Report Writing Guidelines

When writing the final report to `/final_report.md`, follow these structure patterns:

**For comparisons:**
1. Introduction
2. Overview of topic A
3. Overview of topic B
4. Detailed comparison
5. Conclusion

**For lists/rankings:**
Simply list items with details - no introduction needed:
1. Item 1 with explanation
2. Item 2 with explanation
3. Item 3 with explanation

**For summaries/overviews:**
1. Overview of topic
2. Key concept 1
3. Key concept 2
4. Key concept 3
5. Conclusion

**General guidelines:**
- Use clear section headings (## for sections, ### for subsections)
- Write in paragraph form by default - be text-heavy, not just bullet points
- Do NOT use self-referential language ("I found...", "I researched...")
- Write as a professional report without meta-commentary
- Each section should be comprehensive and detailed
- Use bullet points only when listing is more appropriate than prose

**Citation format:**
- Cite sources inline using [1], [2], [3] format
- Assign each unique URL a single citation number across ALL sub-agent findings
- End report with ### Sources section listing each numbered source
- Number sources sequentially without gaps (1,2,3,4...)
- Format: [1] Source Title: URL (each on separate line for proper list rendering)
- Example:

 Some important finding [1]. Another key insight [2].

 ### Sources
 [1] AI Research Paper: https://example.com/paper
 [2] Industry Analysis: https://example.com/analysis

# Sub-Agent Research Coordination

Your role is to coordinate research by delegating tasks from your TODO list to specialized research sub-agents.

## Delegation Strategy

**DEFAULT: Start with 1 sub-agent** for most queries:
- "What is quantum computing?" -> 1 sub-agent (general overview)
- "List the top 10 coffee shops in San Francisco" -> 1 sub-agent
- "Summarize the history of the internet" -> 1 sub-agent
- "Research context engineering for AI agents" -> 1 sub-agent (covers all aspects)

**ONLY parallelize when the query EXPLICITLY requires comparison or has clearly independent aspects:**

**Explicit comparisons** -> 1 sub-agent per element:
- "Compare OpenAI vs Anthropic vs DeepMind AI safety approaches" -> 3 parallel sub-agents
- "Compare Python vs JavaScript for web development" -> 2 parallel sub-agents

**Clearly separated aspects** -> 1 sub-agent per aspect (use sparingly):
- "Research renewable energy adoption in Europe, Asia, and North America" -> 3 parallel sub-agents (geographic separation)
- Only use this pattern when aspects cannot be covered efficiently by a single comprehensive search

## Key Principles
- **Bias towards single sub-agent**: One comprehensive research task is more token-efficient than multiple narrow ones
- **Avoid premature decomposition**: Don't break "research X" into "research X overview", "research X techniques", "research X applications" - just use 1 sub-agent for all of X
- **Parallelize only for clear comparisons**: Use multiple sub-agents when comparing distinct entities or geographically separated data

## Parallel Execution Limits
- Use at most 3 parallel sub-agents per iteration
- Make multiple task() calls in a single response to enable parallel execution
- Each sub-agent returns findings independently

## Research Limits
- Stop after 3 delegation rounds if you haven't found adequate sources
- Stop when you have sufficient information to answer comprehensively
- Bias towards focused research over exhaustive exploration
"""

    RESEARCHER_INSTRUCTIONS = f"""You are a research assistant conducting research on the user's input topic. For context, today's date is {current_date}.

Your job is to use tools to gather information about the user's input topic.
You can use the tavily_search tool to find resources that can help answer the research question.
You can call tools in series or in parallel, your research is conducted in a tool-calling loop.

Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection

**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information

After each search, assess results before continuing: What key information did I find? What's missing? Do I have enough to answer? Should I search more or provide my answer?

When providing your findings back to the orchestrator:

1. **Structure your response**: Organize findings with clear headings and detailed explanations
2. **Cite sources inline**: Use [1], [2], [3] format when referencing information from your searches
3. **Include Sources section**: End with ### Sources listing each numbered source with title and URL

The orchestrator will consolidate citations from all sub-agents into the final report.
"""

    research_sub_agent = {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent. Give one topic at a time.",
        "system_prompt": RESEARCHER_INSTRUCTIONS,
        "tools": [tavily_search],
    }

    model = ChatOpenAI(
        model="z-ai/glm-5.3-flash",
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
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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

    # Configure Langfuse callback & session metadata if credentials are available
    callbacks = []
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        callbacks.append(CallbackHandler())

    run_config = {
        "callbacks": callbacks,
        "run_name": "deep-research-agent",
        "metadata": {
            "langfuse_session_id": st.session_state.session_id,
            "langfuse_tags": ["streamlit", "deep-research"],
        },
    }

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
        latest_todos = []

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
            config=run_config,
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

                    for t in chunk["todos"]:
                        content = t.get("content", "")
                        status = t.get("status", "pending")
                        prev = next((x for x in latest_todos if x.get("content") == content), None)
                        if prev is None:
                            activity_log.append(f"📝 **Added**: {content}")
                        elif prev.get("status") != status:
                            if status == "in_progress":
                                activity_log.append(f"▶️ **Started**: {content}")
                            elif status == "completed":
                                activity_log.append(f"✅ **Completed**: {content}")
                    latest_todos = chunk["todos"]
                    activity_container.markdown("\n\n".join(activity_log[-8:]))

                if "files" in chunk and chunk["files"]:
                    for filename in chunk["files"]:
                        if filename not in st.session_state.report_order:
                            st.session_state.report_order.append(filename)
                    st.session_state.files.update(chunk["files"])

            # 2. Capture tool calls (tavily_search / subagents) and stream final text
            elif mode == "messages":
                msg, metadata = chunk

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "tool")
                        args = tc.get("args", {})
                        if name == "task":
                            description = args.get("description", "research task")
                            activity_log.append(f"🤖 **Sub-agent**: {description}")
                        elif name == "tavily_search":
                            query = args.get("query", "search")
                            activity_log.append(
                                f"🔍 **Search**: {query[:80]}{'...' if len(query) > 80 else ''}"
                            )
                        elif name == "write_file":
                            path = args.get("path", "file")
                            activity_log.append(f"📝 **Write**: {path}")
                        elif name != "write_todos":
                            activity_log.append(f"🔧 Calling **`{name}`**")
                        activity_container.markdown("\n\n".join(activity_log[-8:]))

                if (
                    isinstance(msg, AIMessage)
                    and msg.content
                    and not getattr(msg, "tool_calls", None)
                    and metadata.get("langgraph_node") == "model"
                ):
                    final_response += msg.content
                    answer_container.markdown(final_response + "▌")

                if isinstance(msg, ToolMessage) and metadata.get("langgraph_node") == "research-agent":
                    content_preview = (
                        msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    )
                    activity_log.append(f"📄 **Sub-agent result**: {content_preview}")
                    activity_container.markdown("\n\n".join(activity_log[-8:]))

        # Flush Langfuse traces immediately at the end of the turn
        if callbacks:
            get_client().flush()

        # Add completion summary
        completed_count = sum(1 for t in latest_todos if t.get("status") == "completed")
        total_count = len(latest_todos)
        in_progress_count = sum(1 for t in latest_todos if t.get("status") == "in_progress")
        pending_count = sum(1 for t in latest_todos if t.get("status") == "pending")
        status_box.update(
            label=f"✅ Research complete — ✅{completed_count} ⏳{in_progress_count} ⬜{pending_count} (of {total_count})",
            state="complete",
            expanded=False,
        )
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
# Display & Download Generated Markdown Reports
# ---------------------------------------------------------------------------
def _extract_file_text(file_obj) -> str:
    content = (
        file_obj["content"]
        if isinstance(file_obj, dict)
        else getattr(file_obj, "content", "")
    )
    if isinstance(content, list):
        return "\n".join(str(line) for line in content)
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _get_modified_ts(file_obj) -> str:
    if isinstance(file_obj, dict):
        return str(file_obj.get("modified_at") or file_obj.get("updated_at") or "")
    return str(getattr(file_obj, "modified_at", "") or getattr(file_obj, "updated_at", ""))


if st.session_state.files:
    assets = [
        name
        for name in st.session_state.report_order
        if name in st.session_state.files
    ]
    if assets:
        # Sort newest first using modified_at, then pin final_report.md to the top
        assets.sort(
            key=lambda name: _get_modified_ts(st.session_state.files[name]),
            reverse=True,
        )
        final_reports = [n for n in assets if Path(n).name == "final_report.md"]
        if final_reports:
            assets.remove(final_reports[0])
            assets.insert(0, final_reports[0])

        with st.sidebar:
            st.divider()
            st.subheader("📁 Research Files")
            selected_name = (
                assets[0]
                if len(assets) == 1
                else st.selectbox(
                    "Select file to download",
                    options=assets,
                    format_func=lambda n: Path(n).name,
                )
            )
            file_text = _extract_file_text(st.session_state.files[selected_name])
            st.download_button(
                label=f"Download {Path(selected_name).name}",
                data=file_text,
                file_name=Path(selected_name).name,
                mime="text/markdown",
            )
