import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_mcp_backend import (
    build_graph,
    retrieve_all_threads,
    run_async,
    submit_async_task,
)

# =========================== Graph ===========================
# build_graph() is a coroutine, so it has to be driven on the backend's event
# loop. cache_resource keeps it to one build for the whole server -- without
# it every Streamlit rerun would re-handshake the MCP servers and respawn the
# math server's stdio subprocess.
@st.cache_resource(show_spinner="Connecting to MCP servers …")
def get_chatbot():
    return run_async(build_graph())


chatbot = get_chatbot()


# =========================== Utilities ===========================
def generate_thread_id():
    # str, not UUID: the SQLite checkpointer stores thread ids as text, so
    # retrieve_all_threads() hands back strings. Keeping UUID objects here
    # would make the same thread look like two different ones.
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    # AsyncSqliteSaver implements only the async API -- the sync get_state()
    # raises NotImplementedError against it.
    state = run_async(
        chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
    )
    return state.values.get("messages", [])


def message_decoration(messages):
    """Turn stored graph messages into renderable chat rows.

    Tool traffic and the empty assistant turns that only carry tool_calls are
    dropped: they are graph plumbing, not conversation.
    """
    rows = []
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not str(content).strip():
            continue
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        rows.append({"role": role, "content": content})
    return rows


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for pos, thread_id in enumerate(st.session_state["chat_threads"][::-1]):
    label = "Current chat" if thread_id == st.session_state["thread_id"] else f"Chat {pos}"
    # key= keeps the buttons distinct; labels repeat, thread ids do not.
    if st.sidebar.button(label, key=f"thread-{thread_id}"):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = message_decoration(
            load_conversation(thread_id)
        )

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            # The graph is async and lives on the backend loop; Streamlit's
            # thread is sync. A queue bridges the two.
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("__error__", exc))
                finally:
                    event_queue.put(None)

            future = submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, payload = item

                # Sentinel is checked by identity-ish comparison on a plain
                # str, so it can never collide with a message object.
                if isinstance(message_chunk, str) and message_chunk == "__error__":
                    raise payload

                # Lazily create & update the SAME status container per tool
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant text
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    if content:
                        yield content

            # Surface anything that killed the task without reaching the queue.
            future.result()

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as exc:
            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="❌ Tool failed", state="error", expanded=True
                )
            st.error(f"{type(exc).__name__}: {exc}")
            ai_message = None
        else:
            # Finalize only if a tool was actually used
            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

    # Save assistant message
    if ai_message:
        if isinstance(ai_message, list):
            ai_message = "".join(str(part) for part in ai_message)
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
