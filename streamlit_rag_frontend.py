import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    build_graph,
    ingest_pdf,
    retrieve_all_threads,
    run_async,
    submit_async_task,
    thread_document_metadata,
)

# =========================== Graph ===========================
# build_graph() is async, so it has to be driven on the backend's event loop.
# cache_resource keeps it to a single build for the whole server -- otherwise
# every Streamlit rerun would re-handshake the MCP servers and respawn the
# math server's stdio subprocess.
@st.cache_resource(show_spinner="Starting chatbot …")
def get_chatbot():
    return run_async(build_graph())


chatbot = get_chatbot()


# =========================== Utilities ===========================
def generate_thread_id():
    # str, not UUID: the SQLite checkpointer stores thread ids as text and
    # retrieve_all_threads() hands them back as strings, so a UUID object
    # would never compare equal to a stored thread.
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
    # AsyncSqliteSaver implements only the async API -- sync get_state()
    # raises NotImplementedError against it.
    state = run_async(
        chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
    )
    return state.values.get("messages", [])


def _flatten(content):
    """Some providers return content blocks instead of a plain string."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


def message_decoration(messages):
    """Turn stored graph messages into renderable chat rows.

    Tool traffic and the empty assistant turns that carry only tool_calls are
    dropped -- they are graph plumbing, not conversation.
    """
    rows = []
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        content = _flatten(message.content)
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

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# The backend is the source of truth for what is indexed; session_state only
# remembers which filenames this browser session already uploaded.
doc_meta = thread_document_metadata(thread_key)
if doc_meta:
    st.sidebar.success(
        f"Using `{doc_meta.get('filename')}` "
        f"({doc_meta.get('chunks')} chunks from {doc_meta.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
            except Exception as exc:
                status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                st.sidebar.error(f"{type(exc).__name__}: {exc}")
            else:
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
                st.rerun()

st.sidebar.subheader("Past conversations")
threads = st.session_state["chat_threads"][::-1]
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for pos, thread_id in enumerate(threads):
        label = "Current chat" if thread_id == thread_key else f"Chat {pos}"
        if st.sidebar.button(label, key=f"side-thread-{thread_id}"):
            # Switch immediately, before the chat area renders, so the page
            # never paints one thread's history under another thread's id.
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = message_decoration(
                load_conversation(thread_id)
            )
            st.session_state["ingested_docs"].setdefault(str(thread_id), {})
            st.rerun()

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        # rag_tool's thread_id is injected from here by the backend's tool
        # node, so this value is what scopes retrieval to this chat.
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            # The graph is async and lives on the backend loop; Streamlit's
            # script thread is sync. A queue bridges the two.
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, _ in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, None))
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

                if isinstance(message_chunk, str) and message_chunk == "__error__":
                    raise payload

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

                if isinstance(message_chunk, AIMessage):
                    content = _flatten(message_chunk.content)
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
            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

    if ai_message:
        if isinstance(ai_message, list):
            ai_message = "".join(str(part) for part in ai_message)
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )
