from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite, os
import requests
import asyncio
import threading
import langsmith.client
import langsmith.run_trees as rt


load_dotenv()

langsmith.client._TRACING_SEND_TIMEOUT = (30, 60)
rt.get_cached_client(timeout_ms=(30_000, 60_000))

api_key = os.getenv("GROQ_API_KEY")
MY_API_KEY = os.getenv("FASTMCP_KEY")
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'http://api.weatherstack.com/current?access_key={os.getenv("WEATHERSTACK_API_KEY")}&query={city}'
  response = requests.get(url)

  return response.json()


EXPENSE_URL = "https://voiceless-chocolate-pike.fastmcp.app/mcp"
SERVERS = {
    'math': {
        'transport': 'stdio',
        'command': "G:\\Telegram Desktop\\Mcp_Project\\venv\\Scripts\\uv.exe",
        'args': [
            'run', '--active', 'fastmcp', 'run', "G:\\Telegram Desktop\\Mcp_Project\\src\\mcp_project\\calculator.py"
        ]
    },
    "expense": {
        "transport": "streamable_http", 
        "url": EXPENSE_URL,
        "headers": {
            "Authorization": f"Bearer {MY_API_KEY}"
        }
    },
}

client = MultiServerMCPClient(SERVERS)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def load_mcp_tools() -> list[BaseTool]:
    """Fetch tools from each MCP server independently.

    Per-server on purpose: one unreachable server (say a 401 from the hosted
    expense server) then costs you just that server's tools, instead of
    taking every tool down with it.

    Must be awaited, never run through run_async() -- build_graph() already
    runs *on* _ASYNC_LOOP, and run_async() blocks that same loop thread on
    its own result, which deadlocks.
    """
    tools: list[BaseTool] = []
    for server_name in SERVERS:
        try:
            tools.extend(await client.get_tools(server_name=server_name))
        except Exception as exc:
            print(f"[mcp] server {server_name!r} unavailable, skipping: {exc}")
    return tools


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

async def build_graph():

    
    mcp_tools = await load_mcp_tools()

    tools = [search_tool, get_weather_data, *mcp_tools]
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    
    # Map out tools by their names for our custom node to look up
    tool_map = {t.name: t for t in tools}


    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    # FIX: Replaced standard ToolNode with a clean string conversion interceptor
    async def custom_tool_node(state: ChatState):
        last_message = state["messages"][-1]
        tool_outputs = []
        
        # Loop through every tool call requested by the LLM
        for tool_call in last_message.tool_calls:
            tool_obj = tool_map[tool_call["name"]]
            
            # Execute the tool call
            raw_output = await tool_obj.ainvoke(tool_call["args"])
            
            # Convert output safely to a string payload so Groq doesn't crash
            string_output = str(raw_output)
            if not string_output or string_output.strip() == "":
                string_output = "Tool executed successfully with no data returned."
                
            tool_outputs.append(
                ToolMessage(
                    content=string_output, 
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )
            
        return {"messages": tool_outputs}

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", custom_tool_node) # Using our custom safe node

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    # The checkpointer is what makes get_state / conversation history /
    # retrieve_all_threads() work at all -- compiling without it leaves the
    # sidebar permanently empty.
    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())