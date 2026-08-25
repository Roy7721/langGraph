from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage # Added ToolMessage here
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition


import os, asyncio
import langsmith.client
import langsmith.run_trees as rt
from langchain_mcp_adapters.client import MultiServerMCPClient

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

async def build_graph():

    tools = await client.get_tools()
    
    # Map out tools by their names for our custom node to look up
    tool_map = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

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

    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Give me all my expenses , then add 5 with 5, the multiply with 20, then subtract 1 and divide with 5 and give the result")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())
