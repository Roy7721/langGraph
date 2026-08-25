from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph import START,StateGraph,END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import os, asyncio
import sqlite3, requests
import langsmith.client
import langsmith.run_trees as rt


load_dotenv()

langsmith.client._TRACING_SEND_TIMEOUT = (30, 60)

# Fix 2: Gives the /info check 30 to 60 seconds so it does not timeout
rt.get_cached_client(timeout_ms=(30_000, 60_000))

api_key = os.getenv("GROQ_API_KEY")

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)



@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

tools = [calculator]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():

    async def chat_node(state: ChatState):

        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {"messages" : [response]}

    tool_mode = ToolNode(tools=tools)

    graph = StateGraph(ChatState)

    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_mode)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node",tools_condition)

    graph.add_edge('tools', 'chat_node')    

    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot  = build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Find the modulus of 132354 and 23 and give answer like a cricket commentator.")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())





