from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph import START,StateGraph,END
import os
import sqlite3
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
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
conn = sqlite3.connect(database= 'chatbot.db', check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():

    all_thread = set()

    for i in checkpointer.list(None):
        all_thread.add(i.config["configurable"]['thread_id'])

    return list(all_thread)



# # test
# user_input = "what is my name"
# CONFIG = {"configurable" : {'thread_id' : '1'}}

# response = chatbot.invoke(
#                       {"messages": [HumanMessage(content = user_input)]}, 
#                       config=CONFIG
#                  )

# print(response)