from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
import requests
from langchain_community.tools import DuckDuckGoSearchRun
# LangChain 1.x moved the classic ReAct agent + AgentExecutor into langchain_classic.
# langchain.agents.create_react_agent no longer exists and langchain.hub is gone too.
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client as LangSmithClient
from dotenv import load_dotenv
import langsmith.client
import langsmith.run_trees as rt

# Must run before the LangSmith client is built - it reads the key from the env
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "ragchatbot"

# Fix 1: Gives background uploads more time
langsmith.client._TRACING_SEND_TIMEOUT = (30, 60)

# Fix 2: Gives the /info check 30 to 60 seconds so it does not timeout
# hub.pull() below goes to the same LangSmith API, so this matters here too
rt.get_cached_client(timeout_ms=(30_000, 60_000))

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'http://api.weatherstack.com/current?access_key={os.getenv("WEATHERSTACK_API_KEY")}&query={city}'
  response = requests.get(url)

  return response.json()

# NOT gpt-oss-120b here. That model emits native tool calls instead of plain text,
# which the classic text-based ReAct scaffold cannot use: AgentExecutor streams with
# no tools bound, so Groq rejects it with "Tool choice is none, but model called a
# tool". Qwen returns ordinary text, which is what Thought/Action parsing needs.
llm = ChatOpenAI(
    model="qwen/qwen3.6-27b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# Step 2: Pull the ReAct prompt from LangChain Hub
# Newer langsmith blocks public prompt pulls by default, because a pulled prompt can
# carry serialised LangChain objects. hwchase17/react is the canonical ReAct prompt,
# so we opt in explicitly. hub.pull() has no way to pass that flag through, so go
# straight to the LangSmith client instead.
prompt = LangSmithClient().pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True,
)

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of Rangamati, Bangladesh"})
print(response)

print(response['output'])