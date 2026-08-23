from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import langsmith.client
import langsmith.run_trees as rt

# Must run before the client is built - it reads the LangSmith key from the env
load_dotenv()

os.environ["LANGCHAIN_PROJECT"]= 'Sequential LLM app'

# Fix 1: Gives background uploads more time
langsmith.client._TRACING_SEND_TIMEOUT = (30, 60)

# Fix 2: NEW! Gives the /info check 30 to 60 seconds so it does not timeout
rt.get_cached_client(timeout_ms=(30_000, 60_000))

api_key = os.getenv("GROQ_API_KEY")

model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

prompt1 = PromptTemplate(
    template='Generate a short report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 1 pointer summary from the following text \n {text}',
    input_variables=['text']
)



parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'run_name' : 'sequential chain take 1',
    'tags' : ["llm_app"],
    'metadata' : {"tags": "testing sequential chain", "model" : "groq_model"}
}

result = chain.invoke({'topic': 'Unemployment in Bangladesh'}, config = config)

print(result)
