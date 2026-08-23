# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import langsmith.client
import langsmith.run_trees as rt

# Must run before the LangSmith client is built - it reads the key from the env
load_dotenv()  # expects OPENROUTER_API_KEY and GROQ_API_KEY in .env

os.environ["LANGCHAIN_PROJECT"] = "ragchatbot"

# Fix 1: Gives background uploads more time
langsmith.client._TRACING_SEND_TIMEOUT = (30, 60)

# Fix 2: Gives the /info check 30 to 60 seconds so it does not timeout
rt.get_cached_client(timeout_ms=(30_000, 60_000))

PDF_PATH = r"islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
# The embedding model caps out at 512 tokens, so measure chunks in TOKENS, not
# characters. Dense textbook math tokenises badly (~0.8 tokens/char), which is how
# 1000-character chunks turned into 775-token requests and got rejected.
# 350 leaves headroom because tiktoken is only an approximation of Liquid's tokeniser.
MAX_TOKENS = 350
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=MAX_TOKENS,
    chunk_overlap=50,
)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = OpenAIEmbeddings(
    model="liquid/lfm-2.5-embedding-350m:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    # Required for non-OpenAI providers: without this LangChain pre-tokenises with
    # tiktoken and posts token IDs, which OpenRouter rejects with 400 "Invalid input".
    check_embedding_ctx_length=False,
    chunk_size=64,   # texts per request
    max_retries=6,   # the :free tier rate-limits aggressively
)
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

# Takes dict input {"question": "..."} and pulls the field out before the retriever,
# which needs a bare query string. RunnableParallel broadcasts the same input to
# both branches, so each one does its own extraction.
parallel = RunnableParallel({
    "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
    "question": RunnableLambda(lambda x: x["question"]),
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")

config = {
    "run_name": "ragchatbot",
    "tags": ["rag", "pdf_qa"],
    "metadata": {"embedding_model": "liquid/lfm-2.5-embedding-350m:free",
                 "llm": "groq_gpt_oss_120b"},
}

ans = chain.invoke({"question": q.strip()}, config=config)
print("\nA:", ans)
