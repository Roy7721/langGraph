# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

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
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "ragchatbot"

# Fix 1: Gives background uploads more time
langsmith.client._TRACING_SEND_TIMEOUT = (30, 60)

# Fix 2: Gives the /info check 30 to 60 seconds so it does not timeout
rt.get_cached_client(timeout_ms=(30_000, 60_000))

PDF_PATH = "islr.pdf"  # change to your file
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# The embedding model caps out at 512 tokens, so chunks are measured in TOKENS, not
# characters. Dense textbook math tokenises badly (~0.9 tokens/char), which is how
# 1000-character chunks turned into 775-token requests and got rejected.
# 350 leaves headroom because tiktoken only approximates Liquid's tokeniser.
EMBED_MODEL = "liquid/lfm-2.5-embedding-350m:free"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50


def make_embeddings(model_name: str) -> OpenAIEmbeddings:
    """Single source of truth - building and loading an index MUST agree here."""
    return OpenAIEmbeddings(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        # Required for non-OpenAI providers: without this LangChain pre-tokenises with
        # tiktoken and posts token IDs, which OpenRouter rejects with 400 "Invalid input".
        check_embedding_ctx_length=False,
        chunk_size=64,   # texts per request
        max_retries=6,   # the :free tier rate-limits aggressively
    )

# ----------------- helpers (traced) -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()  # list[Document]

@traceable(name="split_documents")
def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits, embed_model_name: str):
    emb = make_embeddings(embed_model_name)
    return FAISS.from_documents(splits, emb)

# ----------------- cache key / fingerprint -----------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

# ----------------- explicitly traced load/build runs -----------------
@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path, embed_model_name: str):
    emb = make_embeddings(embed_model_name)
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    docs = load_pdf(pdf_path)  # child
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # child
    vs = build_vectorstore(splits, embed_model_name)  # child
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs

# ----------------- dispatcher (not traced) -----------------
def load_or_build_index(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_model_name: str = EMBED_MODEL,
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ----------------- model, prompt, and pipeline -----------------
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, embed_model_name=EMBED_MODEL, force_rebuild=False):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_model_name: str = EMBED_MODEL,
    force_rebuild: bool = False,
):
    vectorstore = setup_pipeline(pdf_path, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Takes dict input {"question": "..."} and pulls the field out before the retriever,
    # which needs a bare query string. RunnableParallel broadcasts the same input to
    # both branches, so each one does its own extraction.
    parallel = RunnableParallel({
        "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"]),
    })
    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(
        {"question": question},
        config={"run_name": "ragchatbot_v4", "tags": ["qa", "rag", "cached_index"],
                "metadata": {"k": 4, "embedding_model": embed_model_name,
                             "llm": "groq_gpt_oss_120b"}}
    )

# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)
