"""
store.py
--------
Utility functions to load and access the FAISS vector store for retrieval.

- Loads embeddings model (must match the one used in ingestion)
- Loads FAISS index from data/index/
- Provides both FAISS (semantic) and BM25 (lexical) retrievers
"""
import unicodedata
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings

from config import INDEX_DIR, EMBEDDING_MODEL, TOP_K


def _normalize_for_bm25(text: str) -> str:
    if not text:
        return ""
    # Accent fold + lowercase
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    # Keep letters+digits (crucial: DO NOT drop numbers like 501)
    tokens = re.findall(r"[0-9a-zñ]+", t)
    return " ".join(tokens)

def load_embeddings():
    """Return an Ollama embeddings instance using the configured model."""
    print(f"🧠 Loading embeddings model: {EMBEDDING_MODEL}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def load_vectorstore():
    """Load the local FAISS index from disk (with embeddings)."""
    index_path = Path(INDEX_DIR)
    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index found at {index_path}. Run ingest.py first.")
    embeddings = load_embeddings()
    print(f"📂 Loading FAISS index from {index_path}")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def get_retriever(k: int = TOP_K):
    """
    FAISS semantic retriever (MMR) with higher recall.
    Returns top-k results after considering a larger candidate pool.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,            # results returned
            "fetch_k": 50,     # candidates considered (increase recall)
            "lambda_mult": 0.5 # balance diversity vs relevance
        },
    )
    print(f"🔎 Retriever ready (type=mmr, top_k={k}, fetch_k=50)")
    return retriever



def get_bm25_retriever(k: int = TOP_K):
    vs = FAISS.load_local(
        str(Path(INDEX_DIR)),
        embeddings=None,
        allow_dangerous_deserialization=True,
    )
    docs = list(vs.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(
        docs,
        preprocess_func=_normalize_for_bm25,   # <-- important
    )
    bm25.k = max(k, 50)  # pull more candidates; we’ll cut to TOP_K later
    print(f"🔎 BM25 retriever ready (top_k={bm25.k}, normalized)")
    return bm25


if __name__ == "__main__":
    # Quick test to verify everything loads correctly
    print(">> FAISS (MMR) test")
    faiss_ret = get_retriever()
    faiss_results = faiss_ret.invoke("¿Qué pasa si el comprador incumple el pago?")
    print(f"Retrieved {len(faiss_results)} chunks (FAISS).")

    print("\n>> BM25 test")
    bm25_ret = get_bm25_retriever()
    bm25_results = bm25_ret.invoke("¿Cómo se apellida el cliente Gilberto?")
    print(f"Retrieved {len(bm25_results)} chunks (BM25).")
