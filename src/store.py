"""
store.py
--------
Utility functions to load and access the FAISS vector store for retrieval.

- Loads embeddings model (must match the one used in ingestion)
- Loads FAISS index from data/index/
- Provides a retriever interface for the RAG chain
"""

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import INDEX_DIR, EMBEDDING_MODEL, TOP_K
from pathlib import Path


def load_embeddings():
    """Return an Ollama embeddings instance using the configured model."""
    print(f"🧠 Loading embeddings model: {EMBEDDING_MODEL}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def load_vectorstore():
    """Load the local FAISS index from disk."""
    index_path = Path(INDEX_DIR)
    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index found at {index_path}. Run ingest.py first.")
    embeddings = load_embeddings()
    print(f"📂 Loading FAISS index from {index_path}")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def get_retriever(k: int = TOP_K):
    """
    Returns a retriever instance configured with the global TOP_K setting.
    This retriever is what the RAG chain will call to find relevant chunks.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"🔎 Retriever ready (top_k={k})")
    return retriever


if __name__ == "__main__":
    # Quick test to verify everything loads correctly
    retriever = get_retriever()
    results = retriever.invoke("¿Qué pasa si el comprador incumple el pago?")
    print(f"\nRetrieved {len(results)} chunks:")
    for doc in results[:2]:
        print(f"- {doc.metadata.get('source', '')}: {doc.page_content[:120]}...\n")
