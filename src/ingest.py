"""
ingest.py
---------
Indexing pipeline for the Legal RAG Agent.

Steps:
1. Load PDFs from data/raw/
2. Split into manageable chunks
3. Embed with a local Ollama model
4. Store vectors in FAISS (data/index/)
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import List
from config import RAW_DIR, INDEX_DIR, EMBEDDING_MODEL


def load_pdfs(folder: Path) -> List:
    """Load all PDF files in the given folder into LangChain Document objects."""
    pdf_paths = list(folder.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {folder}")
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())
        print(f"✅ Loaded {path.name}")
    return docs


def split_docs(docs: List) -> List:
    """Split long documents into smaller chunks using the recommended splitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    print(f"✂️  Split into {len(splits)} chunks")
    return splits


def build_embeddings():
    """Create an Ollama embeddings instance."""
    print(f"🧠 Using embeddings model: {EMBEDDING_MODEL}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_index(chunks: List, index_dir: Path):
    """Generate embeddings for chunks and save FAISS index locally."""
    embeddings = build_embeddings()
    print("⚙️  Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir))
    print(f"💾 Index saved to {index_dir}")


def main():
    """Main ingestion routine."""
    print("🚀 Starting ingestion pipeline...")
    docs = load_pdfs(RAW_DIR)
    chunks = split_docs(docs)
    build_index(chunks, INDEX_DIR)
    print("✅ Ingestion complete.")


if __name__ == "__main__":
    main()
