"""
rag.py
------
Builds and runs the Retrieval-Augmented Generation (RAG) flow.

Adds:
- Preview of FAISS (semantic) Top-5
- Preview of BM25 (lexical) Top-5
- Answers using BM25 context (great for names/IDs) to diagnose retrieval
- Includes human-friendly filename in previews and sources

You can later switch back to FAISS or do a hybrid merge.
"""

from operator import itemgetter
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from store import get_retriever, get_bm25_retriever
from config import LLM_MODEL, TOP_K


def _format_pagestr(d):
    pr = d.metadata.get("page_range")
    if isinstance(pr, tuple):
        return f"(páginas {pr[0]+1}-{pr[1]+1})"
    p = d.metadata.get("page")
    return f"(página {int(p)+1})" if isinstance(p, int) else ""


def _filename_of(d) -> str:
    fn = d.metadata.get("filename")
    if fn:
        return fn
    src = d.metadata.get("source", "")
    return Path(src).name if src else "(desconocido)"


def _print_top_chunks(question: str, k: int = 5):
    """Preview FAISS (semantic) top-k chunks for the question."""
    retriever = get_retriever()
    docs = retriever.invoke(question)
    n = min(k, len(docs))
    print(f"🔍 FAISS Top-{n} retrieved chunks:")
    for i, d in enumerate(docs[:n], 1):
        src = d.metadata.get("source", "")
        fname = _filename_of(d)
        pg = _format_pagestr(d)
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        print(f"\n[{i}] {fname}  |  {src} {pg}\n{snippet}")


def _print_top_chunks_bm25(question: str, k: int = 5):
    """Preview BM25 (lexical) top-k chunks for the question."""
    retriever = get_bm25_retriever(k=k)
    docs = retriever.invoke(question)
    n = min(k, len(docs))
    print(f"\n🔎 BM25 Top-{n} retrieved chunks:")
    for i, d in enumerate(docs[:n], 1):
        src = d.metadata.get("source", "")
        fname = _filename_of(d)
        pg = _format_pagestr(d)
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        print(f"\n[{i}] {fname}  |  {src} {pg}\n{snippet}")


def format_docs(docs):
    """Include filename + source + page to help the model ground its answer."""
    parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        fname = _filename_of(d)
        pg = _format_pagestr(d)
        parts.append(
            f"ARCHIVO: {fname}\nRUTA: {src} {pg}\nCONTENIDO:\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_rag_chain():
    """Simple prompt->LLM chain; context is precomputed in ask()."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """Eres un asistente jurídico. **Responde SOLO con la información del contexto**.
Si la respuesta no está explícitamente en el contexto, di: "No se encuentra en los documentos."
Al final, muestra una sección "Fuentes" con archivo y páginas usadas.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    )
    return prompt | llm | StrOutputParser()


def ask(question: str) -> str:
    # Preview both retrieval modes for debugging
    _print_top_chunks(question, k=5)        # FAISS
    _print_top_chunks_bm25(question, k=5)   # BM25

    # 👉 For this diagnostic phase, answer using BM25 (lexical) results.
    bm25 = get_bm25_retriever(k=TOP_K)
    docs = bm25.invoke(question)

    chain = build_rag_chain()
    print(f"\n💬 Question: {question}\n")
    answer = chain.invoke({"context": format_docs(docs[:TOP_K]), "question": question})
    return answer


if __name__ == "__main__":
    # query = "¿Qué pasa si el comprador incumple el pago?"
    # query = "¿Quién compró el departamento 501?"
    query = "¿Cómo se apellida el cliente Gilberto?"
    result = ask(query)
    print("\n🧾 Respuesta:")
    print(result)
