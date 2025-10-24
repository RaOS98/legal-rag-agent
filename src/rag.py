"""
rag.py
------
Builds and runs the Retrieval-Augmented Generation (RAG) chain.

This connects:
- Retriever (FAISS + OllamaEmbeddings)
- LLM (ChatOllama)
- Prompt (injects retrieved context)
using the LangChain Expression Language (LCEL).
"""

from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from store import get_retriever
from config import LLM_MODEL

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    retriever = get_retriever()
    llm = ChatOllama(model=LLM_MODEL)

    prompt = ChatPromptTemplate.from_template(
        """Eres un asistente especializado en derecho inmobiliario peruano.
Usa el siguiente contexto para responder de forma breve, precisa y en español.

Contexto:
{context}

Pregunta: {question}

Respuesta concisa:"""
    )

    rag_chain = (
        {
            # ✅ send ONLY the question string to the retriever
            "context": itemgetter("question") | retriever | format_docs,
            # keep the original question for the prompt
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"🤖 RAG chain ready (LLM={LLM_MODEL})")
    return rag_chain

def ask(question: str) -> str:
    rag_chain = build_rag_chain()
    print(f"💬 Question: {question}\n")
    return rag_chain.invoke({"question": question})

if __name__ == "__main__":
    # query = "¿Qué pasa si el comprador incumple el pago?"
    query = "¿Cuando firmo la minuta el cliente Gilberto Coronado?"
    result = ask(query)
    print("\n🧾 Respuesta:")
    print(result)
