from pathlib import Path
import unicodedata
import re

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings

from config import INDEX_DIR, EMBEDDING_MODEL

# ---------- helpers ----------
def norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def preview(doc, maxlen=220):
    t = (doc.page_content or "").strip().replace("\n", " ")
    return (t[:maxlen] + "…") if len(t) > maxlen else t

def page_str(d):
    pr = d.metadata.get("page_range")
    if isinstance(pr, tuple):
        return f"(págs {pr[0]+1}-{pr[1]+1})"
    p = d.metadata.get("page")
    return f"(pág {int(p)+1})" if isinstance(p, int) else ""

# ---------- load FAISS + docs ----------
print("📂 Loading FAISS (no embeddings) just to read docstore …")
vs_plain = FAISS.load_local(str(INDEX_DIR), embeddings=None, allow_dangerous_deserialization=True)
docs_all = list(vs_plain.docstore._dict.values())
print(f"Total chunks in index: {len(docs_all)}")

print("\n🧠 Loading embeddings for FAISS similarity …")
emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
vs = FAISS.load_local(str(INDEX_DIR), embeddings=emb, allow_dangerous_deserialization=True)

def find_occurrences(token: str):
    token_n = norm(token)
    hits = []
    for d in docs_all:
        if token_n in norm(d.page_content or ""):
            hits.append(d)
    return hits

def bm25_search(q: str, k=50):
    # build lexical retriever from the same docs
    bm25 = BM25Retriever.from_documents(docs_all)
    bm25.k = k
    return bm25.invoke(q)

def faiss_search(q: str, k=50):
    return vs.similarity_search(q, k=k)

def rank_of(doc_list, target_doc):
    # rank position (1-based) of target_doc by identity (source+page)
    def key(d):
        return (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("page_range"))
    tkey = key(target_doc)
    for i, d in enumerate(doc_list, 1):
        if key(d) == tkey:
            return i
    return None

# ---------- run diagnostics ----------
def diagnose(query: str, token_to_verify: str):
    print("\n" + "="*80)
    print(f"❓ Query: {query}")
    print(f"🔎 Token to verify existence: {token_to_verify}")

    # 1) Is the token really present in any chunk (post-OCR)?
    occ = find_occurrences(token_to_verify)
    print(f"\n1) Docstore scan for '{token_to_verify}': found {len(occ)} chunks.")
    if occ:
        for i, d in enumerate(occ[:3], 1):
            print(f"  [{i}] {Path(d.metadata.get('source', '')).name} {page_str(d)}")
            print("      " + preview(d))
        target = occ[0]  # take first match as the ground truth target
    else:
        target = None

    # 2) BM25 lexical search
    print("\n2) BM25 (lexical) top-10:")
    bm25_docs = bm25_search(query, k=10)
    for i, d in enumerate(bm25_docs, 1):
        print(f"  [{i}] {Path(d.metadata.get('source','')).name} {page_str(d)}")
        print("      " + preview(d))
    if target:
        r = rank_of(bm25_docs, target)
        print(f"   → Rank of target chunk in BM25: {r}")

    # 3) FAISS semantic search (embeddings)
    print("\n3) FAISS (embeddings) top-10:")
    faiss_docs = faiss_search(query, k=10)
    for i, d in enumerate(faiss_docs, 1):
        print(f"  [{i}] {Path(d.metadata.get('source','')).name} {page_str(d)}")
        print("      " + preview(d))
    if target:
        r = rank_of(faiss_docs, target)
        print(f"   → Rank of target chunk in FAISS: {r}")

    # 4) Interpretation
    print("\n4) Interpretation:")
    if not target:
        print("   • The token was NOT found in any chunk → OCR/cleaning issue (not a retriever problem).")
    else:
        b = rank_of(bm25_docs, target)
        f = rank_of(faiss_docs, target)
        if (b and b <= 3) and (not f or f > 10):
            print("   • BM25 finds it, FAISS does not → Embedding model is weak for this query (names/IDs).")
        elif (not b or b > 10) and (f and f <= 3):
            print("   • FAISS finds it, BM25 does not → Lexical mismatch (tokenization/normalization/OCR quirks).")
        elif (b and b <= 3) and (f and f <= 3):
            print("   • Both find it → Downstream prompt/LLM step likely at fault.")
        else:
            print("   • Both fail to rank it high → Try higher k or re-check chunking; token may be rare or noisy.")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Examples you care about:
    diagnose("¿Cómo se apellida el cliente Gilberto?", "Gilberto")
    diagnose("¿Quién compró el departamento 501?", "501")
