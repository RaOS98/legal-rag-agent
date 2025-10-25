from pathlib import Path
from langchain_community.vectorstores import FAISS
from config import INDEX_DIR
vs = FAISS.load_local(str(INDEX_DIR), embeddings=None, allow_dangerous_deserialization=True)

hits = []
for doc_id in vs.docstore._dict:
    d = vs.docstore._dict[doc_id]
    txt = (d.page_content or "")
    if "gilberto" in txt.lower():
        hits.append((d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("page_range"), txt[:200].replace("\n"," ")+"..."))

print(f"Found {len(hits)} occurrences of 'Gilberto':")
for i,(src,page,pr,snip) in enumerate(hits,1):
    print(f"\n[{i}] {src} page={page} range={pr}\n{snip}")