"""
ingest.py
---------
Indexing pipeline for the Legal RAG Agent.

Now with OCR fallback:
- Load pages with PyMuPDF
- If a page has little/no text, render and OCR it (Spanish by default)
- Then split, embed, and index as before
"""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from config import RAW_DIR, INDEX_DIR, EMBEDDING_MODEL

# --- OCR configuration ---
OCR_LANG = "spa"           # Spanish OCR; add "+eng" if mixed
EMPTY_TEXT_THRESHOLD = 30  # chars; below this we OCR the page
OCR_DPI = 200              # render resolution for OCR (trade speed/accuracy)


def load_pdfs(folder: Path) -> List[Document]:
    """Load all PDF files into page-level Documents with PyMuPDFLoader,
    then apply OCR fallback on pages with little/no text.
    Also injects 'filename' into each Document.metadata.
    """
    pdf_paths = sorted(folder.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {folder}")

    all_docs: List[Document] = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(str(path))
        page_docs = loader.load()

        fixed_pages: List[Document] = []
        with fitz.open(str(path)) as doc:
            for d in page_docs:
                text = d.page_content or ""

                # Base metadata: keep everything, add filename explicitly
                base_meta = {
                    **(d.metadata or {}),
                    "filename": path.name,  # <-- add filename here
                }

                if len(text) >= EMPTY_TEXT_THRESHOLD:
                    # Enough native text; just ensure filename is present
                    fixed_pages.append(
                        Document(page_content=text, metadata=base_meta)
                    )
                    continue

                # OCR fallback
                page_index = int(d.metadata.get("page", 0))
                page = doc[page_index]
                pix = page.get_pixmap(dpi=OCR_DPI)
                img = Image.open(BytesIO(pix.tobytes("png")))

                data = pytesseract.image_to_data(
                    img, lang=OCR_LANG, output_type=pytesseract.Output.DICT
                )
                words = [w for w in data["text"] if w.strip()]
                confs = [
                    c for c in data["conf"]
                    if isinstance(c, (int, float)) and c >= 0
                ]
                ocr_text = " ".join(words).strip()
                ocr_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

                if len(ocr_text) >= EMPTY_TEXT_THRESHOLD:
                    fixed_pages.append(
                        Document(
                            page_content=ocr_text,
                            metadata={
                                **base_meta,
                                "ocr_applied": True,
                                "ocr_confidence": round(ocr_conf, 3),
                            },
                        )
                    )
                else:
                    fixed_pages.append(
                        Document(
                            page_content=text,  # keep original (possibly empty)
                            metadata={
                                **base_meta,
                                "ocr_applied": True,
                                "ocr_confidence": round(ocr_conf, 3),
                                "ocr_note": "OCR failed to extract useful text",
                            },
                        )
                    )

        all_docs.extend(fixed_pages)
        print(f"✅ Loaded {path.name} (pages: {len(page_docs)})")
    return all_docs


def split_docs(docs: List[Document]) -> List[Document]:
    """Split long documents into smaller chunks.
    Ensures 'filename' stays in chunk metadata, deriving from 'source' if needed.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    splits = splitter.split_documents(docs)

    # Ensure filename persists/exists in each chunk
    enhanced: List[Document] = []
    for d in splits:
        meta = dict(d.metadata or {})
        if "filename" not in meta:
            src = meta.get("source")
            if src:
                meta["filename"] = Path(src).name
        enhanced.append(Document(page_content=d.page_content, metadata=meta))

    # Optional: drop tiny chunks (noise)
    MIN_CHARS = 50
    filtered = [d for d in enhanced if len((d.page_content or "").strip()) >= MIN_CHARS]

    print(f"✂️  Split into {len(splits)} chunks → kept {len(filtered)} after filtering")
    return filtered


def build_embeddings():
    print(f"🧠 Using embeddings model: {EMBEDDING_MODEL}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_index(chunks: List[Document], index_dir: Path):
    embeddings = build_embeddings()
    print("⚙️  Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir))
    print(f"💾 Index saved to {index_dir}")


def main():
    print("🚀 Starting ingestion pipeline (with OCR fallback)...")
    docs = load_pdfs(RAW_DIR)

    # quick per-file stats
    by_src = {}
    for d in docs:
        # prefer 'filename' but fall back to 'source'
        fname = d.metadata.get("filename") or Path(d.metadata.get("source", "unknown")).name
        by_src.setdefault(fname, 0)
        by_src[fname] += len(d.page_content or "")
    print("\n📊 Extractable characters per file (after OCR fallback):")
    for k, v in sorted(by_src.items(), key=lambda kv: kv[1], reverse=True):
        print(f"- {k:45} chars={v}")

    chunks = split_docs(docs)
    build_index(chunks, INDEX_DIR)
    print("✅ Ingestion complete.")


if __name__ == "__main__":
    main()
