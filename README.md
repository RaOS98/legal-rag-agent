# legal-rag-agent#

Minimal RAG over scanned/legal PDFs using LangChain + FAISS.

## Quickstart
```bash
# 1) create env and install
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) ingest a PDF (or a folder)
python -m cli ingest --path data/raw/minuta_123.pdf

# 3) ask a question
python -m cli ask --q "¿Qué pasa si incumple el pago?" --k 5