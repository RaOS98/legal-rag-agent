"""
cli.py
------
Simple CLI for the Legal RAG Agent.

Commands:
- ingest : build/update the FAISS index from PDFs in data/raw
- ask    : query the index and get an answer from the local LLM
"""

import argparse
from pathlib import Path

# reuse the existing modules
import ingest as ingest_mod
from rag import ask
from config import RAW_DIR, INDEX_DIR


def cmd_ingest(args: argparse.Namespace):
    # Just call the ingest main to keep things DRY
    print(f"📂 RAW_DIR:   {RAW_DIR}")
    print(f"💾 INDEX_DIR: {INDEX_DIR}")
    ingest_mod.main()


def cmd_ask(args: argparse.Namespace):
    question = args.q
    print(f"🧑‍⚖️ Pregunta: {question}\n")
    answer = ask(question)
    print("\n🧾 Respuesta:")
    print(answer)


def main():
    parser = argparse.ArgumentParser(prog="legal_rag_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Build/update the FAISS index from PDFs in data/raw")

    p_ask = sub.add_parser("ask", help="Ask a question against the indexed PDFs")
    p_ask.add_argument("-q", "--q", required=True, help="Pregunta en español")

    args = parser.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
