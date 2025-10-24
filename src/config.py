# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env file if it exists
load_dotenv()

# === Directories ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

# === Models ===
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# === Retrieval Settings ===
TOP_K = int(os.getenv("TOP_K", 5))

# === Utility ===
def ensure_dirs():
    """Ensure required folders exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

ensure_dirs()
