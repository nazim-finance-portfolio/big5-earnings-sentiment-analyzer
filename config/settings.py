"""
Centralized configuration for Big 5 Earnings Analyzer.
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Banks Configuration ────────────────────────────────────
BANKS = {
    "RBC": {
        "full_name": "Royal Bank of Canada",
        "ticker": "RY",
        "ir_url": "https://www.rbc.com/investor-relations/",
    },
    "TD": {
        "full_name": "Toronto-Dominion Bank",
        "ticker": "TD",
        "ir_url": "https://www.td.com/ca/en/about-td/investor-relations",
    },
    "BMO": {
        "full_name": "Bank of Montreal",
        "ticker": "BMO",
        "ir_url": "https://www.bmo.com/main/about-bmo/investor-relations/",
    },
    "BNS": {
        "full_name": "Bank of Nova Scotia (Scotiabank)",
        "ticker": "BNS",
        "ir_url": "https://www.scotiabank.com/ca/en/about/investors-shareholders.html",
    },
    "CIBC": {
        "full_name": "Canadian Imperial Bank of Commerce",
        "ticker": "CM",
        "ir_url": "https://www.cibc.com/en/about-cibc/investor-relations.html",
    },
}

# ── Auto-detect quarters from data/raw folder ──────────────
def detect_quarters():
    """Scan data/raw for all BANK_QUARTER.txt files and return unique quarters."""
    quarters = set()
    if RAW_DIR.exists():
        for f in RAW_DIR.glob("*.txt"):
            # Match pattern like RBC_Q1_2024.txt or TD_Q3_2025.txt
            match = re.match(r"[A-Z]+_(Q[1-4]_\d{4})\.txt", f.name)
            if match:
                quarters.add(match.group(1))
    if not quarters:
        # Fallback default
        return ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]
    return sorted(quarters)

QUARTERS = detect_quarters()

# ── NLP Settings ─────────────────────────────────────────
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # overlapping words between chunks
FINBERT_MODEL = "ProsusAI/finbert"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── RAG Settings ─────────────────────────────────────────
TOP_K_RESULTS = 5  # number of chunks to retrieve
# Use OpenAI if available, otherwise Claude
if OPENAI_API_KEY:
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o-mini"  # cheap and fast
else:
    LLM_PROVIDER = "anthropic"
    LLM_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MODEL = LLM_MODEL  # backward compatibility
MAX_TOKENS = 2048

# ── Dashboard Settings ───────────────────────────────────
STREAMLIT_PORT = 8501
SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#eab308",
    "negative": "#ef4444",
}
BANK_COLORS = {
    "RBC": "#003DA5",
    "TD": "#2B8000",
    "BMO": "#0075BE",
    "BNS": "#EC1C24",
    "CIBC": "#8B0000",
}
