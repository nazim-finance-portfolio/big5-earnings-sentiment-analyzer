"""
Preprocessor: Cleans and chunks earnings call transcripts.

Takes raw transcript text and produces structured, chunked data ready
for sentiment analysis and RAG embedding.

Key features:
- Identifies sections (CEO remarks, CFO remarks, Q&A)
- Chunks text into overlapping segments
- Extracts speaker turns
- Saves processed data as JSON for downstream use
"""

import re
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import BANKS, QUARTERS, RAW_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# ── Section Detection Patterns ──────────────────────────

SECTION_PATTERNS = {
    "opening": [
        r"(?i)good morning.*welcome",
        r"(?i)operator.*instructions",
        r"(?i)forward.looking statements",
    ],
    "ceo_remarks": [
        r"(?i)(president|ceo|chief executive).*(remarks|speaking|comments)",
        r"(?i)(mckay|white|thomson|dodig|bhatt)",  # Big 5 CEOs
    ],
    "cfo_remarks": [
        r"(?i)(cfo|chief financial).*(remarks|speaking|comments)",
        r"(?i)(gibson|tuzun|ahn|sedran|mason)",  # Big 5 CFOs
    ],
    "cro_remarks": [
        r"(?i)(cro|chief risk).*(remarks|speaking|comments)",
        r"(?i)(hepworth|agrawal|guse)",
    ],
    "qa_session": [
        r"(?i)question.and.answer",
        r"(?i)q\s*&\s*a\s*session",
        r"(?i)open.*(?:line|floor).*questions",
        r"(?i)first question",
    ],
    "closing": [
        r"(?i)concludes.*(?:call|presentation|remarks)",
        r"(?i)thank you.*(?:joining|participating|time)",
    ],
}

# Known speakers at Big 5 banks (CEO, CFO, CRO)
KNOWN_SPEAKERS = {
    "RBC": ["Dave McKay", "Katherine Gibson", "Graeme Hepworth", "Nadine Ahn"],
    "TD": ["Bharat Masrani", "Kelvin Tran", "Ajai Bambawale", "Raymond Chun"],
    "BMO": ["Darryl White", "Tayfun Tuzun", "Piyush Agrawal"],
    "BNS": ["Scott Thomson", "Rajagopal Viswanathan", "Phil Thomas"],
    "CIBC": ["Victor Dodig", "Rob Sedran", "Frank Guse", "Hratch Panossian"],
}

# Common analyst names/firms for Q&A detection
ANALYST_FIRMS = [
    "Bank of America", "Scotiabank", "TD Securities", "BMO Capital",
    "CIBC Capital", "RBC Capital", "National Bank", "Jefferies",
    "Desjardins", "Canaccord", "Cormark", "UBS", "KBW", "Veritas",
]


def clean_text(text: str) -> str:
    """Clean raw transcript text."""
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers and headers
    text = re.sub(r"Page \d+ of \d+", "", text)
    # Remove timestamps like "11:43 AM ET"
    text = re.sub(r"\d{1,2}:\d{2}\s*(AM|PM)\s*(ET|EST|EDT|CT|PT)?", "", text)
    # Remove ticker symbols in brackets
    text = re.sub(r"\([A-Z]{1,5}[-/][A-Z]{1,5}\)", "", text)
    # Remove stock exchange references
    text = re.sub(r"(?:NYSE|TSX|TSE):\s*[A-Z]+", "", text)
    # Clean up whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    return text.strip()


def detect_section(text_segment: str) -> str:
    """Detect which section a text segment belongs to."""
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_segment):
                return section
    return "general"


def detect_speaker(line: str, bank: str) -> Optional[str]:
    """Try to detect who is speaking from a line of text."""
    # Check for explicit speaker labels like "Dave McKay --" or "Dave McKay:"
    speaker_match = re.match(r"^([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*[-—:]+", line)
    if speaker_match:
        return speaker_match.group(1).strip()

    # Check known speakers
    if bank in KNOWN_SPEAKERS:
        for speaker in KNOWN_SPEAKERS[bank]:
            if speaker.lower() in line.lower():
                return speaker

    return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks of approximately chunk_size words.

    Returns list of dicts with:
    - text: the chunk text
    - start_word: word index where chunk starts
    - end_word: word index where chunk ends
    - word_count: number of words in chunk
    """
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [{"text": text, "start_word": 0, "end_word": len(words), "word_count": len(words)}]

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
        })

        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def split_into_sections(text: str, bank: str) -> list[dict]:
    """
    Split a transcript into logical sections with speaker attribution.

    Returns list of dicts with section info.
    """
    paragraphs = text.split("\n\n")
    sections = []
    current_section = "opening"
    current_speaker = None

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue

        # Check if this paragraph starts a new section
        detected = detect_section(para)
        if detected != "general":
            current_section = detected

        # Check for speaker
        speaker = detect_speaker(para, bank)
        if speaker:
            current_speaker = speaker

        # Determine if this is Q&A (analyst question)
        is_analyst = any(firm.lower() in para.lower() for firm in ANALYST_FIRMS)
        if is_analyst:
            current_section = "qa_session"

        sections.append({
            "text": para,
            "section": current_section,
            "speaker": current_speaker,
            "is_analyst_question": is_analyst,
            "word_count": len(para.split()),
        })

    return sections


def process_transcript(bank: str, quarter: str) -> Optional[dict]:
    """
    Full processing pipeline for a single transcript.

    Returns a structured dict with sections, chunks, and metadata.
    """
    filepath = RAW_DIR / f"{bank}_{quarter}.txt"

    if not filepath.exists():
        logger.warning(f"Transcript not found: {filepath}")
        return None

    logger.info(f"Processing {bank} {quarter}...")

    # Read and clean
    raw_text = filepath.read_text(encoding="utf-8")
    clean = clean_text(raw_text)

    # Split into sections
    sections = split_into_sections(clean, bank)

    # Create chunks for RAG
    chunks = []
    for i, section in enumerate(sections):
        section_chunks = chunk_text(section["text"])
        for j, chunk in enumerate(section_chunks):
            chunks.append({
                "chunk_id": f"{bank}_{quarter}_s{i}_c{j}",
                "bank": bank,
                "bank_full_name": BANKS[bank]["full_name"],
                "quarter": quarter,
                "section": section["section"],
                "speaker": section["speaker"],
                "text": chunk["text"],
                "word_count": chunk["word_count"],
            })

    # Build processed document
    processed = {
        "bank": bank,
        "bank_full_name": BANKS[bank]["full_name"],
        "ticker": BANKS[bank]["ticker"],
        "quarter": quarter,
        "total_words": len(clean.split()),
        "num_sections": len(sections),
        "num_chunks": len(chunks),
        "sections": sections,
        "chunks": chunks,
        "processed_at": pd.Timestamp.now().isoformat(),
    }

    # Save processed data
    output_path = PROCESSED_DIR / f"{bank}_{quarter}_processed.json"
    output_path.write_text(json.dumps(processed, indent=2, default=str), encoding="utf-8")
    logger.info(f"  → Saved {len(chunks)} chunks to {output_path.name}")

    return processed


def process_all_transcripts() -> pd.DataFrame:
    """
    Process all available transcripts.

    Returns a DataFrame summary of processing results.
    """
    results = []

    for bank in BANKS:
        for quarter in QUARTERS:
            filepath = RAW_DIR / f"{bank}_{quarter}.txt"
            if not filepath.exists():
                results.append({
                    "bank": bank, "quarter": quarter,
                    "status": "missing", "chunks": 0, "words": 0
                })
                continue

            processed = process_transcript(bank, quarter)
            if processed:
                results.append({
                    "bank": bank, "quarter": quarter,
                    "status": "processed",
                    "chunks": processed["num_chunks"],
                    "words": processed["total_words"],
                })
            else:
                results.append({
                    "bank": bank, "quarter": quarter,
                    "status": "error", "chunks": 0, "words": 0
                })

    df = pd.DataFrame(results)
    print("\n📋 Processing Summary:")
    print(df.to_string(index=False))
    return df


def load_all_chunks() -> pd.DataFrame:
    """Load all processed chunks into a single DataFrame."""
    all_chunks = []

    for json_file in PROCESSED_DIR.glob("*_processed.json"):
        data = json.loads(json_file.read_text())
        all_chunks.extend(data["chunks"])

    if not all_chunks:
        logger.warning("No processed chunks found. Run processing first.")
        return pd.DataFrame()

    df = pd.DataFrame(all_chunks)
    logger.info(f"Loaded {len(df)} chunks from {df['bank'].nunique()} banks")
    return df


if __name__ == "__main__":
    process_all_transcripts()
