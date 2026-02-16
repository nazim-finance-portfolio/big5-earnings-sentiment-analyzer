"""
FinBERT Sentiment Analyzer for earnings call transcripts.

FinBERT is a pre-trained NLP model fine-tuned specifically for financial text.
It provides more nuanced sentiment than VADER, especially for complex financial language.

Model: ProsusAI/finbert (Hugging Face)
Note: Requires GPU for fast inference, but works on CPU (slower).
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, FINBERT_MODEL


# Global model cache
_finbert_pipeline = None


def get_finbert_pipeline():
    """Load FinBERT model (cached after first call)."""
    global _finbert_pipeline

    if _finbert_pipeline is None:
        logger.info(f"Loading FinBERT model: {FINBERT_MODEL}")
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=512,
            truncation=True,
        )
        device_name = "GPU" if device == 0 else "CPU"
        logger.info(f"FinBERT loaded on {device_name}")

    return _finbert_pipeline


def analyze_text_finbert(text: str) -> dict:
    """
    Analyze sentiment using FinBERT.

    Returns dict with:
    - label: "positive", "negative", or "neutral"
    - score: confidence (0 to 1)
    - compound: mapped score (-1 to +1) for consistency with VADER
    """
    pipe = get_finbert_pipeline()

    # FinBERT max length is 512 tokens — truncate long text
    # Take first 450 words to stay under token limit
    words = text.split()
    if len(words) > 450:
        text = " ".join(words[:450])

    try:
        result = pipe(text)[0]
        label = result["label"].lower()
        confidence = result["score"]

        # Map to compound score for consistency with VADER
        if label == "positive":
            compound = confidence
        elif label == "negative":
            compound = -confidence
        else:
            compound = 0.0

        return {
            "finbert_label": label,
            "finbert_score": round(confidence, 4),
            "finbert_compound": round(compound, 4),
        }
    except Exception as e:
        logger.error(f"FinBERT error: {e}")
        return {
            "finbert_label": "error",
            "finbert_score": 0.0,
            "finbert_compound": 0.0,
        }


def analyze_transcript_finbert(bank: str, quarter: str) -> Optional[pd.DataFrame]:
    """
    Run FinBERT analysis on all chunks of a processed transcript.
    """
    filepath = PROCESSED_DIR / f"{bank}_{quarter}_processed.json"

    if not filepath.exists():
        logger.warning(f"Processed data not found: {filepath}")
        return None

    data = json.loads(filepath.read_text())
    results = []

    logger.info(f"Running FinBERT on {bank} {quarter} ({len(data['chunks'])} chunks)...")

    for i, chunk in enumerate(data["chunks"]):
        sentiment = analyze_text_finbert(chunk["text"])
        results.append({
            "chunk_id": chunk["chunk_id"],
            "bank": chunk["bank"],
            "quarter": chunk["quarter"],
            "section": chunk["section"],
            "speaker": chunk["speaker"],
            **sentiment,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(data['chunks'])} chunks")

    df = pd.DataFrame(results)
    logger.info(
        f"FinBERT analysis for {bank} {quarter}: "
        f"pos={len(df[df['finbert_label']=='positive'])}, "
        f"neg={len(df[df['finbert_label']=='negative'])}, "
        f"neu={len(df[df['finbert_label']=='neutral'])}"
    )
    return df


def analyze_all_transcripts_finbert() -> pd.DataFrame:
    """
    Run FinBERT on ALL processed transcripts.
    """
    all_results = []

    for json_file in sorted(PROCESSED_DIR.glob("*_processed.json")):
        data = json.loads(json_file.read_text())
        bank = data["bank"]
        quarter = data["quarter"]

        df = analyze_transcript_finbert(bank, quarter)
        if df is not None:
            all_results.append(df)

    if not all_results:
        logger.warning("No transcripts to analyze with FinBERT.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Save results
    output_path = PROCESSED_DIR / "finbert_sentiment_results.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved FinBERT results: {len(combined)} chunks → {output_path}")

    return combined


if __name__ == "__main__":
    analyze_all_transcripts_finbert()
