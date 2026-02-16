"""
VADER Sentiment Analyzer for earnings call transcripts.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is rule-based
and works well for financial text. It's fast and doesn't need a GPU.

Output: compound score (-1 to +1), plus pos/neg/neu percentages.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR


# ── Financial Lexicon Additions ──────────────────────────
# VADER doesn't know finance-specific terms, so we add them
FINANCIAL_LEXICON = {
    # Positive financial terms
    "growth": 1.5,
    "outperform": 2.0,
    "exceeded": 1.8,
    "record": 1.5,
    "strong": 1.5,
    "robust": 1.5,
    "momentum": 1.3,
    "tailwind": 1.5,
    "upgrade": 1.5,
    "dividend": 1.0,
    "synergies": 1.3,
    "resilient": 1.5,
    "beat": 1.5,
    "surpassed": 1.8,
    "accelerating": 1.3,
    "outpaced": 1.5,
    "optimistic": 1.5,
    "confidence": 1.3,

    # Negative financial terms
    "headwind": -1.5,
    "provision": -0.8,
    "impairment": -1.8,
    "writedown": -2.0,
    "restructuring": -1.0,
    "downgrade": -1.5,
    "delinquency": -1.5,
    "default": -1.5,
    "recession": -2.0,
    "deterioration": -1.8,
    "challenging": -1.0,
    "uncertainty": -1.0,
    "headcount reduction": -1.5,
    "credit losses": -1.3,
    "elevated risk": -1.5,
    "pressure": -0.8,
    "miss": -1.5,
    "missed": -1.5,
    "shortfall": -1.5,
    "slowdown": -1.3,
}


def create_analyzer() -> SentimentIntensityAnalyzer:
    """Create a VADER analyzer with financial lexicon additions."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCIAL_LEXICON)
    return analyzer


def analyze_text(analyzer: SentimentIntensityAnalyzer, text: str) -> dict:
    """
    Analyze sentiment of a single text string.

    Returns dict with:
    - compound: overall score (-1 to +1)
    - pos, neg, neu: percentage of text that's positive/negative/neutral
    - label: "positive", "negative", or "neutral"
    """
    scores = analyzer.polarity_scores(text)

    # Classify based on compound score
    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": round(scores["compound"], 4),
        "positive": round(scores["pos"], 4),
        "negative": round(scores["neg"], 4),
        "neutral": round(scores["neu"], 4),
        "label": label,
    }


def analyze_transcript(bank: str, quarter: str) -> pd.DataFrame | None:
    """
    Run VADER analysis on all chunks of a processed transcript.

    Returns DataFrame with sentiment scores per chunk.
    """
    filepath = PROCESSED_DIR / f"{bank}_{quarter}_processed.json"

    if not filepath.exists():
        logger.warning(f"Processed data not found: {filepath}")
        return None

    data = json.loads(filepath.read_text())
    analyzer = create_analyzer()

    results = []
    for chunk in data["chunks"]:
        sentiment = analyze_text(analyzer, chunk["text"])
        results.append({
            "chunk_id": chunk["chunk_id"],
            "bank": chunk["bank"],
            "quarter": chunk["quarter"],
            "section": chunk["section"],
            "speaker": chunk["speaker"],
            "word_count": chunk["word_count"],
            "text_preview": chunk["text"][:100] + "...",
            **sentiment,
        })

    df = pd.DataFrame(results)
    logger.info(
        f"VADER analysis for {bank} {quarter}: "
        f"avg={df['compound'].mean():.3f}, "
        f"pos={len(df[df['label']=='positive'])}, "
        f"neg={len(df[df['label']=='negative'])}, "
        f"neu={len(df[df['label']=='neutral'])}"
    )
    return df


def analyze_all_transcripts() -> pd.DataFrame:
    """
    Run VADER analysis on ALL processed transcripts.

    Returns combined DataFrame and saves to disk.
    """
    all_results = []

    for json_file in sorted(PROCESSED_DIR.glob("*_processed.json")):
        data = json.loads(json_file.read_text())
        bank = data["bank"]
        quarter = data["quarter"]

        df = analyze_transcript(bank, quarter)
        if df is not None:
            all_results.append(df)

    if not all_results:
        logger.warning("No transcripts to analyze.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Save results
    output_path = PROCESSED_DIR / "vader_sentiment_results.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved VADER results: {len(combined)} chunks → {output_path}")

    # Print summary
    summary = (
        combined.groupby(["bank", "quarter"])["compound"]
        .agg(["mean", "std", "min", "max"])
        .round(3)
    )
    print("\n📊 VADER Sentiment Summary:")
    print(summary.to_string())

    return combined


if __name__ == "__main__":
    analyze_all_transcripts()
