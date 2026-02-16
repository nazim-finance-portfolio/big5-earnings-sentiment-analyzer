"""
Combined Sentiment Metrics.

Merges VADER and FinBERT results into a unified scoring system.
Also computes aggregate metrics per bank, per quarter, per section.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, BANKS, QUARTERS


def load_vader_results() -> pd.DataFrame:
    """Load VADER sentiment results."""
    path = PROCESSED_DIR / "vader_sentiment_results.csv"
    if not path.exists():
        logger.error("VADER results not found. Run VADER analysis first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_finbert_results() -> pd.DataFrame:
    """Load FinBERT sentiment results."""
    path = PROCESSED_DIR / "finbert_sentiment_results.csv"
    if not path.exists():
        logger.warning("FinBERT results not found. Using VADER only.")
        return pd.DataFrame()
    return pd.read_csv(path)


def combine_sentiment_scores() -> pd.DataFrame:
    """
    Combine VADER and FinBERT into a single unified sentiment dataset.

    The combined score weights:
    - VADER: 40% (good for general sentiment, fast)
    - FinBERT: 60% (better for financial context)

    If FinBERT is unavailable, uses VADER only.
    """
    vader = load_vader_results()
    finbert = load_finbert_results()

    if vader.empty:
        logger.error("No sentiment data available.")
        return pd.DataFrame()

    if finbert.empty:
        # VADER only
        vader["combined_score"] = vader["compound"]
        vader["combined_label"] = vader["label"]
        vader["model_used"] = "vader_only"
        combined = vader
    else:
        # Merge VADER and FinBERT
        combined = vader.merge(
            finbert[["chunk_id", "finbert_label", "finbert_score", "finbert_compound"]],
            on="chunk_id",
            how="left",
        )

        # Weighted combination
        combined["combined_score"] = (
            0.4 * combined["compound"] +
            0.6 * combined["finbert_compound"].fillna(combined["compound"])
        ).round(4)

        # Label from combined score
        combined["combined_label"] = combined["combined_score"].apply(
            lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
        )
        combined["model_used"] = "vader+finbert"

    # Save combined results
    output_path = PROCESSED_DIR / "combined_sentiment.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved combined sentiment: {len(combined)} chunks → {output_path}")

    return combined


def compute_aggregate_metrics(combined: pd.DataFrame = None) -> dict:
    """
    Compute aggregate sentiment metrics for dashboarding.

    Returns nested dict structure:
    {
        "by_bank_quarter": DataFrame,
        "by_bank": DataFrame,
        "by_section": DataFrame,
        "by_bank_section": DataFrame,
        "overall": dict,
    }
    """
    if combined is None:
        path = PROCESSED_DIR / "combined_sentiment.csv"
        if not path.exists():
            combined = combine_sentiment_scores()
        else:
            combined = pd.read_csv(path)

    if combined.empty:
        return {}

    metrics = {}

    # 1. Per bank per quarter
    metrics["by_bank_quarter"] = (
        combined.groupby(["bank", "quarter"])
        .agg(
            avg_sentiment=("combined_score", "mean"),
            std_sentiment=("combined_score", "std"),
            pct_positive=("combined_label", lambda x: (x == "positive").mean()),
            pct_negative=("combined_label", lambda x: (x == "negative").mean()),
            pct_neutral=("combined_label", lambda x: (x == "neutral").mean()),
            chunk_count=("chunk_id", "count"),
        )
        .round(4)
        .reset_index()
    )

    # 2. Per bank (overall across quarters)
    metrics["by_bank"] = (
        combined.groupby("bank")
        .agg(
            avg_sentiment=("combined_score", "mean"),
            std_sentiment=("combined_score", "std"),
            pct_positive=("combined_label", lambda x: (x == "positive").mean()),
            pct_negative=("combined_label", lambda x: (x == "negative").mean()),
            chunk_count=("chunk_id", "count"),
        )
        .round(4)
        .reset_index()
        .sort_values("avg_sentiment", ascending=False)
    )

    # 3. Per section type
    metrics["by_section"] = (
        combined.groupby("section")
        .agg(
            avg_sentiment=("combined_score", "mean"),
            chunk_count=("chunk_id", "count"),
        )
        .round(4)
        .reset_index()
        .sort_values("avg_sentiment", ascending=False)
    )

    # 4. Per bank per section
    metrics["by_bank_section"] = (
        combined.groupby(["bank", "section"])
        .agg(
            avg_sentiment=("combined_score", "mean"),
            chunk_count=("chunk_id", "count"),
        )
        .round(4)
        .reset_index()
    )

    # 5. Overall
    metrics["overall"] = {
        "avg_sentiment": round(combined["combined_score"].mean(), 4),
        "most_positive_bank": metrics["by_bank"].iloc[0]["bank"],
        "most_negative_bank": metrics["by_bank"].iloc[-1]["bank"],
        "total_chunks": len(combined),
        "total_transcripts": combined.groupby(["bank", "quarter"]).ngroups,
    }

    # Save all metrics
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(PROCESSED_DIR / f"metrics_{key}.csv", index=False)
        elif isinstance(value, dict):
            (PROCESSED_DIR / f"metrics_{key}.json").write_text(
                json.dumps(value, indent=2)
            )

    logger.info(f"Computed aggregate metrics across {metrics['overall']['total_transcripts']} transcripts")

    # Print summary
    print("\n📊 Sentiment Rankings (Most Positive → Most Negative):")
    print(metrics["by_bank"][["bank", "avg_sentiment", "pct_positive", "pct_negative"]].to_string(index=False))
    print(f"\n🏆 Most positive: {metrics['overall']['most_positive_bank']}")
    print(f"📉 Most negative: {metrics['overall']['most_negative_bank']}")

    return metrics


if __name__ == "__main__":
    combined = combine_sentiment_scores()
    if not combined.empty:
        compute_aggregate_metrics(combined)
