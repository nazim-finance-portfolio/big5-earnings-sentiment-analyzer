import re
import json
import pandas as pd
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys

# Add parent directory to path to find config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, BANKS, QUARTERS

# ══════════════════════════════════════════════════════════
#  SOPHISTICATED REGEX PATTERNS (The "54%" Logic)
# ══════════════════════════════════════════════════════════

# Future-looking phrases (Stronger than just keywords)
FORWARD_PATTERNS = [
    r"\b(will|expect|anticipate|forecast|project|predict|intend|believe)\b",
    r"\b(outlook|guidance|target|pipeline|momentum|goal|aim)\b",
    r"\b(next year|next quarter|coming year|coming quarter|2026|2027)\b",
    r"\b(looking ahead|going forward|future|opportunity|potential)\b",
    r"\b(remains? on track|well[- ]positioned|continue to)\b"
]

# Backward-looking phrases
BACKWARD_PATTERNS = [
    r"\b(reported|achieved|delivered|resulted|saw|occurred|ended)\b",
    r"\b(was|were|had|did)\b",
    r"\b(past quarter|last quarter|year-over-year|yoy|quarter-over-quarter)\b",
    r"\b(decreased?|increased?|driven by|impacted by|included)\b"
]

def get_sentence_orientation(sentence):
    """
    Returns 'forward', 'backward', or 'neutral' based on regex matches.
    """
    sentence = sentence.lower()
    
    # Count matches
    fwd_hits = sum(1 for p in FORWARD_PATTERNS if re.search(p, sentence))
    bwd_hits = sum(1 for p in BACKWARD_PATTERNS if re.search(p, sentence))
    
    if fwd_hits > bwd_hits:
        return "forward"
    elif bwd_hits > fwd_hits:
        return "backward"
    return "neutral"

def analyze_forward_guidance(target_file=None):
    """
    Main function to process transcripts.
    NOTE: Takes optional argument but defaults to None to handle standalone execution.
    """
    print("Running forward guidance analysis (Regex Enhanced)...")
    
    # Ensure directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    files = list(PROCESSED_DIR.glob("*_processed.json"))
    if not files:
        print(f"❌ No processed files found in {PROCESSED_DIR}")
        return

    analyzer = SentimentIntensityAnalyzer()
    
    chunk_results = []
    bank_metrics = []
    
    total_fwd = 0
    total_bwd = 0
    total_mixed = 0

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        
        # Parse filename info
        parts = f.stem.replace("_processed", "").split("_")
        bank = parts[0]
        quarter = "_".join(parts[1:])
        
        chunks = data if isinstance(data, list) else data.get("chunks", [])
        
        fwd_scores = []
        bwd_scores = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            
            # 1. Split into sentences
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            
            chunk_fwd_text = []
            chunk_bwd_text = []
            
            # 2. Classify each sentence
            for s in sentences:
                orientation = get_sentence_orientation(s)
                if orientation == "forward":
                    chunk_fwd_text.append(s)
                elif orientation == "backward":
                    chunk_bwd_text.append(s)
            
            # 3. Score the chunk based on dominant content
            fwd_len = len(" ".join(chunk_fwd_text))
            bwd_len = len(" ".join(chunk_bwd_text))
            
            chunk_type = "mixed"
            f_score = 0.0
            b_score = 0.0
            
            if fwd_len > 0:
                f_score = analyzer.polarity_scores(" ".join(chunk_fwd_text))['compound']
            if bwd_len > 0:
                b_score = analyzer.polarity_scores(" ".join(chunk_bwd_text))['compound']

            # Classify chunk for high-level stats
            total_len = fwd_len + bwd_len
            if total_len > 0:
                if fwd_len / total_len > 0.6:
                    chunk_type = "forward"
                    total_fwd += 1
                elif bwd_len / total_len > 0.6:
                    chunk_type = "backward"
                    total_bwd += 1
                else:
                    total_mixed += 1
            else:
                total_mixed += 1

            # Store chunk-level data (if significant)
            if f_score != 0 or b_score != 0:
                fwd_scores.append(f_score)
                bwd_scores.append(b_score)

        # Bank Level Aggregation
        avg_fwd = sum(fwd_scores)/len(fwd_scores) if fwd_scores else 0
        avg_bwd = sum(bwd_scores)/len(bwd_scores) if bwd_scores else 0
        
        bank_metrics.append({
            "bank": bank,
            "quarter": quarter,
            "total_chunks": len(chunks),
            "forward_pct": total_fwd / (total_fwd + total_bwd + total_mixed + 0.1), # approx
            "forward_sentiment": avg_fwd,
            "backward_sentiment": avg_bwd,
            "guidance_gap": avg_fwd - avg_bwd
        })

    # Save outputs
    pd.DataFrame(bank_metrics).to_csv(PROCESSED_DIR / "forward_guidance_metrics.csv", index=False)
    
    # Summary JSON
    summary = {
        "total_chunks": total_fwd + total_bwd + total_mixed,
        "forward_chunks": total_fwd,
        "backward_chunks": total_bwd,
        "mixed_chunks": total_mixed,
        "avg_forward_sentiment": round(sum(d["forward_sentiment"] for d in bank_metrics)/len(bank_metrics), 3),
        "avg_backward_sentiment": round(sum(d["backward_sentiment"] for d in bank_metrics)/len(bank_metrics), 3),
        "most_forward_looking_bank": max(bank_metrics, key=lambda x: x["forward_pct"])["bank"] if bank_metrics else "N/A",
        "most_optimistic_outlook_bank": max(bank_metrics, key=lambda x: x["forward_sentiment"])["bank"] if bank_metrics else "N/A"
    }
    
    with open(PROCESSED_DIR / "forward_guidance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================")
    print("FORWARD GUIDANCE ANALYSIS")
    print("==========================")
    total = total_fwd + total_bwd + total_mixed
    print(f"Forward-looking: {total_fwd} ({total_fwd/total:.1%})")
    print(f"Backward-looking: {total_bwd} ({total_bwd/total:.1%})")
    print(f"Mixed: {total_mixed} ({total_mixed/total:.1%})")

if __name__ == "__main__":
    analyze_forward_guidance()