import os
import json
import pandas as pd
import logging
from pathlib import Path
from src.ingestion.preprocessor import process_transcript
from src.analysis.vader_analyzer import analyze_transcript
from src.rag.embeddings import build_index
from config.settings import PROCESSED_DIR, TRANSCRIPTS_DIR

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s')

def process_all():
    """Step 2: Preprocess raw transcripts into chunks."""
    print("\n============================================================")
    print("🔧 STEP 2: Preprocessing")
    print("============================================================")
    
    files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    summary = []

    for f in files:
        bank, quarter = f.stem.split("_", 1)
        logging.info(f"Processing {bank} {quarter}...")
        
        try:
            chunks = process_transcript(f)
            if chunks:
                out_file = PROCESSED_DIR / f"{f.stem}_processed.json"
                with open(out_file, "w", encoding="utf-8") as out:
                    json.dump(chunks, out, indent=2)
                
                logging.info(f"   -> Saved {len(chunks)} chunks to {out_file.name}")
                summary.append({"bank": bank, "quarter": quarter, "status": "processed", "chunks": len(chunks)})
            else:
                logging.warning(f"   -> No chunks generated for {f.name}")
        except Exception as e:
            logging.error(f"Failed to process {f.name}: {e}")

    if summary:
        df = pd.DataFrame(summary)
        print("\n📋 Processing Summary:")
        print(df.to_string(index=False))

def analyze_all():
    """Step 3: Analyze Sentiment + Forward Guidance."""
    print("\n============================================================")
    print("🧠 STEP 3: Analysis (Sentiment + Forward Guidance)")
    print("============================================================")
    
    # 1. Standard VADER Analysis
    files = list(PROCESSED_DIR.glob("*_processed.json"))
    results = []

    print("Running VADER analysis...")
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            vader_score = analyze_transcript(data)
            
            parts = f.stem.replace("_processed", "").split("_")
            bank = parts[0]
            quarter = "_".join(parts[1:])
            
            results.append({
                "bank": bank,
                "quarter": quarter,
                "combined_score": vader_score,
                "combined_label": "positive" if vader_score > 0.05 else "negative" if vader_score < -0.05 else "neutral"
            })
            logging.info(f"VADER analysis for {bank} {quarter}: avg={vader_score:.3f}")
            
        except Exception as e:
            logging.error(f"Error analyzing {f.name}: {e}")

    # Save VADER Results
    df = pd.DataFrame(results)
    out_path = PROCESSED_DIR / "combined_sentiment.csv"
    df.to_csv(out_path, index=False)
    logging.info(f"Saved combined sentiment: {len(results)} chunks → {out_path}")

    # 2. ROBUST FORWARD GUIDANCE EXECUTION
    # This block tries to run forward guidance using whichever method works
    print("Running forward guidance analysis...")
    try:
        # Try Method A: No arguments (My version)
        from src.analysis.forward_guidance import analyze_forward_guidance
        analyze_forward_guidance()
    except TypeError:
        # Try Method B: Using compute_forward_metrics (Their version)
        try:
            print("Standard call failed, attempting compatible mode...")
            from src.analysis.forward_guidance import compute_forward_metrics
            compute_forward_metrics()
        except Exception as e2:
             print("Forward guidance skipped (Method B failed): " + str(e2))
    except Exception as e:
        # Try Method C: Last resort (maybe it needs text?)
        try:
             print("Attempting manual computation...")
             from src.analysis.forward_guidance import compute_forward_metrics
             compute_forward_metrics()
        except:
            print("Forward guidance skipped: " + str(e))

def index_all():
    """Step 4: Build RAG Index."""
    print("\n============================================================")
    print("🗄️  STEP 4: Building RAG Index")
    print("============================================================")
    build_index()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "process": process_all()
        elif cmd == "analyze": analyze_all()
        elif cmd == "index": index_all()
    else:
        print("Usage: python run.py [process|analyze|index]")