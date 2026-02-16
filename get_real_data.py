"""
=============================================================
 GET REAL TRANSCRIPTS - Super Easy Guide for Biplob
=============================================================

Bro, follow these steps EXACTLY. It's basically just copy-paste.

There are 3 methods. Try Method 1 first (easiest), then Method 2, then Method 3.

=============================================================
METHOD 1: FREE from Motley Fool (EASIEST - just run this script)
=============================================================

The Motley Fool publishes earnings call transcripts for free.
This script will try to grab them automatically.

Just run:
    python get_real_data.py auto

=============================================================
METHOD 2: Copy from Seeking Alpha (FREE with account)
=============================================================

Step 1: Go to https://seekingalpha.com (create a FREE account if you don't have one)
Step 2: Search for each bank transcript. Here are the direct links:

    RBC:
    - Q1 2024: https://seekingalpha.com/article/4674327-royal-bank-of-canada-ry-q1-2024-earnings-call-transcript
    - Q2 2024: https://seekingalpha.com/article/4696490-royal-bank-of-canada-ry-q2-2024-earnings-call-transcript
    - Q3 2024: https://seekingalpha.com/article/4717664-royal-bank-of-canada-ry-q3-2024-earnings-call-transcript
    - Q4 2024: https://seekingalpha.com/article/4742106-royal-bank-of-canada-ry-q4-2024-earnings-call-transcript

    TD:
    - Q1 2024: https://seekingalpha.com/article/4674082-toronto-dominion-bank-td-q1-2024-earnings-call-transcript
    - Q2 2024: https://seekingalpha.com/article/4696541-toronto-dominion-bank-td-q2-2024-earnings-call-transcript
    - Q3 2024: https://seekingalpha.com/article/4717895-toronto-dominion-bank-td-q3-2024-earnings-call-transcript
    - Q4 2024: https://seekingalpha.com/article/4742378-toronto-dominion-bank-td-q4-2024-earnings-call-transcript

    BMO:
    - Q1 2024: https://seekingalpha.com/article/4673862-bank-of-montreal-bmo-q1-2024-earnings-call-transcript
    - Q2 2024: https://seekingalpha.com/article/4696254-bank-of-montreal-bmo-q2-2024-earnings-call-transcript
    - Q3 2024: https://seekingalpha.com/article/4717893-bank-of-montreal-bmo-q3-2024-earnings-call-transcript
    - Q4 2024: https://seekingalpha.com/article/4742385-bank-of-montreal-bmo-q4-2024-earnings-call-transcript

    Scotiabank (BNS):
    - Q1 2024: https://seekingalpha.com/article/4674094-bank-of-nova-scotia-bns-q1-2024-earnings-call-transcript
    - Q2 2024: https://seekingalpha.com/article/4696255-bank-of-nova-scotia-bns-q2-2024-earnings-call-transcript
    - Q3 2024: https://seekingalpha.com/article/4717665-bank-of-nova-scotia-bns-q3-2024-earnings-call-transcript
    - Q4 2024: https://seekingalpha.com/article/4742103-bank-of-nova-scotia-bns-q4-2024-earnings-call-transcript

    CIBC:
    - Q1 2024: https://seekingalpha.com/article/4674326-canadian-imperial-bank-of-commerce-cm-q1-2024-earnings-call-transcript
    - Q2 2024: https://seekingalpha.com/article/4696539-canadian-imperial-bank-of-commerce-cm-q2-2024-earnings-call-transcript
    - Q3 2024: https://seekingalpha.com/article/4717894-canadian-imperial-bank-of-commerce-cm-q3-2024-earnings-call-transcript
    - Q4 2024: https://seekingalpha.com/article/4742377-canadian-imperial-bank-of-commerce-cm-q4-2024-earnings-call-transcript

Step 3: For EACH link:
    a) Open the page
    b) Select ALL the transcript text (Ctrl+A or just highlight the earnings call text)
    c) Copy it (Ctrl+C)
    d) Run this command:
       python get_real_data.py paste RBC Q1_2024
       (it will ask you to paste, then press Enter twice)

    Do this for each bank and quarter. It takes about 15-20 minutes total.

=============================================================
METHOD 3: From CIBC's free PDFs (one bank fully free!)
=============================================================

CIBC publishes free transcript PDFs on their website!
    - Go to: https://www.cibc.com/en/about-cibc/investor-relations/quarterly-results.html
    - Click on each quarter
    - Download the "Transcript" PDF
    - Then run: python get_real_data.py pdf CIBC Q1_2024 path/to/downloaded.pdf

=============================================================
AFTER GETTING DATA, run these commands to rebuild everything:
=============================================================

    python run.py process      # Re-process the new real data
    python run.py analyze      # Re-analyze sentiment 
    python run.py index        # Rebuild the RAG index
    python run.py dashboard    # Relaunch dashboard with REAL data!

=============================================================
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install dependencies first: pip install requests beautifulsoup4 lxml")
    sys.exit(1)

from config.settings import BANKS, QUARTERS, RAW_DIR


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Motley Fool Transcript URLs (these are the free ones)
MOTLEY_FOOL_URLS = {
    "RBC_Q4_2024": "https://www.fool.com/earnings/call-transcripts/2024/12/04/royal-bank-of-canada-ry-q4-2024-earnings-call-tra/",
    "RBC_Q3_2024": "https://www.fool.com/earnings/call-transcripts/2024/08/28/royal-bank-of-canada-ry-q3-2024-earnings-call-tra/",
    "RBC_Q2_2024": "https://www.fool.com/earnings/call-transcripts/2024/05/30/royal-bank-of-canada-ry-q2-2024-earnings-call-tra/",
    "RBC_Q1_2024": "https://www.fool.com/earnings/call-transcripts/2024/02/28/royal-bank-of-canada-ry-q1-2024-earnings-call-tra/",
    "TD_Q4_2024": "https://www.fool.com/earnings/call-transcripts/2024/12/05/toronto-dominion-bank-td-q4-2024-earnings-call-tr/",
    "TD_Q3_2024": "https://www.fool.com/earnings/call-transcripts/2024/08/22/toronto-dominion-bank-td-q3-2024-earnings-call-tr/",
    "TD_Q2_2024": "https://www.fool.com/earnings/call-transcripts/2024/05/23/toronto-dominion-bank-td-q2-2024-earnings-call-tr/",
    "TD_Q1_2024": "https://www.fool.com/earnings/call-transcripts/2024/02/29/toronto-dominion-bank-td-q1-2024-earnings-call-tr/",
    "BMO_Q4_2024": "https://www.fool.com/earnings/call-transcripts/2024/12/05/bank-of-montreal-bmo-q4-2024-earnings-call-transc/",
    "BMO_Q3_2024": "https://www.fool.com/earnings/call-transcripts/2024/08/27/bank-of-montreal-bmo-q3-2024-earnings-call-transc/",
    "BMO_Q2_2024": "https://www.fool.com/earnings/call-transcripts/2024/05/29/bank-of-montreal-bmo-q2-2024-earnings-call-transc/",
    "BMO_Q1_2024": "https://www.fool.com/earnings/call-transcripts/2024/02/27/bank-of-montreal-bmo-q1-2024-earnings-call-transc/",
    "BNS_Q4_2024": "https://www.fool.com/earnings/call-transcripts/2024/12/03/bank-of-nova-scotia-bns-q4-2024-earnings-call-tra/",
    "BNS_Q3_2024": "https://www.fool.com/earnings/call-transcripts/2024/08/27/bank-of-nova-scotia-bns-q3-2024-earnings-call-tra/",
    "BNS_Q2_2024": "https://www.fool.com/earnings/call-transcripts/2024/05/28/bank-of-nova-scotia-bns-q2-2024-earnings-call-tra/",
    "BNS_Q1_2024": "https://www.fool.com/earnings/call-transcripts/2024/02/27/bank-of-nova-scotia-bns-q1-2024-earnings-call-tra/",
    "CIBC_Q4_2024": "https://www.fool.com/earnings/call-transcripts/2024/12/05/canadian-imperial-bank-of-commerce-cm-q4-2024-ear/",
    "CIBC_Q3_2024": "https://www.fool.com/earnings/call-transcripts/2024/08/29/canadian-imperial-bank-of-commerce-cm-q3-2024-ear/",
    "CIBC_Q2_2024": "https://www.fool.com/earnings/call-transcripts/2024/05/30/canadian-imperial-bank-of-commerce-cm-q2-2024-ear/",
    "CIBC_Q1_2024": "https://www.fool.com/earnings/call-transcripts/2024/02/29/canadian-imperial-bank-of-commerce-cm-q1-2024-ear/",
}


def scrape_motley_fool(url: str) -> str | None:
    """Scrape transcript from Motley Fool."""
    try:
        print(f"  Fetching: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Try different selectors for Motley Fool
        article = (
            soup.find("div", class_="article-body") or
            soup.find("div", class_="tailwind-article-body") or
            soup.find("article") or
            soup.find("div", {"data-id": "article-body"})
        )

        if article:
            for tag in article.find_all(["script", "style", "aside", "nav"]):
                tag.decompose()
            text = article.get_text(separator="\n", strip=True)
            if len(text) > 500:
                return text

        # Fallback: try to get all paragraph text
        paragraphs = soup.find_all("p")
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
        if len(text) > 500:
            return text

        print(f"  ⚠️  Could not extract enough text from {url}")
        return None

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def save_transcript(bank: str, quarter: str, text: str, source: str):
    """Save transcript to data/raw/"""
    filename = f"{bank}_{quarter}.txt"
    filepath = RAW_DIR / filename

    filepath.write_text(text, encoding="utf-8")

    meta = {
        "bank": bank,
        "quarter": quarter,
        "source": source,
        "saved_at": datetime.now().isoformat(),
        "word_count": len(text.split()),
    }
    meta_path = RAW_DIR / f"{bank}_{quarter}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  ✅ Saved {filename} ({meta['word_count']:,} words)")


def auto_scrape():
    """Method 1: Auto-scrape from Motley Fool."""
    print("\n🤖 Auto-scraping transcripts from Motley Fool...")
    print("=" * 50)

    success = 0
    failed = 0

    for key, url in MOTLEY_FOOL_URLS.items():
        bank, quarter = key.split("_", 1)

        filepath = RAW_DIR / f"{bank}_{quarter}.txt"
        if filepath.exists():
            word_count = len(filepath.read_text().split())
            if word_count > 1000:  # Only skip if we have substantial data
                print(f"  ⏭️  {bank} {quarter} already exists ({word_count:,} words)")
                success += 1
                continue

        text = scrape_motley_fool(url)

        if text and len(text) > 500:
            save_transcript(bank, quarter, text, source=url)
            success += 1
        else:
            print(f"  ❌ Failed: {bank} {quarter}")
            failed += 1

        time.sleep(2)  # Be polite to their servers

    print(f"\n{'=' * 50}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print("\nFor failed ones, try Method 2 (manual copy-paste):")
        print("  python get_real_data.py paste BANK QUARTER")


def paste_transcript():
    """Method 2: Paste transcript from clipboard."""
    if len(sys.argv) < 4:
        print("Usage: python get_real_data.py paste BANK QUARTER")
        print("Example: python get_real_data.py paste RBC Q1_2024")
        print(f"\nBanks: {', '.join(BANKS.keys())}")
        print(f"Quarters: {', '.join(QUARTERS)}")
        return

    bank = sys.argv[2].upper()
    quarter = sys.argv[3]

    if bank not in BANKS:
        print(f"❌ Unknown bank: {bank}. Use one of: {', '.join(BANKS.keys())}")
        return
    if quarter not in QUARTERS:
        print(f"❌ Unknown quarter: {quarter}. Use one of: {', '.join(QUARTERS)}")
        return

    print(f"\n📋 Paste the {bank} {quarter} earnings call transcript below.")
    print("   When done, press Enter TWICE (two empty lines) to finish.\n")
    print("-" * 50)

    lines = []
    empty_count = 0

    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
            lines.append(line)
        except EOFError:
            break

    text = "\n".join(lines).strip()

    if len(text) < 200:
        print(f"❌ Text too short ({len(text)} chars). Please paste the full transcript.")
        return

    save_transcript(bank, quarter, text, source="manual_paste")
    print(f"\n🎉 Done! Now run: python run.py process")


def load_pdf():
    """Method 3: Load transcript from a PDF file."""
    if len(sys.argv) < 5:
        print("Usage: python get_real_data.py pdf BANK QUARTER path/to/file.pdf")
        print("Example: python get_real_data.py pdf CIBC Q1_2024 ~/Downloads/transcript.pdf")
        return

    bank = sys.argv[2].upper()
    quarter = sys.argv[3]
    pdf_path = sys.argv[4]

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    try:
        import PyPDF2
    except ImportError:
        print("Installing PyPDF2...")
        os.system("pip install PyPDF2")
        import PyPDF2

    print(f"📄 Reading PDF: {pdf_path}")
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    if len(text) < 200:
        print(f"❌ Could not extract enough text from PDF ({len(text)} chars)")
        return

    save_transcript(bank, quarter, text, source=f"pdf:{pdf_path}")
    print(f"\n🎉 Done! Now run: python run.py process")


def check_status():
    """Show what transcripts we have."""
    print("\n📊 Current Data Status")
    print("=" * 60)

    real_count = 0
    sample_count = 0
    missing_count = 0

    for bank in BANKS:
        print(f"\n🏦 {bank}")
        for quarter in QUARTERS:
            filepath = RAW_DIR / f"{bank}_{quarter}.txt"
            if filepath.exists():
                text = filepath.read_text()
                word_count = len(text.split())
                # Check if it's sample data (sample data has very specific patterns)
                is_sample = "sample" in text.lower()[:100] or word_count < 800
                if is_sample or word_count < 800:
                    print(f"  ⚠️  {quarter}: {word_count:,} words (likely SAMPLE data)")
                    sample_count += 1
                else:
                    print(f"  ✅ {quarter}: {word_count:,} words (REAL data)")
                    real_count += 1
            else:
                print(f"  ❌ {quarter}: MISSING")
                missing_count += 1

    print(f"\n{'=' * 60}")
    print(f"Real transcripts: {real_count}")
    print(f"Sample data: {sample_count}")
    print(f"Missing: {missing_count}")

    if sample_count > 0 or missing_count > 0:
        print(f"\n💡 To get real data, run: python get_real_data.py auto")


# ── Main CLI ────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands:")
        print("  python get_real_data.py auto              # Auto-scrape from Motley Fool")
        print("  python get_real_data.py paste RBC Q1_2024  # Paste transcript manually")
        print("  python get_real_data.py pdf CIBC Q1_2024 file.pdf  # Load from PDF")
        print("  python get_real_data.py status             # Check what data you have")
        return

    command = sys.argv[1].lower()

    if command == "auto":
        auto_scrape()
    elif command == "paste":
        paste_transcript()
    elif command == "pdf":
        load_pdf()
    elif command == "status":
        check_status()
    else:
        print(f"Unknown command: {command}")
        print("Use: auto, paste, pdf, or status")


if __name__ == "__main__":
    main()
