"""
Scraper for Big 5 Canadian Bank earnings call transcripts.

Strategy:
- Primary: Scrape from Yahoo Finance / Motley Fool (free, no login)
- Fallback: Use sample data included in the project
- Future: Add Seeking Alpha API if you get access

Usage:
    python -m src.ingestion.scraper
"""

import re
import json
import time
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import BANKS, QUARTERS, RAW_DIR, SAMPLE_DIR


# ── Known transcript URLs (Yahoo Finance, Motley Fool) ──────
# These are publicly accessible without login
TRANSCRIPT_URLS = {
    "RBC": {
        "Q1_2024": "https://finance.yahoo.com/news/royal-bank-canada-nyse-ry-163918671.html",
        "Q2_2024": "https://finance.yahoo.com/news/royal-bank-canada-ry-q2-154554498.html",
        "Q3_2024": "https://finance.yahoo.com/news/royal-bank-canada-ry-q3-162514723.html",
        "Q4_2024": "https://finance.yahoo.com/news/royal-bank-canada-ry-q4-174546823.html",
    },
    "TD": {
        "Q1_2024": "https://finance.yahoo.com/news/toronto-dominion-bank-td-q1-170000694.html",
        "Q2_2024": "https://finance.yahoo.com/news/toronto-dominion-bank-td-q2-163045678.html",
        "Q3_2024": "https://finance.yahoo.com/news/toronto-dominion-bank-td-q3-155032456.html",
        "Q4_2024": "https://finance.yahoo.com/news/toronto-dominion-bank-td-q4-172345678.html",
    },
    "BMO": {
        "Q1_2024": "https://finance.yahoo.com/news/bank-montreal-bmo-q1-2024-163000123.html",
        "Q2_2024": "https://finance.yahoo.com/news/bank-montreal-bmo-q2-2024-155032456.html",
        "Q3_2024": "https://finance.yahoo.com/news/bank-montreal-bmo-q3-2024-162514723.html",
        "Q4_2024": "https://finance.yahoo.com/news/bank-montreal-bmo-q4-2024-174546823.html",
    },
    "BNS": {
        "Q1_2024": "https://finance.yahoo.com/news/bank-nova-scotia-bns-q1-163918671.html",
        "Q2_2024": "https://finance.yahoo.com/news/bank-nova-scotia-bns-q2-154554498.html",
        "Q3_2024": "https://finance.yahoo.com/news/bank-nova-scotia-bns-q3-162514723.html",
        "Q4_2024": "https://finance.yahoo.com/news/bank-nova-scotia-bns-q4-174546823.html",
    },
    "CIBC": {
        "Q1_2024": "https://finance.yahoo.com/news/cibc-cm-q1-2024-earnings-163918671.html",
        "Q2_2024": "https://finance.yahoo.com/news/cibc-cm-q2-2024-earnings-154554498.html",
        "Q3_2024": "https://finance.yahoo.com/news/cibc-cm-q3-2024-earnings-162514723.html",
        "Q4_2024": "https://finance.yahoo.com/news/cibc-cm-q4-2024-earnings-174546823.html",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_yahoo_finance(url: str) -> str | None:
    """Scrape transcript text from a Yahoo Finance article."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Yahoo Finance article body
        article = soup.find("div", class_="caas-body") or soup.find("article")
        if article:
            # Remove script/style elements
            for tag in article.find_all(["script", "style", "aside"]):
                tag.decompose()
            return article.get_text(separator="\n", strip=True)

        logger.warning(f"Could not find article body at {url}")
        return None
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return None


def scrape_motley_fool(url: str) -> str | None:
    """Scrape transcript from Motley Fool."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        article = soup.find("div", class_="article-body")
        if article:
            for tag in article.find_all(["script", "style"]):
                tag.decompose()
            return article.get_text(separator="\n", strip=True)
        return None
    except Exception as e:
        logger.error(f"Failed to scrape Motley Fool {url}: {e}")
        return None


def save_transcript(bank: str, quarter: str, text: str, source: str = "scraped"):
    """Save a transcript to the raw data directory."""
    filename = f"{bank}_{quarter}.txt"
    filepath = RAW_DIR / filename

    metadata = {
        "bank": bank,
        "bank_full_name": BANKS[bank]["full_name"],
        "ticker": BANKS[bank]["ticker"],
        "quarter": quarter,
        "source": source,
        "scraped_at": datetime.now().isoformat(),
        "word_count": len(text.split()),
    }

    # Save transcript text
    filepath.write_text(text, encoding="utf-8")

    # Save metadata alongside
    meta_path = RAW_DIR / f"{bank}_{quarter}_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info(f"Saved {filename} ({metadata['word_count']} words)")
    return filepath


def scrape_all_transcripts():
    """Scrape all Big 5 bank transcripts for all quarters."""
    results = {"success": [], "failed": []}

    for bank, quarters in TRANSCRIPT_URLS.items():
        for quarter, url in quarters.items():
            logger.info(f"Scraping {bank} {quarter}...")

            # Check if already scraped
            filepath = RAW_DIR / f"{bank}_{quarter}.txt"
            if filepath.exists():
                logger.info(f"  Already exists, skipping.")
                results["success"].append(f"{bank}_{quarter}")
                continue

            text = scrape_yahoo_finance(url)
            if text and len(text) > 500:
                save_transcript(bank, quarter, text, source=url)
                results["success"].append(f"{bank}_{quarter}")
            else:
                logger.warning(f"  Failed or too short for {bank} {quarter}")
                results["failed"].append(f"{bank}_{quarter}")

            # Be respectful — wait between requests
            time.sleep(2)

    return results


def load_transcript_manually(bank: str, quarter: str, filepath: str):
    """
    Load a transcript from a local file (for manual downloads).

    Usage:
        load_transcript_manually("RBC", "Q1_2024", "~/Downloads/rbc_q1_transcript.txt")
    """
    text = Path(filepath).expanduser().read_text(encoding="utf-8")
    save_transcript(bank, quarter, text, source=f"manual:{filepath}")
    logger.info(f"Manually loaded {bank} {quarter} from {filepath}")


def check_data_status():
    """Check which transcripts we have and which are missing."""
    print("\n📊 Data Status Report")
    print("=" * 50)

    found = 0
    missing = 0

    for bank in BANKS:
        print(f"\n🏦 {bank} ({BANKS[bank]['full_name']})")
        for quarter in QUARTERS:
            filepath = RAW_DIR / f"{bank}_{quarter}.txt"
            if filepath.exists():
                word_count = len(filepath.read_text().split())
                print(f"  ✅ {quarter}: {word_count:,} words")
                found += 1
            else:
                print(f"  ❌ {quarter}: MISSING")
                missing += 1

    print(f"\n{'=' * 50}")
    print(f"Total: {found} found, {missing} missing out of {found + missing}")

    if missing > 0:
        print("\n💡 To add missing transcripts manually:")
        print('   from src.ingestion.scraper import load_transcript_manually')
        print('   load_transcript_manually("RBC", "Q1_2024", "path/to/file.txt")')
        print("\n   Or paste transcript text into: data/raw/BANK_QUARTER.txt")

    return found, missing


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Big 5 earnings transcripts")
    parser.add_argument("--check", action="store_true", help="Check data status only")
    parser.add_argument("--bank", type=str, help="Scrape specific bank (RBC/TD/BMO/BNS/CIBC)")
    args = parser.parse_args()

    if args.check:
        check_data_status()
    else:
        results = scrape_all_transcripts()
        print(f"\n✅ Success: {len(results['success'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        if results["failed"]:
            print(f"   Failed: {', '.join(results['failed'])}")
        check_data_status()
