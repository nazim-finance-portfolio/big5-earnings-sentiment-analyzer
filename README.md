# 🏦 Big 5 Canadian Bank Earnings Sentiment Analyzer

An AI-powered multi-agent RAG system that analyzes earnings call transcripts from Canada's Big 5 banks using NLP sentiment analysis, vector search, and an interactive Streamlit dashboard.

## 🎯 What It Does

1. **Ingests** earnings call transcripts from RBC, TD, BMO, Scotiabank, and CIBC
2. **Analyzes sentiment** using VADER and transformer-based NLP models
3. **Builds a RAG system** — ask natural language questions about any bank's earnings
4. **Visualizes trends** on an interactive Streamlit dashboard

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Sentiment Analysis | VADER + FinBERT (transformer) |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers |
| RAG / LLM | Anthropic Claude API |
| Dashboard | Streamlit + Plotly |
| Data Processing | pandas, numpy |
| Web Scraping | BeautifulSoup4, requests |

## 📁 Project Structure

```
big5-earnings-analyzer/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── config/
│   └── settings.py          # Centralized configuration
├── data/
│   ├── raw/                  # Raw transcript text files
│   ├── processed/            # Cleaned & chunked data
│   └── sample/               # Sample data to get started quickly
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── scraper.py        # Scrapes transcripts from public sources
│   │   └── preprocessor.py   # Cleans and chunks transcripts
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── vader_analyzer.py # VADER sentiment scoring
│   │   ├── finbert_analyzer.py # FinBERT deep sentiment
│   │   └── metrics.py        # Combined sentiment metrics
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py     # Generate & store embeddings
│   │   ├── retriever.py      # Vector search retrieval
│   │   └── agent.py          # Multi-agent Q&A system
│   └── dashboard/
│       ├── __init__.py
│       └── app.py            # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb     # Data exploration notebook
└── run.py                    # Main entry point
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/big5-earnings-analyzer.git
cd big5-earnings-analyzer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run the Pipeline

```bash
# Step 1: Scrape transcripts (or use sample data)
python run.py ingest

# Step 2: Analyze sentiment
python run.py analyze

# Step 3: Build RAG index
python run.py index

# Step 4: Launch dashboard
python run.py dashboard
```

Or run everything at once:
```bash
python run.py all
```

## 📊 Features

### Sentiment Dashboard
- Bank-by-bank sentiment trends over quarters
- Comparative sentiment heatmap across all Big 5
- Key topic extraction per earnings call
- Sentiment breakdown by section (CEO remarks, CFO remarks, Q&A)

### RAG Q&A System
Ask questions like:
- *"How did RBC's CEO describe loan growth in Q4 2024?"*
- *"Compare TD and BMO's outlook on credit losses"*
- *"What risks did Scotiabank highlight about US operations?"*
- *"Which bank was most optimistic about 2025?"*

### Multi-Agent Architecture
- **Retriever Agent**: Finds relevant transcript chunks via vector search
- **Analyst Agent**: Interprets sentiment and financial context
- **Synthesizer Agent**: Combines findings into coherent answers

## 🏦 Banks Covered

| Bank | Ticker | Quarters |
|------|--------|----------|
| Royal Bank of Canada | RY | Q1-Q4 2024 |
| Toronto-Dominion Bank | TD | Q1-Q4 2024 |
| Bank of Montreal | BMO | Q1-Q4 2024 |
| Bank of Nova Scotia | BNS | Q1-Q4 2024 |
| CIBC | CM | Q1-Q4 2024 |

## 📝 License

MIT License — built by Biplob as a data science portfolio project.
