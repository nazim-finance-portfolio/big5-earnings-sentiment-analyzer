#  Big 5 Canadian Bank Earnings Sentiment Analyzer

NLP-powered sentiment analysis of Canada's Big 5 bank earnings calls with **forward guidance detection** and **RAG-powered Q&A**.

##  Key Finding

> **All 5 Canadian banks show a negative guidance gap.** Management teams are consistently more positive about reported results (0.55 avg sentiment) than their forward outlook (0.39 avg sentiment) — a -0.16 gap that signals caution despite strong earnings.

##  Bank-by-Bank Results

| Bank | Avg Sentiment | Guidance Gap | Signal |
|------|:---:|:---:|:---:|
| **TD** | 0.517 | -0.10 | CONFIDENT |
| **BMO** | 0.507 | -0.12 | CONFIDENT |
| **BNS** | 0.393 | -0.18 | CAUTIOUS |
| **RBC** | 0.379 | -0.29 | HEDGING |
| **CIBC** | 0.269 | -0.06 | WEAKEST |

##  What It Does

- Analyzes **40 earnings call transcripts** across **8 quarters** (Q1 2024 – Q4 2025)
- Scores sentiment using **VADER NLP** on **1,547 text chunks**
- Classifies each chunk as **forward-looking** or **backward-looking** using **35+ regex patterns**
- Computes a **guidance gap** metric (forward sentiment – backward sentiment)
- Delivers insights through a **5-tab interactive Streamlit dashboard**
- Lets users ask natural language questions via **RAG-powered Q&A with OpenAI GPT-4o**

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Sentiment Analysis | VADER |
| Forward Guidance | Regex NLP (35+ patterns) |
| Dashboard | Streamlit + Plotly |
| RAG / LLM | OpenAI GPT-4o |
| Data Processing | Pandas |

##  Project Structure

```
big5-earnings-analyzer/
├── config/
│   └── settings.py              # Configuration
├── data/
│   ├── raw/                     # Raw transcript files
│   └── processed/               # Sentiment results & metrics
├── src/
│   ├── analysis/
│   │   ├── vader_analyzer.py    # VADER sentiment scoring
│   │   ├── forward_guidance.py  # Forward/backward classification
│   │   └── metrics.py           # Aggregated metrics
│   ├── dashboard/
│   │   └── app.py               # Streamlit dashboard (5 tabs)
│   ├── ingestion/
│   │   ├── preprocessor.py      # Text chunking & section detection
│   │   └── scraper.py           # Transcript ingestion
│   └── rag/
│       ├── agent.py             # RAG Q&A with GPT-4o
│       ├── embeddings.py        # Text embeddings
│       └── retriever.py         # Chunk retrieval
├── run.py                       # Main entry point
├── requirements.txt
└── README.md
```

##  Quick Start

```bash
# Clone
git clone https://github.com/nazim-finance-portfolio/big5-earnings-sentiment-analyzer.git
cd big5-earnings-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Set up API key
# Create .env file with: OPENAI_API_KEY=your-key-here

# Run the pipeline
python run.py analyze        # Run sentiment analysis
python run.py dashboard      # Launch Streamlit dashboard
```

## Dashboard Features

| Tab | Description |
|-----|-------------|
| **Overview** | KPI cards + sentiment heatmap across all banks/quarters |
| **Trends** | Line chart tracking sentiment movement over 8 quarters |
| **Comparison** | Bar chart + metrics table ranking all 5 banks |
| **Forward Guidance** | Forward vs backward sentiment, guidance gap, content ratios |
| **Ask AI (RAG)** | Natural language Q&A powered by GPT-4o with source citations |

##  Coverage

| Bank | Ticker | Quarters |
|------|--------|----------|
| Royal Bank of Canada | RY | Q1 2024 – Q4 2025 |
| Toronto-Dominion Bank | TD | Q1 2024 – Q4 2025 |
| Bank of Montreal | BMO | Q1 2024 – Q4 2025 |
| Bank of Nova Scotia | BNS | Q1 2024 – Q4 2025 |
| CIBC | CM | Q1 2024 – Q4 2025 |

##  License

MIT License — Built by **Biplob** | MSc Big Data Financial Analytics, Trent University | CSC | FMVA
