"""
Streamlit Dashboard for Big 5 Canadian Bank Earnings Sentiment Analyzer.
V3 - With Forward Guidance Analysis Tab & Secure API Loading
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════
#  1. API KEY CONFIGURATION (Securely loads from .env)
# ══════════════════════════════════════════════════════════
# Load environment variables from the .env file in the project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, BANKS, QUARTERS, BANK_COLORS, SENTIMENT_COLORS


BANK_ALIASES = {
    "RBC": ["rbc", "royal bank", "royal bank of canada"],
    "TD": ["td", "td bank", "toronto-dominion", "toronto dominion"],
    "BMO": ["bmo", "bank of montreal"],
    "BNS": ["bns", "scotiabank", "bank of nova scotia", "scotia"],
    "CIBC": ["cibc", "canadian imperial", "imperial bank"],
}

STOPWORDS = {
    "what", "were", "the", "key", "how", "did", "for", "and",
    "that", "this", "with", "from", "about", "have", "has",
    "their", "they", "which", "when", "where", "who", "all",
    "bank", "banks", "report", "reported", "quarter", "fiscal",
    "compare", "comparison", "between", "performance", "results",
    "can", "you", "tell", "please", "show", "give", "was",
}


class InternalAgent:
    def __init__(self):
        # SECURE: Loads strictly from environment variables
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            self.client = None
            return
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception:
            self.client = None

    def query(self, user_query, context_text):
        if not self.client:
            return "OpenAI API Key is missing. Please add OPENAI_API_KEY to your .env file."
        
        prompt = """You are a senior Canadian banking analyst. Answer based ONLY on the transcript excerpts below. 
Cite specific banks and quarters. If info is missing, say so. Be concise (200-400 words)."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Transcripts:\n" + context_text + "\n\nQuestion: " + user_query}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return "Error: " + str(e)


st.set_page_config(page_title="Big 5 Bank Earnings Analyzer", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .guidance-positive { color: #27ae60; font-weight: bold; }
    .guidance-negative { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sentiment_data():
    path = PROCESSED_DIR / "combined_sentiment.csv"
    if path.exists():
        return pd.read_csv(path)
    path = PROCESSED_DIR / "vader_sentiment_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["combined_score"] = df["compound"]
        df["combined_label"] = df["label"]
        return df
    return pd.DataFrame()


@st.cache_data
def load_metrics():
    metrics = {}
    for name in ["by_bank_quarter", "by_bank", "by_section", "by_bank_section"]:
        path = PROCESSED_DIR / f"metrics_{name}.csv"
        if path.exists():
            metrics[name] = pd.read_csv(path)
    path = PROCESSED_DIR / "metrics_overall.json"
    if path.exists():
        metrics["overall"] = json.loads(path.read_text())
    return metrics


@st.cache_data
def load_forward_guidance():
    """Load forward guidance analysis results."""
    fg_metrics = PROCESSED_DIR / "forward_guidance_metrics.csv"
    fg_results = PROCESSED_DIR / "forward_guidance_results.csv"
    fg_summary = PROCESSED_DIR / "forward_guidance_summary.json"

    data = {}
    if fg_metrics.exists():
        data["metrics"] = pd.read_csv(fg_metrics)
    if fg_results.exists():
        data["results"] = pd.read_csv(fg_results)
    if fg_summary.exists():
        data["summary"] = json.loads(fg_summary.read_text())
    return data


def render_sidebar():
    st.sidebar.markdown("# 🏦 Filters")
    banks = st.sidebar.multiselect("Select Banks", options=list(BANKS.keys()), default=list(BANKS.keys()), format_func=lambda x: f"{x} — {BANKS[x]['full_name']}")
    quarters = st.sidebar.multiselect("Select Quarters", options=QUARTERS, default=QUARTERS)
    sections = st.sidebar.multiselect("Filter by Section", options=["ceo_remarks", "cfo_remarks", "cro_remarks", "qa_session", "general"], default=["ceo_remarks", "cfo_remarks", "cro_remarks", "qa_session", "general"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("AI-powered sentiment analysis of Canada's Big 5 bank earnings calls with forward guidance detection.")
    st.sidebar.markdown("**Built by Biplob**")
    return banks, quarters, sections


def render_header():
    st.markdown('<p class="main-header">🏦 Big 5 Canadian Bank Earnings Sentiment Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered NLP analysis of RBC, TD, BMO, Scotiabank & CIBC earnings calls</p>', unsafe_allow_html=True)


def render_kpi_cards(df, metrics):
    overall = metrics.get("overall", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg = overall.get("avg_sentiment", df["combined_score"].mean() if not df.empty else 0)
        st.metric("Overall Sentiment", f"{avg:.3f}")
    with col2:
        st.metric("Most Positive", overall.get("most_positive_bank", "N/A"))
    with col3:
        st.metric("Most Negative", overall.get("most_negative_bank", "N/A"))
    with col4:
        st.metric("Transcripts", overall.get("total_transcripts", 0))
    if not df.empty:
        st.markdown("---")
        st.markdown("### Sentiment Heatmap")
        pivot = df.groupby(["bank", "quarter"])["combined_score"].mean().reset_index()
        pivot_table = pivot.pivot(index="bank", columns="quarter", values="combined_score")
        fig = px.imshow(pivot_table, text_auto=".2f", aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, labels=dict(x="Quarter", y="Bank", color="Sentiment"))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_sentiment_trend(df, banks):
    st.markdown("### Sentiment Trends")
    if df.empty:
        st.warning("No data to display.")
        return
    trend = df[df["bank"].isin(banks)].groupby(["bank", "quarter"])["combined_score"].mean().reset_index()
    fig = px.line(trend, x="quarter", y="combined_score", color="bank", markers=True, color_discrete_map=BANK_COLORS)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_bank_comparison(df, banks):
    st.markdown("### Bank Comparison")
    if df.empty:
        st.warning("No data to display.")
        return
    filtered = df[df["bank"].isin(banks)]
    bank_stats = filtered.groupby("bank")["combined_score"].mean().sort_values().reset_index()
    fig = px.bar(bank_stats, x="combined_score", y="bank", orientation="h", color="bank", color_discrete_map=BANK_COLORS)
    st.plotly_chart(fig, use_container_width=True)
    summary = filtered.groupby("bank").agg(
        avg_sentiment=("combined_score", "mean"),
        pct_positive=("combined_label", lambda x: (x == "positive").mean()),
        pct_negative=("combined_label", lambda x: (x == "negative").mean()),
        volatility=("combined_score", "std"),
    ).round(3).reset_index()
    summary.columns = ["Bank", "Avg Sentiment", "% Positive", "% Negative", "Volatility"]
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
#  FORWARD GUIDANCE TAB
# ══════════════════════════════════════════════════════════

def render_forward_guidance(banks):
    """Render the Forward Guidance analysis tab."""
    st.markdown("### Forward Guidance Sentiment Analysis")
    st.markdown("*Separating backward-looking results from forward-looking outlook — because markets price expectations, not history.*")

    fg = load_forward_guidance()

    if fg.get("metrics") is None or not fg.get("summary"):
        st.warning("Forward guidance data not found. Run: python run.py analyze")
        return

    metrics = fg["metrics"]
    summary = fg["summary"]

    if banks:
        metrics = metrics[metrics["bank"].isin(banks)]

    # KPI Cards
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fwd_pct = round(summary.get("forward_chunks", 0) / max(summary.get("total_chunks", 1), 1) * 100, 1)
        st.metric("Forward-Looking Chunks", str(fwd_pct) + "%")
    with col2:
        st.metric("Forward Sentiment", str(summary.get("avg_forward_sentiment", 0)))
    with col3:
        st.metric("Backward Sentiment", str(summary.get("avg_backward_sentiment", 0)))
    with col4:
        st.metric("Most Forward-Looking", summary.get("most_forward_looking_bank", "N/A"))
    with col5:
        st.metric("Most Optimistic Outlook", summary.get("most_optimistic_outlook_bank", "N/A"))

    st.markdown("---")

    # Chart 1: Forward vs Backward Sentiment by Bank
    st.markdown("#### Forward vs Backward Sentiment by Bank")
    bank_agg = metrics.groupby("bank").agg(
        forward_sentiment=("forward_sentiment", "mean"),
        backward_sentiment=("backward_sentiment", "mean")
    ).round(4).reset_index()

    if not bank_agg.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Backward (Results)",
            x=bank_agg["bank"],
            y=bank_agg["backward_sentiment"],
            marker_color="#3498db",
        ))
        fig.add_trace(go.Bar(
            name="Forward (Outlook)",
            x=bank_agg["bank"],
            y=bank_agg["forward_sentiment"],
            marker_color="#2ecc71",
        ))
        fig.update_layout(barmode="group", height=400, yaxis_title="Avg Sentiment Score")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Guidance Gap Over Time
    st.markdown("#### Guidance Gap Over Time (Forward - Backward Sentiment)")
    if not metrics.empty and "guidance_gap" in metrics.columns:
        fig = px.line(
            metrics,
            x="quarter",
            y="guidance_gap",
            color="bank",
            markers=True,
            color_discrete_map=BANK_COLORS,
            labels={"guidance_gap": "Guidance Gap (Fwd - Bwd)"}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Neutral")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 3: Forward-Looking Content Ratio
    st.markdown("#### Forward-Looking Content Ratio by Bank/Quarter")
    if not metrics.empty and "forward_pct" in metrics.columns:
        fig = px.bar(
            metrics,
            x="quarter",
            y="forward_pct",
            color="bank",
            barmode="group",
            color_discrete_map=BANK_COLORS,
            labels={"forward_pct": "% Forward-Looking"}
        )
        fig.update_layout(height=400, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 4: Guidance Heatmap
    st.markdown("#### Forward Guidance Sentiment Heatmap")
    if not metrics.empty and "forward_sentiment" in metrics.columns:
        fwd_pivot = metrics.pivot_table(index="bank", columns="quarter", values="forward_sentiment")
        if not fwd_pivot.empty:
            fig = px.imshow(
                fwd_pivot,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0.4,
                labels=dict(x="Quarter", y="Bank", color="Forward Sentiment"),
            )
            fig.update_layout(height=300, title="Forward-Looking Sentiment Only")
            st.plotly_chart(fig, use_container_width=True)

    # Summary Table
    st.markdown("#### Detailed Metrics Table")
    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
#  ASK AI (RAG) - SMART SEARCH
# ══════════════════════════════════════════════════════════

def detect_banks_in_query(query):
    query_lower = query.lower()
    detected = []
    for bank_code, aliases in BANK_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            detected.append(bank_code)
    return detected


def detect_quarters_in_query(query):
    query_lower = query.lower()
    detected = []
    matches = re.findall(r'q(\d)\s*[\-_]?\s*(20\d{2})', query_lower)
    for q_num, year in matches:
        quarter = "Q" + q_num + "_" + year
        if quarter in QUARTERS:
            detected.append(quarter)
    year_matches = re.findall(r'\b(20\d{2})\b', query_lower)
    for year in year_matches:
        for q in QUARTERS:
            if year in q and q not in detected:
                detected.append(q)
    return detected


def render_rag_chat():
    st.markdown("### Ask Questions About Earnings")
    st.markdown("*Powered by multi-agent RAG with OpenAI*")

    # CHECK FOR KEY IN ENVIRONMENT (Not hardcoded!)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API Key is missing. Please add it to your .env file.")
        return

    agent = InternalAgent()
    if not agent.client:
        st.error("Could not connect to OpenAI. Check your API key.")
        return

    total_chunks = 0
    for f in PROCESSED_DIR.glob("*_processed.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "chunks" in data:
                total_chunks += len(data["chunks"])
        except Exception:
            pass
    st.info("Loaded " + str(total_chunks) + " transcript chunks across " + str(len(QUARTERS)) + " quarters")

    col1, col2 = st.columns([3, 1])
    with col2:
        bank_filter = st.selectbox("Filter by bank", ["All"] + list(BANKS.keys()))
        quarter_filter = st.selectbox("Filter by quarter", ["All"] + QUARTERS)
    with col1:
        question = st.text_input("Your question:", placeholder="e.g., How did CIBC perform in Q1 2025?")

    st.markdown("**Try these:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("Compare banks Q4 2024"):
            question = "Compare the performance and outlook of all five banks in Q4 2024"
    with ex2:
        if st.button("Key risks mentioned?"):
            question = "What are the main risks and challenges mentioned by the banks?"
    with ex3:
        if st.button("Forward guidance outlook"):
            question = "What is the forward guidance and outlook for 2025 across the banks?"

    if question:
        with st.spinner("Searching transcripts..."):
            detected_banks = detect_banks_in_query(question)
            detected_quarters = detect_quarters_in_query(question)

            if bank_filter != "All":
                detected_banks = [bank_filter]
            if quarter_filter != "All":
                detected_quarters = [quarter_filter]

            all_aliases = []
            for aliases in BANK_ALIASES.values():
                all_aliases.extend(aliases)
            keywords = []
            for w in question.split():
                clean_w = w.lower().strip("?.,!")
                if len(clean_w) > 2 and clean_w not in STOPWORDS and clean_w not in all_aliases:
                    keywords.append(clean_w)

            source_info = []
            context_text = ""
            try:
                files = list(PROCESSED_DIR.glob("*_processed.json"))
                relevant_chunks = []

                for f in files:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if not isinstance(data, dict) or "chunks" not in data:
                        continue

                    file_bank = data.get("bank", "")
                    file_quarter = data.get("quarter", "")

                    if detected_banks and file_bank not in detected_banks:
                        continue
                    if detected_quarters and file_quarter not in detected_quarters:
                        continue

                    for chunk in data["chunks"]:
                        text = chunk.get("text", "")
                        if not text: continue

                        text_lower = text.lower()
                        keyword_score = sum(1 for k in keywords if k in text_lower)
                        
                        if keyword_score > 0:
                            relevant_chunks.append({
                                "score": keyword_score,
                                "text": text,
                                "bank": chunk.get("bank", file_bank),
                                "quarter": chunk.get("quarter", file_quarter),
                                "section": chunk.get("section", "general")
                            })

                relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
                top_chunks = relevant_chunks[:15]

                context_parts = []
                for ch in top_chunks:
                    label = "[" + ch["bank"] + " " + ch["quarter"] + " - " + ch["section"] + "]"
                    context_parts.append(label + "\n" + ch["text"])
                    source_info.append(ch)

                context_text = "\n\n---\n\n".join(context_parts)

            except Exception as e:
                st.error("Search error: " + str(e))
                context_text = ""

            if not context_text:
                st.info("No matching transcript segments found. Try different keywords or adjust filters.")
            else:
                answer = agent.query(question, context_text)
                st.markdown("#### Answer")
                st.markdown(answer)
                with st.expander("Sources (" + str(len(source_info)) + " transcript segments)"):
                    for i, ch in enumerate(source_info):
                        st.markdown("**" + str(i + 1) + ". " + ch["bank"] + " " + ch["quarter"] + "** (" + ch["section"] + ")")
                        st.caption(ch["text"][:300] + "...")
                        st.markdown("---")


def main():
    render_header()
    df = load_sentiment_data()
    metrics = load_metrics()

    if df.empty:
        st.error("No data found. Run: python run.py process && python run.py analyze")
        return

    banks, quarters, sections = render_sidebar()
    filtered = df[(df["bank"].isin(banks)) & (df["quarter"].isin(quarters)) & (df["section"].isin(sections))]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Trends", "Comparison", "Forward Guidance", "Ask AI (RAG)"
    ])

    with tab1: render_kpi_cards(filtered, metrics)
    with tab2: render_sentiment_trend(filtered, banks)
    with tab3: render_bank_comparison(filtered, banks)
    with tab4: render_forward_guidance(banks)
    with tab5: render_rag_chat()


if __name__ == "__main__":
    main()