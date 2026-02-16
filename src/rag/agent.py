"""
Multi-agent RAG system for earnings call analysis.

Supports both OpenAI (GPT) and Anthropic (Claude) APIs.

Architecture:
1. Retriever Agent: Finds relevant chunks from ChromaDB
2. Analyst Agent: Analyzes sentiment and key themes
3. Synthesizer Agent: Combines analysis into coherent response
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    MAX_TOKENS, TOP_K_RESULTS
)
from src.rag.embeddings import EarningsVectorStore


def get_llm_response(system_prompt: str, user_prompt: str) -> str:
    """Get response from LLM (OpenAI or Anthropic)."""

    if LLM_PROVIDER == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error calling OpenAI API: {e}"

    elif LLM_PROVIDER == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error calling Anthropic API: {e}"

    else:
        return "No LLM provider configured. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env"


# ── Agent System Prompts ─────────────────────────────────

RETRIEVER_PROMPT = """You are a financial document retrieval specialist focusing on Canadian banking.
Your role is to analyze a user's question and determine the most relevant search queries
to find information in earnings call transcripts from Canada's Big 5 banks
(RBC, TD, BMO, Scotiabank/BNS, CIBC).

Given the user's question, generate 2-3 focused search queries that would help find
the most relevant transcript segments. Consider:
- Specific banks mentioned
- Financial metrics or topics discussed
- Time periods referenced
- Sentiment-related queries

Return your queries as a JSON array of strings."""

ANALYST_PROMPT = """You are a senior Canadian banking analyst specializing in earnings call analysis.
You have deep expertise in analyzing the Big 5 Canadian banks: RBC, TD, BMO, Scotiabank, and CIBC.

Given transcript excerpts from earnings calls, provide insightful analysis including:
- Key financial themes and metrics discussed
- Management tone and sentiment assessment
- Comparison between banks where relevant
- Notable risks or opportunities mentioned
- Forward guidance and outlook

Be specific, cite the transcript segments, and provide actionable insights.
Use a professional, analytical tone appropriate for institutional investors."""

SYNTHESIZER_PROMPT = """You are a financial communications specialist who creates clear,
well-structured summaries of complex banking analysis.

Your role is to take raw analysis and search results and synthesize them into
a cohesive, well-organized response that directly answers the user's question.

Guidelines:
- Lead with the direct answer to the question
- Support with specific data points from transcripts
- Note any significant differences between banks
- Highlight sentiment patterns (positive/negative shifts)
- Keep responses focused and concise (aim for 200-400 words)
- Use professional financial terminology appropriate for an informed audience"""


class EarningsRAGAgent:
    """Multi-agent RAG system for earnings call Q&A."""

    def __init__(self):
        """Initialize the RAG agent with vector store."""
        self.vector_store = EarningsVectorStore()
        self.has_api_key = bool(OPENAI_API_KEY or ANTHROPIC_API_KEY)

        if self.has_api_key:
            logger.info(f"RAG Agent initialized with {LLM_PROVIDER} ({LLM_MODEL})")
        else:
            logger.warning("No API key found. RAG Q&A will not work.")

    def retrieve(self, query: str, bank: str = None, quarter: str = None,
                 n_results: int = TOP_K_RESULTS) -> list[dict]:
        """Retrieve relevant chunks from vector store."""
        where_filter = {}
        if bank:
            where_filter["bank"] = bank
        if quarter:
            where_filter["quarter"] = quarter

        results = self.vector_store.query(
            query_text=query,
            n_results=n_results,
            where=where_filter if where_filter else None,
        )
        return results

    def analyze(self, query: str, context_chunks: list[dict]) -> str:
        """Have the analyst agent analyze the retrieved chunks."""
        if not self.has_api_key:
            return "API key required for AI analysis."

        # Format context
        context = "\n\n---\n\n".join([
            f"[{c.get('bank', '?')} {c.get('quarter', '?')} | {c.get('section', '?')}]\n{c.get('text', '')}"
            for c in context_chunks
        ])

        user_prompt = f"""Question: {query}

Relevant transcript excerpts:
{context}

Please analyze these excerpts to answer the question. Focus on specific data points,
management commentary, and sentiment patterns."""

        return get_llm_response(ANALYST_PROMPT, user_prompt)

    def synthesize(self, query: str, analysis: str, chunks: list[dict]) -> str:
        """Have the synthesizer create a final response."""
        if not self.has_api_key:
            return analysis

        user_prompt = f"""Original question: {query}

Analysis from our banking analyst:
{analysis}

Number of transcript segments reviewed: {len(chunks)}
Banks covered: {', '.join(set(c.get('bank', '?') for c in chunks))}
Quarters covered: {', '.join(set(c.get('quarter', '?') for c in chunks))}

Please synthesize this into a clear, direct response to the user's question."""

        return get_llm_response(SYNTHESIZER_PROMPT, user_prompt)

    def ask(self, query: str, bank: str = None, quarter: str = None) -> dict:
        """
        Full RAG pipeline: Retrieve → Analyze → Synthesize.

        Returns dict with answer, sources, and metadata.
        """
        if not self.has_api_key:
            return {
                "answer": f"RAG system requires an API key. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file.",
                "sources": [],
                "metadata": {"error": "no_api_key"},
            }

        logger.info(f"Processing query: {query[:80]}...")

        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(query, bank=bank, quarter=quarter, n_results=TOP_K_RESULTS)

        if not chunks:
            return {
                "answer": "No relevant transcript data found. Please check that transcripts have been indexed.",
                "sources": [],
                "metadata": {"error": "no_results"},
            }

        # Step 2: Analyze with context
        analysis = self.analyze(query, chunks)

        # Step 3: Synthesize final response
        final_answer = self.synthesize(query, analysis, chunks)

        # Prepare source citations
        sources = [
            {
                "bank": c.get("bank", "Unknown"),
                "quarter": c.get("quarter", "Unknown"),
                "section": c.get("section", "Unknown"),
                "preview": c.get("text", "")[:150] + "...",
            }
            for c in chunks
        ]

        return {
            "answer": final_answer,
            "sources": sources,
            "metadata": {
                "provider": LLM_PROVIDER,
                "model": LLM_MODEL,
                "chunks_retrieved": len(chunks),
                "banks_covered": list(set(c.get("bank", "") for c in chunks)),
                "quarters_covered": list(set(c.get("quarter", "") for c in chunks)),
            },
        }


if __name__ == "__main__":
    agent = EarningsRAGAgent()

    # Test query
    result = agent.ask("How did RBC perform in Q4 2024? What was the CEO's tone?")
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['bank']} {s['quarter']} ({s['section']})")
