"""
Retriever module — thin wrapper around embeddings.search for clean imports.
"""

from src.rag.embeddings import search, build_index, get_chroma_collection

__all__ = ["search", "build_index", "get_chroma_collection"]
