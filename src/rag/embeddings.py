"""
Embeddings Generator & ChromaDB Vector Store.

Converts transcript chunks into vector embeddings and stores them
in ChromaDB for fast similarity search (RAG retrieval).

Model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, CHROMA_DIR, EMBEDDING_MODEL


# Global model cache
_embedding_model = None
_chroma_client = None
_collection = None

COLLECTION_NAME = "big5_earnings"


def get_embedding_model() -> SentenceTransformer:
    """Load sentence transformer model (cached)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_chroma_collection():
    """Get or create ChromaDB collection (cached)."""
    global _chroma_client, _collection

    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Big 5 Canadian Bank earnings call chunks"},
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}': {_collection.count()} documents")

    return _collection


def build_index():
    """
    Build the full vector index from all processed transcripts.

    Reads all *_processed.json files, generates embeddings, and stores
    them in ChromaDB with metadata for filtering.
    """
    model = get_embedding_model()
    collection = get_chroma_collection()

    # Check if already indexed
    if collection.count() > 0:
        logger.info(f"Collection already has {collection.count()} documents. Rebuilding...")
        # Delete existing to rebuild
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_client.delete_collection(COLLECTION_NAME)
        collection = _chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Big 5 Canadian Bank earnings call chunks"},
        )

    # Collect all chunks
    all_chunks = []
    for json_file in sorted(PROCESSED_DIR.glob("*_processed.json")):
        data = json.loads(json_file.read_text())
        all_chunks.extend(data["chunks"])

    if not all_chunks:
        logger.error("No processed chunks found. Run preprocessing first.")
        return

    logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")

    # Process in batches
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "bank": c["bank"],
                "bank_full_name": c["bank_full_name"],
                "quarter": c["quarter"],
                "section": c["section"],
                "speaker": c["speaker"] or "unknown",
                "word_count": c["word_count"],
            }
            for c in batch
        ]

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Add to ChromaDB
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        if (i + batch_size) % 200 == 0 or i + batch_size >= len(all_chunks):
            logger.info(f"  Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

    logger.info(f"✅ Index built: {collection.count()} documents in ChromaDB")


def search(
    query: str,
    top_k: int = 5,
    bank_filter: str = None,
    quarter_filter: str = None,
    section_filter: str = None,
) -> list[dict]:
    """
    Search the vector index for relevant transcript chunks.

    Args:
        query: Natural language question
        top_k: Number of results to return
        bank_filter: Filter by bank code (e.g., "RBC")
        quarter_filter: Filter by quarter (e.g., "Q4_2024")
        section_filter: Filter by section (e.g., "qa_session")

    Returns:
        List of dicts with text, metadata, and distance score
    """
    model = get_embedding_model()
    collection = get_chroma_collection()

    if collection.count() == 0:
        logger.error("Index is empty. Run build_index() first.")
        return []

    # Generate query embedding
    query_embedding = model.encode([query]).tolist()

    # Build metadata filter
    where_filter = {}
    if bank_filter:
        where_filter["bank"] = bank_filter
    if quarter_filter:
        where_filter["quarter"] = quarter_filter
    if section_filter:
        where_filter["section"] = section_filter

    # Search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": round(results["distances"][0][i], 4),
            "relevance": round(1 - results["distances"][0][i], 4),  # Higher = more relevant
        })

    return formatted


if __name__ == "__main__":
    build_index()

    # Test search
    print("\n🔍 Test search: 'credit losses and provisions'")
    results = search("credit losses and provisions", top_k=3)
    for r in results:
        print(f"\n  [{r['metadata']['bank']} {r['metadata']['quarter']}] "
              f"(relevance: {r['relevance']:.2f})")
        print(f"  {r['text'][:200]}...")
