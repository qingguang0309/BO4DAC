"""
ChromaDB persistence layer for the RAG knowledge base.

Two collections, both using cosine distance:
  - papers       : chunked academic PDFs
  - experiments  : one card per historical experiment record
"""

import os
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RAG_DB_PATH = os.getenv("RAG_DB_PATH", "data/rag_chroma")
COLL_PAPERS = "papers"
COLL_EXPERIMENTS = "experiments"

_client = None


def get_client(db_path: str = None):
    """Return a singleton persistent Chroma client."""
    global _client
    if _client is None:
        import chromadb
        path = db_path or RAG_DB_PATH
        os.makedirs(path, exist_ok=True)
        _client = chromadb.PersistentClient(path=path)
    return _client


def get_or_create(name: str):
    """Get/create a collection (cosine space)."""
    return get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def get_if_exists(name: str):
    """Return the collection if it exists, else None (no exception)."""
    try:
        return get_client().get_collection(name=name)
    except Exception:
        return None
