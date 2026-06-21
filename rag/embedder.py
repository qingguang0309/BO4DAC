"""
Singleton embedding model for the RAG knowledge base.

Defaults to BAAI/bge-m3 (bilingual zh/en, long context, strong retrieval).
The model is loaded lazily on first use and reused across calls. First load
downloads ~2GB from HuggingFace; set HF_ENDPOINT=https://hf-mirror.com in .env
if the download is slow.
"""

import os
import logging
from typing import List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# "cpu" is safe everywhere; set EMBEDDING_DEVICE=mps on Apple Silicon for speed.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

_model = None


def get_model():
    """Load (once) and return the SentenceTransformer embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model {EMBEDDING_MODEL} on {EMBEDDING_DEVICE} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _model


def embed(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Encode texts into L2-normalized embeddings (ready for cosine similarity)."""
    if not texts:
        return []
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )
    return vecs.tolist()
