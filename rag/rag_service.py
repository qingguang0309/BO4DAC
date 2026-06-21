"""
Online RAG retrieval for BO4DAC.

Mirrors the Tavily web-search interface: run_rag_retrieval() returns a list of
raw source items that llm_service assembles (with unified [N] numbering) into
the prompt alongside the live web-search results.

Each returned item: {"title", "url", "snippet", "kind": "experiment"|"paper", "score"}
Degrades gracefully to [] when RAG is disabled, the index is missing, the model
is not yet downloaded, or any error occurs.
"""

import os
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv

from rag import store
from rag.embedder import embed

load_dotenv()
logger = logging.getLogger(__name__)

ENABLE_RAG = os.getenv("ENABLE_RAG", "False").lower() in ("true", "1", "yes")
RAG_TOP_K_PAPERS = int(os.getenv("RAG_TOP_K_PAPERS", "4"))
RAG_TOP_K_EXPERIMENTS = int(os.getenv("RAG_TOP_K_EXPERIMENTS", "4"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))


def _build_rag_query(search_bounds: Dict, conditions: Dict) -> str:
    """Compose one rich retrieval query from the experiment context."""
    supports = search_bounds.get("supports", []) or []
    amine1 = search_bounds.get("amine1", []) or []
    co2 = conditions.get("co2Concentration", 0.04)
    temp = conditions.get("temperature", 25)
    context = "direct air capture" if co2 <= 0.1 else "post-combustion CO2 capture"
    amines = " ".join(amine1[:2]) if amine1 else "amine"
    sups = " ".join(supports[:3]) if supports else "porous support"
    return (
        f"{amines} impregnated {sups} solid sorbent CO2 adsorption capacity "
        f"{context} {int(round(temp))}C"
    )


def _doi_url(doi: str) -> str:
    doi = (doi or "").strip()
    if not doi:
        return ""
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"


def _query_collection(coll_name: str, query_emb, top_k: int, kind: str) -> List[Dict[str, Any]]:
    col = store.get_if_exists(coll_name)
    if col is None:
        return []
    try:
        res = col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"RAG query on '{coll_name}' failed: {e}")
        return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1.0 - float(dist)  # cosine distance -> similarity
        if score < RAG_MIN_SCORE:
            continue
        meta = meta or {}
        url = _doi_url(meta.get("doi", ""))
        if kind == "experiment":
            title = (
                f"实验记录: {meta.get('support', '?')} + {meta.get('amine', '?')}"
                f" → {meta.get('capacity_mmol_g', '?')} mmol/g"
            )
        else:
            title = f"文献: {meta.get('source_file', 'paper')}"
        out.append({
            "title": title,
            "url": url,
            "snippet": doc,
            "kind": kind,
            "score": round(score, 3),
        })
    return out


def run_rag_retrieval(search_bounds: Dict, conditions: Dict) -> List[Dict[str, Any]]:
    """Return raw RAG source items (experiments first, then papers); [] if unavailable."""
    if not ENABLE_RAG:
        return []
    try:
        query = _build_rag_query(search_bounds, conditions)
        query_emb = embed([query])[0]
    except Exception as e:
        logger.warning(f"RAG embedding failed (is bge-m3 downloaded?): {e}")
        return []

    items: List[Dict[str, Any]] = []
    items += _query_collection(store.COLL_EXPERIMENTS, query_emb, RAG_TOP_K_EXPERIMENTS, "experiment")
    items += _query_collection(store.COLL_PAPERS, query_emb, RAG_TOP_K_PAPERS, "paper")
    return items
