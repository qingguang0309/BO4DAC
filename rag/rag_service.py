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
RAG_TOP_K_PAPERS = int(os.getenv("RAG_TOP_K_PAPERS", "6"))
RAG_TOP_K_EXPERIMENTS = int(os.getenv("RAG_TOP_K_EXPERIMENTS", "4"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))
# 同一篇文献最多保留的 chunk 数，避免单篇刷屏挤掉其他来源
RAG_MAX_CHUNKS_PER_PAPER = int(os.getenv("RAG_MAX_CHUNKS_PER_PAPER", "2"))


def _build_rag_queries(search_bounds: Dict, conditions: Dict,
                       max_queries: int = 4) -> List[str]:
    """Compose several targeted retrieval queries (mirrors the web-search queries)
    plus one combined query; results are merged so coverage matches web search."""
    supports = search_bounds.get("supports", []) or []
    amine1 = search_bounds.get("amine1", []) or []
    co2 = conditions.get("co2Concentration", 0.04)
    temp = conditions.get("temperature", 25)
    context = "direct air capture" if co2 <= 0.1 else "post-combustion CO2 capture"
    amine = amine1[0] if amine1 else "amine"

    queries: List[str] = []
    for sp in supports[:2]:
        queries.append(f"{amine} impregnated {sp} sorbent CO2 adsorption capacity {context}")
    if amine1:
        queries.append(f"{amine} amine solid sorbent CO2 capture {context} {int(round(temp))}C")
    amines = " ".join(amine1[:2]) if amine1 else "amine"
    sups = " ".join(supports[:3]) if supports else "porous support"
    queries.append(
        f"{amines} impregnated {sups} solid sorbent CO2 adsorption capacity "
        f"{context} {int(round(temp))}C"
    )

    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_queries]


def _doi_url(doi: str) -> str:
    doi = (doi or "").strip()
    if not doi:
        return ""
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"


def _item_title(meta: Dict[str, Any], kind: str) -> str:
    if kind == "experiment":
        return (
            f"实验记录: {meta.get('support', '?')} + {meta.get('amine', '?')}"
            f" → {meta.get('capacity_mmol_g', '?')} mmol/g"
        )
    title = meta.get("title") or meta.get("source_file", "paper")
    if len(title) > 90:
        title = title[:87] + "..."
    sec = meta.get("section", "")
    if sec and sec not in ("Title & Abstract", "body"):
        return f"文献: {title}（§{sec}）"
    return f"文献: {title}"


def _query_collection(coll_name: str, query_embs: List, top_k: int, kind: str) -> List[Dict[str, Any]]:
    """Query one collection with several query embeddings, merge hits by id
    (keeping each hit's best score), and return them sorted by score."""
    col = store.get_if_exists(coll_name)
    if col is None:
        return []
    try:
        res = col.query(
            query_embeddings=query_embs,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"RAG query on '{coll_name}' failed: {e}")
        return []

    best: Dict[str, Dict[str, Any]] = {}
    for qi in range(len(query_embs)):
        ids = (res.get("ids") or [[]])[qi]
        docs = (res.get("documents") or [[]])[qi]
        metas = (res.get("metadatas") or [[]])[qi]
        dists = (res.get("distances") or [[]])[qi]
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            if score < RAG_MIN_SCORE:
                continue
            if id_ in best and best[id_]["score"] >= score:
                continue
            meta = meta or {}
            best[id_] = {
                "title": _item_title(meta, kind),
                "url": _doi_url(meta.get("doi", "")),
                "snippet": doc,
                "kind": kind,
                "score": round(score, 3),
                "paper_id": meta.get("paper_id") or meta.get("source_file", ""),
            }

    items = sorted(best.values(), key=lambda x: x["score"], reverse=True)

    if kind == "paper":
        # 同一篇文献最多保留 RAG_MAX_CHUNKS_PER_PAPER 个最高分 chunk
        kept, per_paper = [], {}
        for it in items:
            pid = it.get("paper_id", "")
            if per_paper.get(pid, 0) >= RAG_MAX_CHUNKS_PER_PAPER:
                continue
            per_paper[pid] = per_paper.get(pid, 0) + 1
            kept.append(it)
        items = kept

    for it in items:
        it.pop("paper_id", None)
    return items[:top_k]


def run_rag_retrieval(search_bounds: Dict, conditions: Dict) -> List[Dict[str, Any]]:
    """Return raw RAG source items (experiments first, then papers); [] if unavailable."""
    if not ENABLE_RAG:
        return []
    try:
        queries = _build_rag_queries(search_bounds, conditions)
        query_embs = embed(queries)
    except Exception as e:
        logger.warning(f"RAG embedding failed (is bge-m3 downloaded?): {e}")
        return []

    items: List[Dict[str, Any]] = []
    items += _query_collection(store.COLL_EXPERIMENTS, query_embs, RAG_TOP_K_EXPERIMENTS, "experiment")
    items += _query_collection(store.COLL_PAPERS, query_embs, RAG_TOP_K_PAPERS, "paper")
    return items


_local_paper_dois = None


def get_local_paper_dois(refresh: bool = False) -> set:
    """Set of DOIs (lowercased) covered by the local papers collection.

    Used by llm_service to drop web-search results that duplicate papers we
    already hold in full text locally. Cached after first load; returns an
    empty set on any failure."""
    global _local_paper_dois
    if _local_paper_dois is not None and not refresh:
        return _local_paper_dois
    dois = set()
    try:
        col = store.get_if_exists(store.COLL_PAPERS)
        if col is not None:
            data = col.get(include=["metadatas"])
            for meta in data.get("metadatas") or []:
                doi = ((meta or {}).get("doi") or "").strip().lower()
                if doi:
                    dois.add(doi)
    except Exception as e:
        logger.warning(f"Loading local paper DOIs failed: {e}")
    _local_paper_dois = dois
    return dois
