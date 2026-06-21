"""
LLM-assisted suggestion service for DAC optimization.

Uses an OpenAI-compatible chat.completions endpoint (e.g. DashScope/Qwen or
GpuGeek/GLM) for the suggestions, with optional literature grounding via the
Tavily web search API. Supports both synchronous and streaming (SSE) responses.

Web search runs when ENABLE_WEB_SEARCH is true and TAVILY_API_KEY is set:
targeted academic queries are derived from the experiment context, the
retrieved sources are injected into the prompt for [N]-style citation, and the
source list is surfaced to the client. The OpenAI Responses API and its
built-in web_search tool are intentionally not used (not supported by these
chat.completions endpoints).
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")
MODEL = os.getenv("MODEL", "qwen-plus")
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "False").lower() in ("true", "1", "yes")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "True").lower() in ("true", "1", "yes")
# Reasoning models (GLM) spend many tokens on hidden reasoning before the
# final answer, so allow a generous completion budget.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"

_client = None

# Local RAG retrieval (optional). Import defensively so a missing rag package or
# dependency never breaks the core LLM service — it just disables RAG.
try:
    from rag.rag_service import run_rag_retrieval
except Exception as _rag_import_err:  # pragma: no cover
    logger.warning(f"RAG retrieval unavailable: {_rag_import_err}")

    def run_rag_retrieval(search_bounds, conditions):
        return []


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def _build_search_queries(search_bounds: Dict, conditions: Dict, max_queries: int = 3) -> List[str]:
    """Derive targeted academic search queries from the experiment context."""
    supports = search_bounds.get("supports", []) or []
    amine1 = search_bounds.get("amine1", []) or []
    co2 = conditions.get("co2Concentration", 0.04)
    temp = conditions.get("temperature", 25)
    # ~400 ppm (0.04 vol%) or very low → direct air capture; higher → flue gas
    context = "direct air capture" if co2 <= 0.1 else "post-combustion CO2 capture"
    amine = amine1[0] if amine1 else "amine"

    queries: List[str] = []
    for sp in supports[:2]:
        queries.append(f"{amine} impregnated {sp} sorbent CO2 adsorption capacity {context}")
    if amine1:
        queries.append(f"{amine} amine solid sorbent CO2 capture {context} {int(round(temp))}C")
    if not queries:
        queries.append(f"amine functionalized solid sorbent CO2 capture {context}")

    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_queries]


def _tavily_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Run a single Tavily search; returns [{title, url, content}], [] on failure."""
    if not TAVILY_API_KEY:
        return []
    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in data.get("results", [])
            if r.get("url")
        ]
    except Exception as e:
        logger.warning(f"Tavily search failed for '{query}': {e}")
        return []


def _run_web_search(
    search_bounds: Dict, conditions: Dict, max_queries: int = 3, per_query: int = 3
) -> List[Dict[str, Any]]:
    """Run web searches; return raw items [{title, url, snippet, kind:'web'}].

    Deduplicated by URL across all queries. Numbering/blocks are assigned later
    by _assemble_context so RAG and web sources share one [N] sequence.
    """
    if not (ENABLE_WEB_SEARCH and TAVILY_API_KEY):
        return []

    seen_urls = set()
    items: List[Dict[str, Any]] = []
    for q in _build_search_queries(search_bounds, conditions, max_queries):
        for r in _tavily_search(q, per_query):
            url = r["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = r["title"] or _title_from_url(url)
            items.append({
                "title": title,
                "url": url,
                "snippet": " ".join((r["content"] or "").split()),
                "kind": "web",
            })
    return items


def _assemble_context(
    rag_items: List[Dict[str, Any]], web_items: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Merge RAG + web sources under one continuous [N] numbering.

    Returns (sources, rag_block, web_block):
      - sources: [{index, title, url, type}] for the search_source SSE event
      - rag_block / web_block: numbered text for the two prompt sections
    RAG items keep their own URLs (multiple records may share a DOI); web items
    are skipped if their URL was already surfaced by RAG or an earlier web item.
    """
    sources: List[Dict[str, Any]] = []
    rag_lines: List[str] = []
    web_lines: List[str] = []
    seen_urls = set()

    def _render(item: Dict[str, Any], idx: int) -> str:
        snippet = " ".join((item.get("snippet") or "").split())
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."
        url = item.get("url", "")
        ref = f"    来源: {url}" if url else "    来源: 本地知识库"
        return f"[{idx}] {item.get('title', '')}\n    {snippet}\n{ref}"

    for it in rag_items:
        idx = len(sources) + 1
        sources.append({"index": idx, "title": it.get("title", ""),
                        "url": it.get("url", ""), "type": it.get("kind", "rag")})
        rag_lines.append(_render(it, idx))
        if it.get("url"):
            seen_urls.add(it["url"])

    for it in web_items:
        url = it.get("url", "")
        if url and url in seen_urls:
            continue
        idx = len(sources) + 1
        sources.append({"index": idx, "title": it.get("title", ""),
                        "url": url, "type": "web"})
        web_lines.append(_render(it, idx))
        if url:
            seen_urls.add(url)

    return sources, "\n".join(rag_lines), "\n".join(web_lines)


def _build_prompt(
    experiments: List[Dict],
    search_bounds: Dict,
    conditions: Dict,
    candidates: List[Dict],
    avg_uncertainty: float,
    optimization_info: Optional[Dict] = None,
    rag_results_block: Optional[str] = None,
    web_results_block: Optional[str] = None,
) -> str:
    """Construct the LLM prompt with full experimental context."""

    # --- Search space ---
    supports = search_bounds.get("supports", [])
    amine1 = search_bounds.get("amine1", [])
    amine2 = search_bounds.get("amine2", [])
    oc_range = search_bounds.get("ocRange", [0, 100])
    support_specific = search_bounds.get("supportSpecificRanges", {})

    # Aggregate BET / pore ranges
    bet_min, bet_max = float("inf"), float("-inf")
    pore_min, pore_max = float("inf"), float("-inf")
    for sp in supports:
        sr = support_specific.get(sp, {})
        br = sr.get("betRange", [0, 1000])
        pr = sr.get("poreRange", [0, 20])
        bet_min = min(bet_min, br[0])
        bet_max = max(bet_max, br[1])
        pore_min = min(pore_min, pr[0])
        pore_max = max(pore_max, pr[1])
    if bet_min == float("inf"):
        bet_min, bet_max = 0, 1000
    if pore_min == float("inf"):
        pore_min, pore_max = 0, 20

    # --- Conditions ---
    temp = conditions.get("temperature", 25.0)
    co2 = conditions.get("co2Concentration", 0.04)
    humidity = conditions.get("humidity", 0)
    flow = conditions.get("flowRate", 100.0)
    method = conditions.get("testMethod", "TGA")

    # --- Experimental data table ---
    filtered_experiments = [e for e in experiments if e.get('matches_filter', True)]
    header = "| # | Support | Amine1 | Amine2 | OC% | BET | Pore | T(C) | CO2% | RH% | Pred | Actual | Hist? |"
    sep = "|---|---------|--------|--------|-----|-----|------|------|------|-----|------|--------|-------|"
    rows = []
    for i, exp in enumerate(filtered_experiments):
        c = exp.get("candidate", {})
        cap = exp.get("experimental_performance", 0.0)
        pred = exp.get("predicted_performance", 0.0)
        is_hist = "Y" if exp.get("is_historical", False) else ""
        exp_temp = exp.get("Temperature", c.get("Temperature", temp))
        exp_co2 = exp.get("CO2_Concentration", c.get("CO2_Concentration", co2))
        exp_rh = exp.get("Humidity", c.get("Humidity", humidity))
        rows.append(
            f"| {i+1} | {c.get('Support','')} | {c.get('Amine_1_or_Additive_1','')} "
            f"| {c.get('Amine_2_or_Additive_2','')} "
            f"| {c.get('Organic_Content_pct',0):.1f} "
            f"| {c.get('BET_Bare_Surface_Area_m2_g',0):.1f} "
            f"| {c.get('Average_Bare_Pore_Diameter_nm',0):.2f} "
            f"| {exp_temp} "
            f"| {exp_co2} "
            f"| {exp_rh} "
            f"| {pred:.4f} "
            f"| {cap:.4f} "
            f"| {is_hist} |"
        )
    data_table = "\n".join([header, sep] + rows) if rows else "(no experiments yet)"

    # --- Best result ---
    best_cap = 0.0
    best_form = ""
    for exp in filtered_experiments:
        cap = exp.get("experimental_performance", 0.0)
        if cap > best_cap:
            best_cap = cap
            c = exp.get("candidate", {})
            best_form = (
                f"{c.get('Support','')}|{c.get('Amine_1_or_Additive_1','')}"
                f"|{c.get('Amine_2_or_Additive_2','')}|OC={c.get('Organic_Content_pct',0):.1f}%"
            )

    # --- Optimization configuration block ---
    opt = optimization_info or {}
    opt_block = f"""- Surrogate model: {opt.get('model_type', 'Gaussian Process (SingleTaskGP)')}
- Acquisition function: {opt.get('acquisition_function', 'qExpectedImprovement')}
- Candidates sampled per iteration: {opt.get('n_candidates_sampled', 1000)}
- Exploration parameter (xi): {opt.get('exploration_xi', 0.01)}
- Confidence level: {opt.get('confidence_level', 0.95)}
- Observed data points: {opt.get('n_observed_data_points', 0)}
- Best observed capacity: {opt.get('best_observed_capacity', 0.0):.4f} mmol/g"""

    # --- Current candidates ---
    cand_lines = []
    for i, c in enumerate(candidates):
        cand_lines.append(
            f"  {i+1}. {c.get('Support','')} + {c.get('Amine_1_or_Additive_1','')} "
            f"+ {c.get('Amine_2_or_Additive_2','')} | OC={c.get('Organic_Content_pct',0):.1f}% "
            f"| BET={c.get('BET_Bare_Surface_Area_m2_g',0):.1f} "
            f"| Pore={c.get('Average_Bare_Pore_Diameter_nm',0):.2f} "
            f"| Pred={c.get('Predicted_CO2_Capacity_mmol_g',0):.4f} "
            f"| Unc={c.get('Uncertainty',0):.4f}"
        )
    cand_block = "\n".join(cand_lines) if cand_lines else "(no candidates generated yet)"

    # --- Pre-compute experimental findings ---
    findings_lines = []

    sorted_exps = sorted(filtered_experiments, key=lambda e: e.get("experimental_performance", 0.0), reverse=True)
    if sorted_exps:
        top3 = sorted_exps[:min(3, len(sorted_exps))]
        findings_lines.append("Top-performing formulations:")
        for rank, exp in enumerate(top3, 1):
            c = exp.get("candidate", {})
            cap = exp.get("experimental_performance", 0.0)
            findings_lines.append(
                f"  {rank}. {c.get('Support','')} + {c.get('Amine_1_or_Additive_1','')} + "
                f"{c.get('Amine_2_or_Additive_2','')} | OC={c.get('Organic_Content_pct',0):.1f}% | "
                f"Capacity={cap:.4f} mmol/g"
            )

    tested_combos = set()
    for exp in filtered_experiments:
        c = exp.get("candidate", {})
        key = (c.get("Support", ""), c.get("Amine_1_or_Additive_1", ""), c.get("Amine_2_or_Additive_2", ""))
        tested_combos.add(key)
    untested = []
    for sp in supports:
        for a1 in amine1:
            for a2 in amine2:
                if (sp, a1, a2) not in tested_combos:
                    untested.append(f"{sp} + {a1} + {a2}")
    if untested:
        findings_lines.append(f"\nTested combinations: {len(tested_combos)} | Untested: {len(untested)}")
        if len(untested) <= 10:
            findings_lines.append("Untested combos: " + ", ".join(untested))
        else:
            findings_lines.append("Sample untested combos: " + ", ".join(untested[:10]) + f" ... (+{len(untested)-10} more)")

    if sorted_exps:
        top_oc_vals = [exp.get("candidate", {}).get("Organic_Content_pct", 0) for exp in sorted_exps[:min(5, len(sorted_exps))]]
        if top_oc_vals:
            findings_lines.append(f"\nOC% range of top 5 performers: {min(top_oc_vals):.1f}-{max(top_oc_vals):.1f}% (search space allows {oc_range[0]}-{oc_range[1]}%)")

    pred_actual_diffs = []
    for exp in filtered_experiments:
        pred = exp.get("predicted_performance", 0.0)
        actual = exp.get("experimental_performance", 0.0)
        if pred > 0 and actual > 0:
            pred_actual_diffs.append(actual - pred)
    if pred_actual_diffs:
        avg_bias = sum(pred_actual_diffs) / len(pred_actual_diffs)
        direction = "over-predicting" if avg_bias < 0 else "under-predicting"
        findings_lines.append(f"\nGP model bias: {direction} by {abs(avg_bias):.4f} mmol/g on average (across {len(pred_actual_diffs)} experiments with predictions)")

    support_caps = {}
    for exp in filtered_experiments:
        c = exp.get("candidate", {})
        sp = c.get("Support", "")
        cap = exp.get("experimental_performance", 0.0)
        if sp:
            support_caps.setdefault(sp, []).append(cap)
    if support_caps:
        findings_lines.append("\nPerformance by support:")
        for sp, caps in sorted(support_caps.items(), key=lambda x: max(x[1]), reverse=True):
            findings_lines.append(f"  {sp}: max={max(caps):.4f}, mean={sum(caps)/len(caps):.4f}, n={len(caps)}")

    amine_eff = {}
    for exp in filtered_experiments:
        c = exp.get("candidate", {})
        oc = c.get("Organic_Content_pct", 0)
        cap = exp.get("experimental_performance", 0.0)
        a1 = c.get("Amine_1_or_Additive_1", "")
        if oc > 0 and cap > 0 and a1:
            eff = cap / (oc / 100.0)
            amine_eff.setdefault(a1, []).append(eff)
    if amine_eff:
        findings_lines.append("\nAmine efficiency (capacity per OC fraction) by Amine1:")
        for a1, effs in sorted(amine_eff.items(), key=lambda x: max(x[1]), reverse=True):
            findings_lines.append(f"  {a1}: best={max(effs):.4f}, mean={sum(effs)/len(effs):.4f}")

    findings_block = "\n".join(findings_lines) if findings_lines else "(insufficient data for analysis)"

    if rag_results_block:
        rag_section = (
            "[Local Knowledge Base] (curated DAC literature + historical experiment "
            "records — authoritative, prioritize these; cite as [N])\n" + rag_results_block
        )
    else:
        rag_section = "[Local Knowledge Base] (no relevant local records retrieved)"

    if web_results_block:
        web_section = (
            "[Web Search Results] (live web search — cite as [N])\n" + web_results_block
        )
    else:
        web_section = "[Web Search Results] (none available)"

    evidence_section = rag_section + "\n\n" + web_section

    if rag_results_block or web_results_block:
        step2 = (
            "STEP 2 — REVIEW the evidence in [Local Knowledge Base] (curated/authoritative — "
            "prioritize) and [Web Search Results] below. Ground your reasoning in these sources "
            "and cite specific items with [N] markers."
        )
    else:
        step2 = (
            "STEP 2 — Draw on your knowledge of the relevant academic literature for this "
            "search space and conditions (no retrieved evidence is available for this query)."
        )

    prompt = f"""You are an expert in Direct Air Capture (DAC) CO2 capture materials optimization.
We are optimizing amine-impregnated solid sorbents for CO2 capture using Bayesian Optimization.

STEP 1 — ANALYZE the experimental settings and data and findings below. Identify key patterns, gaps, and opportunities.

{step2}

STEP 3 — Suggest 5 NOVEL and REASONABLE formulations guided by and experimental findings and search results.
Particularly at the beginning several iterations at less number of historical experiments, searching results with diverse optimal formulation for different material discovery is quite important.

[Search Space]
- Supports: {', '.join(supports)}
- Amines 1: {', '.join(amine1)}
- Amines 2: {', '.join(amine2)}
- Organic Content: {oc_range[0]}-{oc_range[1]}%
- BET Surface Area: {bet_min}-{bet_max} m2/g
- Pore Diameter: {pore_min}-{pore_max} nm

[Experimental Conditions]
- Temperature: {temp} C, CO2: {co2} vol%, Humidity: {humidity}%, Flow: {flow} mL/min, Method: {method}

[Optimization Configuration]
{opt_block}

[All Experimental Data]
{data_table}

[Best Result] {best_cap:.4f} mmol/g with {best_form}

[Experimental Findings Summary]
{findings_block}

{evidence_section}

[Current BO Candidates (avg uncertainty={avg_uncertainty:.4f} mmol/g — LOW, model converging)]
{cand_block}

IMPORTANT: All suggested materials MUST be from the search space lists above. Do not invent materials not in the lists.

CITE YOUR SOURCES: In the "reasoning" field, reference web search results using citation markers like [1], [2], etc. Example: "PEI on SBA-15 showed 2.1 mmol/g at 25C [1], suggesting..."

Format your response as a JSON array. Each element must have:
- "Support": string (from the supports list)
- "Amine_1_or_Additive_1": string (from the amines 1 list)
- "Amine_2_or_Additive_2": string (from the amines 2 list)
- "Organic_Content_pct": number
- "BET_Bare_Surface_Area_m2_g": float
- "Average_Bare_Pore_Diameter_nm": float
- "Expected_CO2_Capacity_mmol_g": float (2 decimal) (your estimated CO2 capture capacity in mmol/g for the test method {method}, based on literature and the experimental data)
- "reasoning": string (scientific explanation with [N] citation markers referencing search sources). 

Output ONLY the JSON array, no other text."""
    return prompt


def _parse_suggestions(raw: str, search_bounds: Dict) -> List[Dict[str, Any]]:
    """Extract and validate structured suggestions from the LLM response."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    suggestions = []

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            suggestions = result
    except json.JSONDecodeError:
        pass

    if not suggestions:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    suggestions = result
            except json.JSONDecodeError:
                pass

    if not suggestions:
        return [{"reasoning": raw, "raw_response": True}]

    valid_supports = set(search_bounds.get("supports", []))
    valid_amine1 = set(search_bounds.get("amine1", []))
    valid_amine2 = set(search_bounds.get("amine2", []))

    validated = []
    for s in suggestions:
        if s.get("raw_response"):
            validated.append(s)
            continue
        if s.get("Support") not in valid_supports and valid_supports:
            s["Support"] = list(valid_supports)[0]
        if s.get("Amine_1_or_Additive_1") not in valid_amine1 and valid_amine1:
            s["Amine_1_or_Additive_1"] = list(valid_amine1)[0]
        if s.get("Amine_2_or_Additive_2") not in valid_amine2 and valid_amine2:
            s["Amine_2_or_Additive_2"] = "No"
        for field in ["Organic_Content_pct", "BET_Bare_Surface_Area_m2_g", "Average_Bare_Pore_Diameter_nm", "Expected_CO2_Capacity_mmol_g"]:
            try:
                s[field] = float(s.get(field, 0))
            except (TypeError, ValueError):
                s[field] = 0.0
        validated.append(s)

    return validated


def _title_from_url(url: str) -> str:
    """Derive a readable title from a URL when no title is provided.

    Produces academic-style references: Publisher — Path hint.
    """
    try:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path
        host = re.sub(r'^www\.', '', host)

        # Map common academic domains to publisher names
        publisher_map = {
            'sciencedirect.com': 'ScienceDirect',
            'springer.com': 'Springer',
            'link.springer.com': 'Springer',
            'nature.com': 'Nature',
            'pubs.acs.org': 'ACS Publications',
            'acs.org': 'ACS',
            'wiley.com': 'Wiley',
            'onlinelibrary.wiley.com': 'Wiley',
            'pubmed.ncbi.nlm.nih.gov': 'PubMed',
            'ncbi.nlm.nih.gov': 'NCBI',
            'arxiv.org': 'arXiv',
            'doi.org': 'DOI',
            'researchgate.net': 'ResearchGate',
            'semanticscholar.org': 'Semantic Scholar',
            'scholar.google.com': 'Google Scholar',
            'mdpi.com': 'MDPI',
            'rsc.org': 'RSC',
            'pubs.rsc.org': 'RSC Publishing',
            'elsevier.com': 'Elsevier',
            'tandfonline.com': 'Taylor & Francis',
            'ieee.org': 'IEEE',
            'dl.acm.org': 'ACM Digital Library',
            'jstor.org': 'JSTOR',
            'pnas.org': 'PNAS',
            'cell.com': 'Cell Press',
            'sciencedirect.io': 'ScienceDirect',
        }
        publisher = publisher_map.get(host, host)

        # Extract a hint from the path
        path = unquote(parsed.path).strip('/')
        # Remove common prefixes like /science/article, /content, /article
        path = re.sub(r'^(science/article|content/article|article/abs|article/full|doi/abs|doi/full|doi/pdf|book|chapter|journal)/?', '', path, flags=re.IGNORECASE)
        # Clean up the remaining path
        path = path.replace('_', ' ').replace('-', ' ')
        segments = [s for s in path.split('/') if s and not re.match(r'^[a-f0-9]{8,}$', s)]

        if segments:
            hint = segments[-1]
            # Truncate long hints
            if len(hint) > 80:
                hint = hint[:77] + '...'
            # Capitalize first letter
            hint = hint[0].upper() + hint[1:] if hint else ''
            return f"{publisher} — {hint}"
        return publisher
    except Exception:
        return url[:60]


def _extract_sources_from_output(output_items: list) -> List[Dict[str, Any]]:
    """Extract web search sources from Responses API output items.

    Deduplicates by URL and derives titles when missing.
    """
    seen_urls = set()
    sources = []
    for item in output_items:
        if hasattr(item, 'type') and item.type == 'web_search_call':
            action = getattr(item, 'action', None)
            if action and hasattr(action, 'sources'):
                for src in action.sources:
                    url = getattr(src, 'url', '')
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    title = getattr(src, 'title', '') or ''
                    if not title:
                        title = _title_from_url(url)
                    sources.append({
                        "index": len(sources) + 1,
                        "title": title,
                        "url": url,
                    })
    return sources


def stream_llm_suggestions(
    experiments: List[Dict],
    search_bounds: Dict,
    conditions: Dict,
    candidates: List[Dict],
    avg_uncertainty: float,
    optimization_info: Optional[Dict] = None,
) -> Generator[str, None, None]:
    """
    Stream LLM suggestions as SSE events using the chat.completions API.

    Yields SSE-formatted strings:
      - event: search_source  data: {"sources": [...]}  — web search sources
      - event: token   data: {"text": "..."}            — streamed text
      - event: thinking data: {"text": "..."}           — reasoning tokens
      - event: done    data: {"suggestions": [...], ...} — final result
      - event: error   data: {"error": "..."}           — on failure
    """
    if not API_KEY:
        yield _sse("error", {"error": "DASHSCOPE_API_KEY not set in .env"})
        return

    # Evidence grounding: local RAG knowledge base + live Tavily web search.
    # Emit merged sources up-front so the UI can show them while the model streams.
    rag_items = run_rag_retrieval(search_bounds, conditions)
    web_items = _run_web_search(search_bounds, conditions)
    search_sources, rag_block, web_block = _assemble_context(rag_items, web_items)
    if search_sources:
        yield _sse("search_source", {"sources": search_sources})

    prompt = _build_prompt(
        experiments, search_bounds, conditions, candidates, avg_uncertainty,
        optimization_info, rag_results_block=rag_block, web_results_block=web_block,
    )

    try:
        client = _get_client()

        # OpenAI-compatible chat.completions streaming API (DashScope/Qwen or GpuGeek/GLM).
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        full_text = ""

        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # Reasoning/thinking tokens (GLM returns these in reasoning_content)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield _sse("thinking", {"text": reasoning})

            # Content text tokens
            if getattr(delta, "content", None):
                full_text += delta.content
                yield _sse("token", {"text": delta.content})

        # Parse the full accumulated text into structured suggestions
        suggestions = _parse_suggestions(full_text, search_bounds)
        yield _sse("done", {
            "suggestions": suggestions,
            "raw_response": full_text,
            "avg_uncertainty": avg_uncertainty,
            "search_sources": search_sources,
        })

    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")
        yield _sse("error", {"error": str(e)})


def get_llm_suggestions(
    experiments: List[Dict],
    search_bounds: Dict,
    conditions: Dict,
    candidates: List[Dict],
    avg_uncertainty: float,
    n_suggestions: int = 5,
    optimization_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Non-streaming variant using the chat.completions API.

    Returns dict with: suggestions, raw_response, avg_uncertainty, search_sources, error.
    """
    if not API_KEY:
        return {
            "suggestions": [],
            "raw_response": "",
            "avg_uncertainty": avg_uncertainty,
            "error": "DASHSCOPE_API_KEY not set in .env",
        }

    # Evidence grounding: local RAG knowledge base + live Tavily web search.
    rag_items = run_rag_retrieval(search_bounds, conditions)
    web_items = _run_web_search(search_bounds, conditions)
    search_sources, rag_block, web_block = _assemble_context(rag_items, web_items)
    prompt = _build_prompt(
        experiments, search_bounds, conditions, candidates, avg_uncertainty,
        optimization_info, rag_results_block=rag_block, web_results_block=web_block,
    )

    try:
        client = _get_client()

        # OpenAI-compatible chat.completions endpoint (DashScope/Qwen or GpuGeek/GLM).
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
        )

        raw_content = (response.choices[0].message.content or "").strip()

        suggestions = _parse_suggestions(raw_content, search_bounds)

        return {
            "suggestions": suggestions[:n_suggestions],
            "raw_response": raw_content,
            "avg_uncertainty": avg_uncertainty,
            "search_sources": search_sources,
            "web_search_count": 0,
            "error": None,
        }

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return {
            "suggestions": [],
            "raw_response": "",
            "avg_uncertainty": avg_uncertainty,
            "error": str(e),
        }


def _sse(event: str, data: Any) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
