"""
LLM-assisted suggestion service for DAC optimization.

When the GP model's uncertainty drops below a threshold, this module calls
an LLM (via DashScope OpenAI-compatible API) to suggest novel formulations
that the Bayesian optimizer might miss.

Supports both synchronous and streaming (SSE) responses.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Generator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("MODEL", "deepseek-v4-pro")
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "False").lower() in ("true", "1", "yes")


def _build_prompt(
    experiments: List[Dict],
    search_bounds: Dict,
    conditions: Dict,
    candidates: List[Dict],
    avg_uncertainty: float,
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
    header = "| # | Support | Amine1 | Amine2 | OC% | BET | Pore | Capacity |"
    sep = "|---|---------|--------|--------|-----|-----|------|----------|"
    rows = []
    for i, exp in enumerate(experiments):
        c = exp.get("candidate", {})
        cap = exp.get("experimental_performance", 0.0)
        rows.append(
            f"| {i+1} | {c.get('Support','')} | {c.get('Amine_1_or_Additive_1','')} "
            f"| {c.get('Amine_2_or_Additive_2','')} "
            f"| {c.get('Organic_Content_pct',0):.1f} "
            f"| {c.get('BET_Bare_Surface_Area_m2_g',0):.1f} "
            f"| {c.get('Average_Bare_Pore_Diameter_nm',0):.2f} "
            f"| {cap:.4f} |"
        )
    data_table = "\n".join([header, sep] + rows)

    # --- Best result ---
    best_cap = 0.0
    best_form = ""
    for exp in experiments:
        cap = exp.get("experimental_performance", 0.0)
        if cap > best_cap:
            best_cap = cap
            c = exp.get("candidate", {})
            best_form = (
                f"{c.get('Support','')}|{c.get('Amine_1_or_Additive_1','')}"
                f"|{c.get('Amine_2_or_Additive_2','')}|OC={c.get('Organic_Content_pct',0):.1f}%"
            )

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
    cand_block = "\n".join(cand_lines)

    prompt = f"""You are an expert in Direct Air Capture (DAC) CO2 capture materials optimization.
We are optimizing amine-impregnated solid sorbents for CO2 capture using Bayesian Optimization (Gaussian Process).

[Search Space]
- Supports: {', '.join(supports)}
- Amines 1: {', '.join(amine1)}
- Amines 2: {', '.join(amine2)}
- Organic Content: {oc_range[0]}-{oc_range[1]}%
- BET Surface Area: {bet_min}-{bet_max} m2/g
- Pore Diameter: {pore_min}-{pore_max} nm

[Experimental Conditions]
- Temperature: {temp} C, CO2: {co2} vol%, Humidity: {humidity}%, Flow: {flow} mL/min, Method: {method}

[All Experimental Data So Far]
{data_table}

[Best Result] {best_cap:.4f} mmol/g with {best_form}

[Current BO Candidates (low uncertainty - model is converging)]
{cand_block}

The Bayesian optimization model's average uncertainty is LOW ({avg_uncertainty:.4f} mmol/g), suggesting the model is converging and may miss unexplored regions.

Please suggest 3-5 NOVEL formulations that the BO model might miss, with scientific reasoning for why they could outperform current candidates. Consider:
1. Unexplored support-amine combinations not yet tested
2. Different organic content ranges than the current focus
3. Literature-informed synergies between support porosity and amine chemistry
4. Potential for improved CO2 chemisorption via amine efficiency (capacity per organic content)
5. Pore size effects on amine dispersion and CO2 diffusion

IMPORTANT: All suggested materials MUST be from the search space lists above. Do not invent materials not in the lists.

Format your response as a JSON array. Each element must have:
- "Support": string (from the supports list)
- "Amine_1_or_Additive_1": string (from the amines 1 list)
- "Amine_2_or_Additive_2": string (from the amines 2 list)
- "Organic_Content_pct": number
- "BET_Bare_Surface_Area_m2_g": number
- "Average_Bare_Pore_Diameter_nm": number
- "reasoning": string (scientific explanation)

Output ONLY the JSON array, no other text."""
    return prompt


def _parse_suggestions(raw: str, search_bounds: Dict) -> List[Dict[str, Any]]:
    """Extract and validate structured suggestions from the LLM response."""
    # Strip markdown code fences
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    suggestions = []

    # Try direct JSON parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            suggestions = result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array with regex
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

    # Validate and sanitize
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
        for field in ["Organic_Content_pct", "BET_Bare_Surface_Area_m2_g", "Average_Bare_Pore_Diameter_nm"]:
            try:
                s[field] = float(s.get(field, 0))
            except (TypeError, ValueError):
                s[field] = 0.0
        validated.append(s)

    return validated


def stream_llm_suggestions(
    experiments: List[Dict],
    search_bounds: Dict,
    conditions: Dict,
    candidates: List[Dict],
    avg_uncertainty: float,
) -> Generator[str, None, None]:
    """
    Stream LLM suggestions as SSE events.

    Yields SSE-formatted strings:
      - event: token   data: {"text": "..."}    — each chunk of streamed text
      - event: done    data: {"suggestions": [...], "raw_response": "..."}  — final result
      - event: error   data: {"error": "..."}   — on failure
    """
    if not API_KEY:
        yield _sse("error", {"error": "DASHSCOPE_API_KEY not set in .env"})
        return

    prompt = _build_prompt(experiments, search_bounds, conditions, candidates, avg_uncertainty)

    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        messages = [{"role": "user", "content": prompt}]

        extra_body = {}
        if ENABLE_THINKING:
            extra_body["enable_thinking"] = True

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            stream=True,
            extra_body=extra_body if extra_body else None,
        )

        full_text = ""
        for chunk in stream:
            # Standard content delta
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                yield _sse("token", {"text": token})

            # DashScope thinking content (reasoning_content field)
            if chunk.choices and hasattr(chunk.choices[0].delta, "reasoning_content"):
                rc = chunk.choices[0].delta.reasoning_content
                if rc:
                    full_text += rc
                    yield _sse("thinking", {"text": rc})

        # Parse the full accumulated text into structured suggestions
        suggestions = _parse_suggestions(full_text, search_bounds)
        yield _sse("done", {
            "suggestions": suggestions,
            "raw_response": full_text,
            "avg_uncertainty": avg_uncertainty,
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
) -> Dict[str, Any]:
    """
    Non-streaming fallback. Returns dict with: suggestions, raw_response, avg_uncertainty, error.
    """
    if not API_KEY:
        return {
            "suggestions": [],
            "raw_response": "",
            "avg_uncertainty": avg_uncertainty,
            "error": "DASHSCOPE_API_KEY not set in .env",
        }

    prompt = _build_prompt(experiments, search_bounds, conditions, candidates, avg_uncertainty)

    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        messages = [{"role": "user", "content": prompt}]

        extra_body = {}
        if ENABLE_THINKING:
            extra_body["enable_thinking"] = True

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            extra_body=extra_body if extra_body else None,
        )

        raw_content = response.choices[0].message.content or ""
        suggestions = _parse_suggestions(raw_content, search_bounds)

        return {
            "suggestions": suggestions[:n_suggestions],
            "raw_response": raw_content,
            "avg_uncertainty": avg_uncertainty,
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
