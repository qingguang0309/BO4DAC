"""
Offline ingestion for the BO4DAC RAG knowledge base.

Builds two Chroma collections from local sources:
  - papers       : chunked academic papers, from two source types:
                     (a) MinerU-parsed folders  <parsed-dir>/NNN.pdf-id/full.md   (preferred)
                     (b) raw PDFs in data/papers/                                  (fallback)
  - experiments  : one "experiment card" per row of historical_experiments.csv

Paper DOIs are resolved from the CSV's Source_File column (e.g.
"001.pdf-id_comprehensive_extraction.json" -> folder "001.pdf-id"), with a
regex scan of the paper text as fallback.

Usage:
  python -m rag.ingest                                  # default paths
  python -m rag.ingest --parsed-dir "Solid Sorbent Papers"
  python -m rag.ingest --reset                          # rebuild from scratch

First run downloads the bge-m3 embedding model (~2GB).
"""

import os
import re
import glob
import argparse
import logging
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from rag import store
from rag.embedder import embed

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rag.ingest")

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# 无检索价值的章节（作者信息、致谢、参考文献列表等）
SKIP_SECTION_RE = re.compile(
    r"AUTHOR INFORMATION|Corresponding Author|ORCID|^Notes$|ACKNOWLEDG|REFERENCES"
    r"|ASSOCIATED CONTENT|Supporting Information|Data availability|Declaration"
    r"|Conflict|CRediT",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# MinerU-parsed papers (<parsed-dir>/NNN.pdf-id/full.md)
# ----------------------------------------------------------------------
def _clean_latex(s: str) -> str:
    """Flatten the light LaTeX markup MinerU leaves in full.md into plain text."""
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.replace(r"\cdot", "·").replace(r"\%", "%")
    s = re.sub(r"\\(?:mathrm|mathbf|mathit|text|textit|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"[_^](\w)", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _table_to_text(html: str) -> str:
    """Convert a MinerU single-line HTML table into ' | '-separated text rows."""
    t = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", html)
    t = re.sub(r"</tr>\s*<tr[^>]*>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return t.strip()


def _split_sections(md_text: str) -> List[tuple]:
    """Split markdown into (heading, body) pairs on level-1 headings."""
    sections, cur_head, cur = [], "", []
    for ln in md_text.split("\n"):
        if ln.startswith("# "):
            if cur_head or any(x.strip() for x in cur):
                sections.append((cur_head, "\n".join(cur)))
            cur_head, cur = ln[2:].strip(), []
        else:
            cur.append(ln)
    if cur_head or any(x.strip() for x in cur):
        sections.append((cur_head, "\n".join(cur)))
    return sections


def _load_doi_map(csv_path: str) -> Dict[str, str]:
    """Map paper folder id ('001.pdf-id') -> DOI, from the CSV's Source_File column."""
    m: Dict[str, str] = {}
    if not os.path.isfile(csv_path):
        return m
    try:
        df = pd.read_csv(csv_path, usecols=["Source_File", "DOI"])
    except Exception as e:
        logger.warning(f"DOI map unavailable ({e})")
        return m
    for _, r in df.dropna(subset=["Source_File"]).drop_duplicates().iterrows():
        pid = str(r["Source_File"]).split("_comprehensive")[0].strip()
        doi = str(r["DOI"]).strip() if pd.notna(r["DOI"]) else ""
        if pid and doi and doi != "-" and pid not in m:
            m[pid] = doi
    return m


def parse_mineru_papers(parsed_dir: str, doi_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Chunk MinerU-parsed papers (full.md) into embeddable items with metadata."""
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(parsed_dir):
        logger.info(f"Parsed-papers dir not found: {parsed_dir} (skipping)")
        return items
    dirs = sorted(glob.glob(os.path.join(parsed_dir, "*.pdf-id")))
    if not dirs:
        logger.info(f"No *.pdf-id folders in {parsed_dir} (skipping)")
        return items

    for d in dirs:
        paper_id = os.path.basename(d)
        md_path = os.path.join(d, "full.md")
        if not os.path.isfile(md_path):
            logger.warning(f"  {paper_id}: no full.md — skipped")
            continue
        with open(md_path, encoding="utf-8") as f:
            raw = f.read()

        raw = re.sub(r"<table>.*?</table>", lambda mt: _table_to_text(mt.group(0)),
                     raw, flags=re.S)
        raw = MD_IMAGE_RE.sub(" ", raw)

        sections = _split_sections(raw)
        # 论文标题 = 第一个"像标题"的 heading（跳过预置噪声行和编号章节名）
        title, title_idx = paper_id, -1
        for si, (heading, _) in enumerate(sections):
            h = _clean_latex(heading)
            if h and not re.match(r"^\d", h) and not SKIP_SECTION_RE.search(h) \
                    and h.upper() not in ("ABSTRACT", "INTRODUCTION"):
                title, title_idx = h, si
                break
        doi = doi_map.get(paper_id, "")
        if not doi:
            mdoi = DOI_RE.search(raw)
            doi = mdoi.group(0).rstrip(".)") if mdoi else ""

        n_before = len(items)
        chunk_idx = 0
        for si, (heading, body) in enumerate(sections):
            sec_name = _clean_latex(heading) if heading else ""
            if si <= title_idx:
                sec_name = "Title & Abstract"
            elif sec_name and SKIP_SECTION_RE.search(sec_name):
                continue
            text = _clean_latex(body)
            if len(text.split()) < 15:
                continue
            for chunk in _chunk_words(text):
                # 每个 chunk 自带论文标题+章节上下文，提升独立检索质量
                doc = f"[文献] {title} — {sec_name or 'body'}: {chunk}"
                meta = {
                    "doc_type": "paper_chunk",
                    "source": "parsed_md",
                    "paper_id": paper_id,
                    "source_file": paper_id,
                    "title": title,
                    "section": sec_name or "body",
                    "chunk_index": chunk_idx,
                }
                if doi:
                    meta["doi"] = doi
                items.append({"id": f"{paper_id}::c{chunk_idx}", "text": doc, "metadata": meta})
                chunk_idx += 1
        logger.info(f"  {paper_id}: {len(items) - n_before} chunks "
                    f"(doi={doi or 'n/a'}) — {title[:60]}")
    return items


# ----------------------------------------------------------------------
# Raw PDF papers (fallback path)
# ----------------------------------------------------------------------
def _chunk_words(text: str, size: int = 450, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks


def parse_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(pdf_dir):
        logger.warning(f"PDF dir not found: {pdf_dir} (skipping papers)")
        return items
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        logger.info(f"No PDFs in {pdf_dir} (skipping papers — add PDFs and re-run)")
        return items

    import fitz  # PyMuPDF
    for path in pdfs:
        fname = os.path.basename(path)
        try:
            doc = fitz.open(path)
            full = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to parse {fname}: {e}")
            continue
        full = re.sub(r"\s+", " ", full).strip()
        if not full:
            logger.warning(f"No text layer in {fname} (scanned PDF? needs OCR) — skipped")
            continue
        m = DOI_RE.search(full)
        doi = m.group(0).rstrip(".)") if m else ""
        chunks = _chunk_words(full)
        for idx, chunk in enumerate(chunks):
            meta = {"doc_type": "paper_chunk", "source_file": fname, "chunk_index": idx}
            if doi:
                meta["doi"] = doi
            items.append({"id": f"{fname}::c{idx}", "text": chunk, "metadata": meta})
        logger.info(f"  {fname}: {len(chunks)} chunks (doi={doi or 'n/a'})")
    return items


# ----------------------------------------------------------------------
# Experiment cards (from historical_experiments.csv)
# ----------------------------------------------------------------------
def _v(row, col):
    val = row.get(col)
    if pd.isna(val):
        return None
    return val


def _num(row, col):
    val = _v(row, col)
    if val is None:
        return None
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _build_card(row) -> str:
    support = _v(row, "Support") or "未知载体"
    support_type = _v(row, "Support_Type")
    amine = _v(row, "Amine_1_or_Additive_1")
    amine_str = "无胺(裸载体)" if amine in (None, 0, "0") else str(amine)

    head = f"[实验记录] {support}"
    if support_type:
        head += f"（{support_type}）"
    head += f" 负载 {amine_str}"
    parts = [head + "。"]

    det = []
    if _num(row, "Organic_Content_pct") is not None:
        det.append(f"有机含量 {_num(row, 'Organic_Content_pct')} wt%")
    if _num(row, "N_Content_mmol_g") is not None:
        det.append(f"N含量 {_num(row, 'N_Content_mmol_g')} mmol/g")
    if _num(row, "BET_Bare_Surface_Area_m2_g") is not None:
        det.append(f"裸载体BET {_num(row, 'BET_Bare_Surface_Area_m2_g')} m²/g")
    if _num(row, "Average_Bare_Pore_Diameter_nm") is not None:
        det.append(f"平均孔径 {_num(row, 'Average_Bare_Pore_Diameter_nm')} nm")
    if det:
        parts.append("，".join(det) + "。")

    cond = []
    if _num(row, "Adsorption_Temperature_C") is not None:
        cond.append(f"{_num(row, 'Adsorption_Temperature_C')}°C")
    if _v(row, "CO2_Concentration_vol_pct") is not None:
        cond.append(f"CO₂ {_v(row, 'CO2_Concentration_vol_pct')} vol%")
    if _num(row, "Relative_Humidity_pct") is not None:
        cond.append(f"RH {_num(row, 'Relative_Humidity_pct')}%")
    if _num(row, "Flow_Rate_mL_min") is not None:
        cond.append(f"流速 {_num(row, 'Flow_Rate_mL_min')} mL/min")
    if _v(row, "CO2_Test_Method") is not None:
        cond.append(f"方法 {_v(row, 'CO2_Test_Method')}")
    if cond:
        parts.append("测试条件：" + "，".join(cond) + "。")

    res = []
    if _num(row, "CO2_Capacity_mmol_g") is not None:
        res.append(f"CO₂容量 {_num(row, 'CO2_Capacity_mmol_g')} mmol/g")
    if _num(row, "Amine_Efficiency_mmol_mmol") is not None:
        res.append(f"胺效率 {_num(row, 'Amine_Efficiency_mmol_mmol')} mmol/mmol")
    if _num(row, "Heat_of_Adsorption_kJ_mol") is not None:
        res.append(f"吸附热 {_num(row, 'Heat_of_Adsorption_kJ_mol')} kJ/mol")
    if _num(row, "Capacity_Loss_Stability_pct") is not None:
        res.append(f"循环容量损失 {_num(row, 'Capacity_Loss_Stability_pct')}%")
    if res:
        parts.append("结果：" + "，".join(res) + "。")

    doi = _v(row, "DOI")
    if doi:
        parts.append(f"来源DOI: {doi}。")
    return " ".join(parts)


def build_experiment_cards(csv_path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.isfile(csv_path):
        logger.warning(f"CSV not found: {csv_path} (skipping experiments)")
        return items
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        if _num(row, "CO2_Capacity_mmol_g") is None:
            continue  # skip rows without the optimization target
        meta: Dict[str, Any] = {"doc_type": "experiment", "row_index": int(i)}
        for key, col in [("record_id", "Record_ID"), ("support", "Support"),
                         ("amine", "Amine_1_or_Additive_1"), ("doi", "DOI"),
                         ("method", "CO2_Test_Method")]:
            val = _v(row, col)
            if val is not None:
                meta[key] = str(val)
        for key, col in [("organic_content_pct", "Organic_Content_pct"),
                         ("temperature_c", "Adsorption_Temperature_C"),
                         ("capacity_mmol_g", "CO2_Capacity_mmol_g")]:
            val = _num(row, col)
            if val is not None:
                meta[key] = val
        items.append({"id": f"exp-{i}", "text": _build_card(row), "metadata": meta})
    logger.info(f"  built {len(items)} experiment cards from {len(df)} rows")
    return items


# ----------------------------------------------------------------------
# Embed + store
# ----------------------------------------------------------------------
def _store(items: List[Dict[str, Any]], coll_name: str, batch: int = 128) -> int:
    if not items:
        return 0
    col = store.get_or_create(coll_name)
    for s in range(0, len(items), batch):
        part = items[s:s + batch]
        embs = embed([it["text"] for it in part])
        col.upsert(
            ids=[it["id"] for it in part],
            embeddings=embs,
            documents=[it["text"] for it in part],
            metadatas=[it["metadata"] for it in part],
        )
        logger.info(f"  {coll_name}: stored {min(s + batch, len(items))}/{len(items)}")
    return len(items)


def main():
    ap = argparse.ArgumentParser(description="Build the BO4DAC RAG knowledge base.")
    ap.add_argument("--parsed-dir", default="Solid Sorbent Papers",
                    help="dir of MinerU-parsed papers (NNN.pdf-id/full.md)")
    ap.add_argument("--pdf-dir", default="data/papers")
    ap.add_argument("--csv", default="data/historical_experiments.csv")
    ap.add_argument("--db-path", default=store.RAG_DB_PATH)
    ap.add_argument("--reset", action="store_true", help="delete existing collections first")
    args = ap.parse_args()

    client = store.get_client(args.db_path)
    if args.reset:
        for name in (store.COLL_PAPERS, store.COLL_EXPERIMENTS):
            try:
                client.delete_collection(name)
                logger.info(f"reset: dropped collection '{name}'")
            except Exception:
                pass

    logger.info("Parsing MinerU-parsed papers ...")
    papers = parse_mineru_papers(args.parsed_dir, _load_doi_map(args.csv))
    logger.info("Parsing raw PDFs ...")
    papers += parse_pdfs(args.pdf_dir)
    logger.info("Building experiment cards ...")
    experiments = build_experiment_cards(args.csv)

    logger.info("Embedding + storing (first run downloads bge-m3 ~2GB) ...")
    n_p = _store(papers, store.COLL_PAPERS)
    n_e = _store(experiments, store.COLL_EXPERIMENTS)
    logger.info(f"DONE. papers_chunks={n_p}, experiments={n_e}, db={args.db_path}")


if __name__ == "__main__":
    main()
