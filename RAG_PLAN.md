# BO4DAC RAG 系统执行方案

> 目标：为 BO4DAC（DAC 催化剂贝叶斯优化系统）增加一套**本地知识库检索增强（RAG）**，
> 让 LLM 给出的配方建议不再只依赖通用知识 + Tavily 联网搜索，而是**优先检索本地权威语料**
> （DAC/CO₂ 捕集学术文献 + 系统自身 334 条实验记录），并与 Tavily 联网结果**互补合并**，
> 共同注入 prompt，输出带 `[N]` 引用、可溯源的建议。

- **技术栈**：本地 `BAAI/bge-m3` 嵌入（中英双语、离线、免费）+ `ChromaDB` 持久化向量库
- **与现有系统关系**：复用 `llm_service.py` 已有的「检索结果注入 prompt + `search_source` SSE 回传来源」模式，**最小侵入**地并入一条 RAG 来源
- **与 Tavily 关系**：互补。`ENABLE_RAG` 与 `ENABLE_WEB_SEARCH` 各自独立开关，两路来源统一编号合并

---

## 1. 为什么做 / 能解决什么

当前 `llm_service.py` 的建议链路：
```
实验上下文 → _build_prompt() → [Web Search Results](Tavily) 注入 → qwen3.7-max → 5 条建议 + [N] 引用
```

存在的局限：
1. **Tavily 是通用网页搜索**，命中的不一定是高质量 DAC 文献，且每次实时检索、受联网/代理影响、有调用配额。
2. **系统自身的 334 条实验记录**（`data/historical_experiments.csv`，含 DOI、载体、胺、条件、容量、胺效率、稳定性）目前只有**当前 session 过滤后的少量数据**进入 prompt（`[All Experimental Data]`），**全量历史知识没有被语义检索利用**。
3. 模型引用容易"悬空"——`[N]` 缺乏稳定、可控、可溯源的本地证据。

RAG 带来的价值：
- **权威性**：从你精选的 DAC 文献 + 真实实验库检索，证据质量可控。
- **离线 + 零调用成本**：嵌入与检索全本地，断网也能跑。
- **可溯源**：每条建议的 `[N]` 对应具体文献片段或实验记录（带 DOI）。
- **盘活历史数据**：334 条记录可按"与当前搜索空间/实验条件语义相似"被检索出来，跨越 session 边界。

---

## 2. 整体架构

```
                          ┌─────────────────── 离线入库（一次性 / 增量）───────────────────┐
  data/papers/*.pdf  ──▶  │  PDF 解析 → 清洗 → 分块（~600 token, 重叠）→ 元数据(DOI/页码)   │
  historical_…csv    ──▶  │  逐行 → "实验卡片"自然语言模板 → 元数据(载体/胺/条件/容量/DOI)  │
                          │                       │                                        │
                          │             bge-m3 嵌入 (1024 维)                              │
                          │                       ▼                                        │
                          │     ChromaDB 持久化  data/rag_chroma/                          │
                          │       ├─ collection: papers      (doc_type=paper_chunk)        │
                          │       └─ collection: experiments (doc_type=experiment)         │
                          └──────────────────────────────────────────────────────────────┘
                                                  ▲
                                                  │ 在线检索
  ┌────────────────────────── llm_service.py（在线推理）───────────────────────────────┐
  │  实验上下文(search_bounds, conditions)                                              │
  │        │                                                                            │
  │        ├──▶ _run_rag_retrieval()  ─┐  (rag_service.py：bge-m3 编码 query → Chroma)  │
  │        │                            ├─▶ _merge_sources() 统一编号 [1..N]            │
  │        └──▶ _run_web_search()  ─────┘  (现有 Tavily)                                │
  │                            │                                                        │
  │             _build_prompt(rag_block, web_block) → qwen3.7-max(stream)               │
  │                            │                                                        │
  │             SSE: search_source(合并来源) / thinking / token / done                  │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

设计要点：**RAG 完全镜像现有 Tavily 的接口形状**——
`_run_web_search(search_bounds, conditions) -> (sources, block)`，
新增的 `_run_rag_retrieval(search_bounds, conditions) -> (sources, block)` 返回同样结构，
因此可以无缝合并、复用现有的 `search_source` SSE 事件和前端展示。

---

## 3. 技术选型与理由

| 组件 | 选型 | 理由 |
|---|---|---|
| 嵌入模型 | `BAAI/bge-m3`（sentence-transformers / FlagEmbedding） | 中英双语、长上下文(8192)、检索效果强、可离线、免费；你已装 torch |
| 向量库 | `ChromaDB`（持久化本地模式） | 轻量、零运维、`PersistentClient` 落盘到 `data/rag_chroma/`；元数据过滤方便 |
| PDF 解析 | `PyMuPDF (fitz)`，扫描件回退 OCR（可选 `pytesseract`） | 速度快、版面与文本质量好 |
| 分块 | 递归字符/Token 分块，~600 token、重叠 ~100 | 兼顾召回与上下文完整；段落优先 |
| 重排（可选，二期） | `BAAI/bge-reranker-v2-m3` | 召回后精排，进一步降噪 |

**新增依赖**（追加到 `requirements.txt`）：
```
chromadb>=0.5.0
sentence-transformers>=3.0.0      # 或 FlagEmbedding>=1.2.0
PyMuPDF>=1.24.0
```
> bge-m3 首次会从 HuggingFace 下载（约 2GB）。国内网络可设 `HF_ENDPOINT=https://hf-mirror.com`，
> 详见 §10 风险与回退。

---

## 4. 语料库设计

### 4.1 学术文献 PDF（`data/papers/`）
- **放置**：把 DAC/CO₂ 捕集相关 PDF 放进 `data/papers/`（该目录随 `data/` 已被 `.gitignore` 忽略，不入库 git）。
- **解析**：PyMuPDF 逐页提取文本；去页眉页脚、参考文献噪声（可选）。
- **DOI 提取**：正则从首页/全文匹配 `10.\d{4,}/\S+`，写入 chunk 元数据，**与实验 CSV 的 DOI 字段打通**。
- **分块**：~600 token / 块，重叠 100；元数据 `{source_file, doi, page, doc_type:"paper_chunk"}`。

### 4.2 系统实验数据（`data/historical_experiments.csv`，334 行 × 58 列）
- **粒度**：**一行一条"实验卡片"**（不是整表一块），让"相似配方/条件下的历史结果"可被语义检索。
- **卡片模板**（用真实列名拼成自然语言，便于嵌入与引用）：
  ```
  [实验记录] {Support}（{Support_Type}）负载 {Amine_1_or_Additive_1}
  （有机含量 {Organic_Content_pct} wt%，N 含量 {N_Content_mmol_g} mmol/g）。
  裸载体 BET {BET_Bare_Surface_Area_m2_g} m²/g、平均孔径 {Average_Bare_Pore_Diameter_nm} nm。
  测试条件：{Adsorption_Temperature_C} °C，CO₂ {CO2_Concentration_vol_pct} vol%，
  RH {Relative_Humidity_pct}%，流速 {Flow_Rate_mL_min} mL/min，方法 {CO2_Test_Method}。
  结果：CO₂ 容量 {CO2_Capacity_mmol_g} mmol/g，胺效率 {Amine_Efficiency_mmol_mmol} mmol/mmol，
  吸附热 {Heat_of_Adsorption_kJ_mol} kJ/mol，半饱和时间 {Time_to_Half_Saturation} s。
  循环稳定性：容量损失 {Capacity_Loss_Stability_pct}%。来源 DOI: {DOI}。
  ```
- **元数据**（供后续按条件过滤）：
  `{record_id, support, amine, organic_content_pct, temperature_c, co2_vol_pct, capacity_mmol_g, doi, doc_type:"experiment"}`
- **去重/清洗**：跳过 `CO2_Capacity_mmol_g` 为空的行；数值字段缺失填 "N/A"。

> **与现有 prompt 不重复**：`_build_prompt` 现有的 `[All Experimental Data]` 只含**当前 session 过滤后**的数据；
> RAG experiments 检索的是**全量 334 条**里与当前搜索空间语义最相近的若干条，二者互补不冲突。

---

## 5. 实施步骤（分阶段，建议按序）

### Phase 0 — 环境与依赖（~0.5 天）
- [ ] 追加依赖到 `requirements.txt`，`pip install` 上述 3 个包
- [ ] 新建目录：`rag/`（代码）、`data/papers/`（放 PDF）；确认 `data/` 已被 gitignore
- [ ] 在 `.env` 增加 RAG 配置项（见 §7）
- [ ] 首次下载 bge-m3，写一个 `rag/embedder.py` 单例加载器（避免每次请求重载模型）

### Phase 1 — 离线入库管道 `rag/ingest.py`（~1.5 天）
- [ ] `parse_pdfs(dir)`：PyMuPDF 解析 + 分块 + DOI/页码元数据
- [ ] `build_experiment_cards(csv)`：逐行套 §4.2 模板 + 元数据
- [ ] `embed_and_store()`：bge-m3 批量编码 → 写入 Chroma 两个 collection（`papers` / `experiments`）
- [ ] CLI：`python -m rag.ingest --pdf-dir data/papers --csv data/historical_experiments.csv`
- [ ] 支持增量：按 `record_id` / `source_file+page` 做幂等 upsert（重复跑不重复入库）
- [ ] 入库后打印统计：papers chunks 数、experiments 数、维度、落盘路径

### Phase 2 — 检索服务 `rag/rag_service.py`（~1 天）
- [ ] `_build_rag_query(search_bounds, conditions)`：复用 `llm_service._build_search_queries` 的思路，把载体/胺/条件拼成一段检索文本
- [ ] `_run_rag_retrieval(search_bounds, conditions) -> (sources, block)`：
  - 用 bge-m3 编码 query；分别从 `papers`（top‑k=`RAG_TOP_K_PAPERS`）和 `experiments`（top‑k=`RAG_TOP_K_EXPERIMENTS`）检索
  - 可选元数据预过滤（如只取 `support ∈ 当前搜索空间` 的实验卡片）
  - 组装成与 `_run_web_search` **完全相同的返回结构**：`sources=[{index,title,url/doi}]`、`block="[1] ...\n  片段\n  来源"`
  - 失败/空结果优雅降级返回 `([], "")`

### Phase 3 — 接入 `llm_service.py`（~1 天）
- [ ] `_build_prompt` 增加 `rag_results_block` 参数，新增 `[Local Knowledge Base]` 区块（与现有 `[Web Search Results]` 并列）
- [ ] 新增 `_merge_sources(rag_sources, web_sources)`：统一连续编号 `[1..N]`、按 URL/DOI 去重，保证 `[N]` 引用一致
- [ ] `stream_llm_suggestions` / `get_llm_suggestions`：
  ```python
  rag_sources, rag_block = _run_rag_retrieval(search_bounds, conditions)   # 新增
  web_sources, web_block = _run_web_search(search_bounds, conditions)      # 现有
  sources = _merge_sources(rag_sources, web_sources)
  if sources: yield _sse("search_source", {"sources": sources})            # 复用现有事件
  prompt = _build_prompt(..., rag_results_block=rag_block, web_results_block=web_block)
  ```
- [ ] prompt STEP 2 文案更新：先看 `[Local Knowledge Base]`，再参考 `[Web Search Results]`
- [ ] `.env` 开关：`ENABLE_RAG=False` 时 `_run_rag_retrieval` 直接返回空，行为回退到当前纯 Tavily

### Phase 4 — 前端/可观测（可选，~0.5 天）
- [ ] 前端 `search_source` 已有展示；可在来源上加 `type: rag|web` 标签区分本地/联网（在 `sources` item 里加字段）
- [ ] 新增 `POST /api/rag/reindex` 端点：上传新实验后一键重建实验卡片索引（或复用现有提交流程触发增量 upsert）

### Phase 5 — 评估与调优（~1 天）
- [ ] 见 §8

**总工作量估计：约 5～6 个工作日**（不含 PDF 语料收集时间）。

---

## 6. 与现有代码的集成点（清单）

| 文件 | 改动 |
|---|---|
| `requirements.txt` | + chromadb, sentence-transformers, PyMuPDF |
| `.env` | + RAG 配置项（§7） |
| `rag/embedder.py`（新增） | bge-m3 单例加载（CPU/MPS） |
| `rag/ingest.py`（新增） | 离线入库 CLI |
| `rag/rag_service.py`（新增） | `_run_rag_retrieval()`、镜像 `_run_web_search` 结构 |
| `llm_service.py` | `_build_prompt(+rag_results_block)`、`_merge_sources()`、两个调用函数接入 RAG |
| `data/papers/`（新增目录） | 放 PDF（gitignore） |
| `data/rag_chroma/`（运行时生成） | Chroma 持久化（gitignore） |
| 前端（可选） | 来源加 `type` 标签、`/api/rag/reindex` |

> **核心优势**：因为 RAG 输出结构与 Tavily 一致，`app.py`、SSE 协议、前端**几乎不用改**——
> 这是当初把 Tavily 设计成"返回 (sources, block) 并注入 prompt"留下的扩展点。

---

## 7. `.env` 新增配置项

```ini
# ── RAG ──
ENABLE_RAG=True
RAG_DB_PATH=data/rag_chroma
EMBEDDING_MODEL=BAAI/bge-m3
RAG_TOP_K_PAPERS=4
RAG_TOP_K_EXPERIMENTS=4
RAG_MIN_SCORE=0.3                 # 相似度阈值，低于则丢弃
# 国内下载加速（可选）
# HF_ENDPOINT=https://hf-mirror.com
```
> `.env` 仍在 `.gitignore` 内，不会被推送（与现有 API key 同处理）。

---

## 8. 评估方案

1. **检索质量（retrieval）**
   - 自建小金标集：~15 个 (query → 期望命中 DOI) 对，测 `recall@k` / `MRR`。
   - 人工抽查 top‑k：相关 / 部分相关 / 无关 三档打分。
2. **引用扎实度（grounding）**
   - 校验建议里的 `[N]` 是否真对应被检索片段；统计"悬空引用率"。
3. **端到端 A/B**
   - 同一组实验上下文，分别在 `仅Tavily / 仅RAG / RAG+Tavily` 下生成建议，比较：来源权威性、与已知文献一致性、配方多样性。
4. **性能**
   - 单次检索延迟（bge-m3 编码 + Chroma 查询，目标 < 1s）；入库吞吐（334 条 + PDF）。

---

## 9. 端到端示例（RAG+Tavily 合并）

输入条件：`Support=SBA-15, Amine=PEI, T=25°C, CO₂=0.04 vol%（DAC）`

1. `_build_rag_query` → `"PEI 负载 SBA-15 直接空气捕集 CO2 容量 25°C 低浓度"`
2. RAG 命中：
   - experiments：`[1] SBA-15 + PEI 50wt% → 1.92 mmol/g @25°C TGA（DOI 10.1021/...）`
   - papers：`[2] "long-channel SBA-15-PEI 在 400ppm 下…"（页 4）`
3. Tavily 命中：`[3] ScienceDirect…`、`[4] ResearchGate…`
4. `_merge_sources` → `[1..4]` 统一编号 → `search_source` SSE 回传前端
5. `_build_prompt` 注入 `[Local Knowledge Base]([1][2])` + `[Web Search Results]([3][4])`
6. qwen3.7-max 输出 5 条建议，reasoning 形如 *"SBA-15 大孔道利于 50wt% PEI 分散 [1][2]…"*

---

## 10. 风险与回退

| 风险 | 缓解 |
|---|---|
| bge-m3 下载慢/失败（~2GB） | 设 `HF_ENDPOINT=https://hf-mirror.com`；或离线手动放到 `~/.cache/huggingface` |
| 首次加载模型慢 | 单例加载 + 应用启动时预热；mac 可用 MPS 加速 |
| 扫描版 PDF 无文本层 | 检测空文本 → 回退 OCR（pytesseract），或人工剔除 |
| CPU 入库慢 | 离线批量 ingest（一次性）；查询只编码单条 query，很快 |
| Flask debug reloader 与 Chroma 文件锁 | Chroma 用 `PersistentClient` 只读查询；入库走独立 CLI，不在 web 进程并发写 |
| RAG 检索为空/异常 | `_run_rag_retrieval` 内部 try/except，返回 `([],"")`，自动退回 Tavily-only（与现有降级一致） |
| 索引过期（新增实验未入库） | 提交实验后触发增量 upsert，或定期 `python -m rag.ingest` |
| RAG 与 Tavily 来源重复 | `_merge_sources` 按 DOI/URL 去重 |

---

## 11. 后续可扩展（二期）

- **重排**：召回后用 `bge-reranker-v2-m3` 精排，提升 top‑k 精度。
- **混合检索**：bge-m3 原生支持 dense + sparse(lexical)，可做混合召回提升专有名词命中。
- **按条件硬过滤**：先用元数据（温度/CO₂ 浓度区间、载体）过滤，再语义排序，让"相似条件"更精准。
- **自动重索引**：实验提交 API 成功后异步 upsert 该条卡片。
- **引用回链**：前端点击 `[N]` 跳转 DOI / 打开 PDF 对应页。

---

## 附：建议的新增目录结构

```
BO4DAC/
├── rag/
│   ├── __init__.py
│   ├── embedder.py        # bge-m3 单例
│   ├── ingest.py          # 离线入库 CLI（PDF + CSV → Chroma）
│   └── rag_service.py     # _run_rag_retrieval()
├── data/
│   ├── papers/            # 放 PDF（gitignore）
│   ├── rag_chroma/        # Chroma 持久化（运行时生成，gitignore）
│   └── historical_experiments.csv
├── llm_service.py         # 接入 RAG（+rag_results_block, _merge_sources）
└── RAG_PLAN.md            # 本文档
```
