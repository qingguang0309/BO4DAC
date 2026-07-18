# BO4DAC 配置与改造工作总结

## 0. 总览

| # | 任务 | 结果 | 对应提交 |
|---|---|---|---|
| 1 | 克隆仓库到本地 | ✅ | — |
| 2 | 了解项目 + 在本机跑起来 | ✅ 服务运行于 `http://127.0.0.1:5001` | — |
| 3 | LLM 服务适配（先 GpuGeek/GLM）+ 修复 .gitignore | ✅ | `6abef34` |
| 4 | 配置 Git：私人库 push + 保留 upstream 拉取 | ✅ 双远程 | `6abef34` |
| 5 | 更新 .env（切回 DashScope qwen3.7-max） | ✅ | （.env 本地，不入库） |
| 6 | 集成 Tavily 联网搜索 | ✅ | `daa8ef2` |
| 7 | 落地本地 RAG 知识库（文献 + 实验数据） | ✅ | `bf337f4` |
| 8 | 文献全文入库（MinerU 解析件）+ Web 搜索缓存与本地优先去重 | ✅ | 待提交（详见 §10） |

**当前系统能力**：贝叶斯优化（GP + Expected Improvement）+ LLM 配方建议（DashScope `qwen3.7-max`），
建议同时由**本地 RAG 知识库**（314 条实验记录 + 5 篇文献全文共 124 个 chunk）和 **Tavily 联网搜索**
两路证据支撑，输出带统一 `[N]` 引用、可溯源。联网搜索带 **7 天持久缓存**（迭代不重复搜索），
且与本地文献 DOI 重复的网页结果会被丢弃（本地全文优先）。

---

## 1. 克隆仓库

- 从 `https://github.com/decarbon-shenzhen/BO4DAC` 克隆到
  `…/20260603-BORA/BO4DAC`。

## 2. 项目了解与本机运行

**项目本质**：基于 Flask 的 Web 应用，用**贝叶斯优化**为 DAC（直接空气捕集）CO₂ 捕集催化剂
推荐下一组最值得做实验的配方。核心模块：
- `app.py`（Flask 主程序，端口 5001）
- `optimization_system.py`（`DACOptimizer`：高斯过程 + Expected Improvement，BoTorch/GPyTorch）
- `llm_service.py`（LLM 辅助建议）
- `database_manager.py` / `config_manager.py` / `encoder.py` / `visualizer.py`
- `templates/` + `static/`（前端）

**环境**：conda 环境 `ccus`（Python 3.10.19），解释器 `/opt/miniconda3/envs/ccus/bin/python`。

**踩到的坑**：`requirements.txt` 漏了代码实际 import 的 `openai` 和 `python-dotenv`，
直接按 README 装会启动报 `ModuleNotFoundError`。已补装并最终写回 `requirements.txt`。

**已安装的缺失依赖**：`flask`、`gpytorch`、`flask-cors`、`requests`、`botorch`、`dashscope`、
`openai`、`python-dotenv`（torch/numpy/pandas/sklearn 等环境里已有）。

## 3. LLM 服务适配 + 安全修复

**起因**：原 `llm_service.py` 用的是 OpenAI **Responses API** + 内置 `web_search` 工具（DashScope 特有），
而当时给到的是 **GpuGeek 平台**（仅支持标准 `chat.completions`，模型 `Vendor2/GLM-5.1`）。

**改动**：
- `responses.create()` → `chat.completions.create()`
- 去掉不支持的内置 `web_search` 工具
- 正确处理 GLM 的 `reasoning_content`（作为 `thinking` 流式事件）
- 新增 `MAX_TOKENS`（推理模型耗 token 多）
- 安装 `socksio` 修复本地 SOCKS 代理（`127.0.0.1:7899`）导致的 httpx 报错

**安全修复（重要）**：原 `.gitignore` 是 **UTF-16 编码**，git 无法解析 → `.env`（含 API Key）
**未被忽略**，险些泄露。已重写为 UTF-8，确认 `.env`、`__pycache__/`、`data/`、`models/` 等被正确忽略。

## 4. Git 双远程配置

目标：本地改动 push 到私人库，同时保留从原始库拉取更新的能力。

| 远程 | 地址 | 用途 |
|---|---|---|
| `origin` | `github.com/qingguang0309/BO4DAC-private`（私人库，已用 `gh` 创建） | push 目标 |
| `upstream` | `github.com/decarbon-shenzhen/BO4DAC`（原始库） | 拉取更新 |

**日常用法**：
```bash
# 推送本地改动到私人库
git add -A && git commit -m "..." && git push

# 拉取原始库更新并合并
git fetch upstream && git merge upstream/master && git push
```

## 5. 切换到 DashScope（更新 .env）

用新的 `.env` 替换，从 GpuGeek 切回**阿里云 DashScope（通义千问）**：
- `MODEL=qwen3.7-max`，`BASE_URL=…dashscope…/compatible-mode/v1`
- `ENABLE_THINKING=True`、`ENABLE_WEB_SEARCH=True`
- 新增 `TAVILY_API_KEY`（Tavily 联网搜索）

> `.env` 始终被 gitignore 忽略，**从不入库**。换机器需手动重建（见 §8）。

## 6. Tavily 联网搜索集成

由于 GpuGeek/DashScope-compatible 的 `chat.completions` 不支持内置 web_search，改用 **Tavily REST API**：
- `_build_search_queries()`：从搜索空间 + 实验条件自动生成针对性学术检索词
- `_tavily_search()`：调 Tavily，失败优雅降级
- `_run_web_search()`：去重、组装，注入 prompt，并通过 `search_source` SSE 事件回传来源
- prompt 新增 `[Web Search Results]` 区块，模型据此输出 `[N]` 引用

**验证**：端到端流式测试命中真实文献（ScienceDirect/ResearchGate/OSTI），推理含真实引用。

## 7. 本地 RAG 知识库（核心成果）

详见 [`RAG_PLAN.md`](./RAG_PLAN.md)（执行方案）。

**技术栈**：本地 `BAAI/bge-m3` 嵌入（中英双语、离线、免费）+ `ChromaDB` 持久化向量库。
**与 Tavily 关系**：互补——本地权威语料 + 联网最新进展，两路来源统一编号合并。

**新增 `rag/` 包**：
| 文件 | 作用 |
|---|---|
| `rag/embedder.py` | bge-m3 单例（可 `EMBEDDING_DEVICE=mps` 提速） |
| `rag/store.py` | ChromaDB 封装（`papers` / `experiments` 两集合，cosine） |
| `rag/ingest.py` | 离线入库 CLI：PDF（PyMuPDF 分块）+ CSV（逐行→实验卡片）→ 向量库 |
| `rag/rag_service.py` | `run_rag_retrieval()`，镜像 Tavily 接口，优雅降级 |

**`llm_service.py` 接入**：
- `_assemble_context()`：把 RAG + Tavily 两路来源**统一连续编号 `[1..N]`**、按 URL/DOI 去重
- prompt 新增 `[Local Knowledge Base]` 区块（与 `[Web Search Results]` 并列），并提示优先采信本地语料
- 两个建议函数（流式/非流式）都先 RAG 检索再联网，合并后注入

**关键设计**：RAG 输出结构与 Tavily 完全一致 → `app.py`、SSE 协议、前端**几乎不用改**。

**语料设计亮点**：实验 CSV（334 行 × 58 列）与 PDF 通过 **DOI 天然关联**，可交叉印证。
实验数据逐行转成自然语言"实验卡片"（载体/胺/条件/容量/胺效率/稳定性 + DOI），可语义检索。

**入库结果**：334 行 → **314 条实验卡片**入库（跳过 20 行无 `CO2_Capacity_mmol_g`）。
PDF 暂未提供，`papers` 集合为空（加 PDF 后重跑 `python -m rag.ingest` 即生效）。

**端到端验证**：
- 检索：查询命中 4 条高相关历史实验（SBA-15+TEPA，相似度 0.68~0.69）
- 合并：`search_source` 共 12 个来源——`[1-4]` 本地实验、`[5-12]` 联网，统一编号
- 引用：模型推理真实引用本地实验 `[1,4]`（如"TEPA/SBA-15 在 60wt% 达峰、70wt% 因孔堵塞下降"）+ 联网 `[5]`

---

## 8. 当前系统状态与使用

### 启动服务
```bash
conda activate ccus      # 或直接用 /opt/miniconda3/envs/ccus/bin/python
cd <…>/BO4DAC
python app.py            # → http://127.0.0.1:5001
```
> 网页端**首次**点"生成建议"会把 bge-m3 载入内存（约 10 秒，模型已缓存到本地），之后很快。

### `.env` 关键配置（本地，不入库）
```ini
DASHSCOPE_API_KEY=sk-…              # DashScope key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen3.7-max
ENABLE_THINKING=True
ENABLE_WEB_SEARCH=True              # Tavily 联网
TAVILY_API_KEY=tvly-…
# ── RAG ──
ENABLE_RAG=True
RAG_DB_PATH=data/rag_chroma
EMBEDDING_MODEL=BAAI/bge-m3
RAG_TOP_K_PAPERS=6
RAG_TOP_K_EXPERIMENTS=4
RAG_MIN_SCORE=0.3
RAG_MAX_CHUNKS_PER_PAPER=2         # 同一篇文献最多保留的检索 chunk 数
# ── Web 搜索缓存 ──
WEB_CACHE_TTL_DAYS=7               # 缓存有效期；缓存文件 data/web_search_cache.json
# EMBEDDING_DEVICE=mps             # Apple Silicon 提速（可选）
```

### RAG 入库 / 更新
```bash
# 把 DAC/CO2 捕集 PDF 放进 data/papers/，然后：
python -m rag.ingest               # 增量入库（实验卡片幂等 upsert，新增 PDF）
python -m rag.ingest --reset       # 需要重建时
```

### 开关组合
- 只用联网、不用本地库：`.env` 设 `ENABLE_RAG=False`
- 完全离线、只用本地库：`ENABLE_WEB_SEARCH=False`
- 两者任一失败都会优雅降级，不影响贝叶斯优化主功能

---

## 9. 提交记录（私人库 master）

| 提交 | 内容 |
|---|---|
| `bf337f4` | Add local RAG knowledge base (papers + experiment records) |
| `daa8ef2` | Add Tavily web search for literature-grounded LLM suggestions |
| `6abef34` | Adapt LLM service to GpuGeek/GLM and fix .gitignore encoding |
| `521f98c` | （原始库基线） |

---

## 10.（2026-07-17 新增）文献全文入库 + Web 搜索缓存

**背景**：公司反馈两个问题——LLM 联网搜索在 BO 迭代时**重复搜索同样的文献**（浪费额度、慢），
且网页摘要**覆盖不全**。方案：把已解析的文献全文放进本地知识库作为主要证据源，联网只做补充并加缓存。

### 文献全文入库（`rag/ingest.py`）
- 新增 `parse_mineru_papers()`：扫描 `Solid Sorbent Papers/*.pdf-id/full.md`（MinerU 解析件，
  阅读顺序正确、含表格），按章节切分 + ~450 词滑窗分块；HTML 表格转竖线分隔文本（容量数据在表里）；
  轻量 LaTeX 清理；跳过作者信息/致谢/参考文献等无检索价值章节。
- 每个 chunk 元数据：`paper_id`、论文标题、章节名、DOI——DOI 优先取自
  `historical_experiments.csv` 的 `Source_File ↔ DOI` 映射（与实验记录天然交叉关联），正则兜底。
- **测试结果**：5 篇文献 → 124 chunks 全部入库；4 篇带 DOI（002 原文与 CSV 均无 DOI，降级无链接）。
- 原裸 PDF 路径（`data/papers/`）保留为兜底；`python -m rag.ingest --parsed-dir "Solid Sorbent Papers"` 可自定义目录。

### 检索增强（`rag/rag_service.py`）
- **多查询检索**：与联网搜索相同的"每载体×胺"多条查询 + 1 条综合查询（最多 4 条），
  各自检索后按 chunk 去重合并（同一 chunk 取最高分）——本地覆盖面与联网对齐。
- 同一篇文献最多保留 2 个最高分 chunk（`RAG_MAX_CHUNKS_PER_PAPER`），避免单篇刷屏。
- 文献命中展示为 `文献: <论文标题>（§章节）`，URL 为 DOI 链接，前端可点击溯源。
- 新增 `get_local_paper_dois()`：返回本地文献 DOI 集合（带缓存），供联网结果去重。

### Web 搜索缓存 + 本地优先（`llm_service.py`）
- **持久缓存**：`data/web_search_cache.json`，按查询词缓存 Tavily 结果，TTL 默认 7 天
  （`WEB_CACHE_TTL_DAYS`）。同一 session 迭代 N 次，Tavily 只真实调用 1 次；
  搜索失败不写缓存并回退旧缓存。实测第 2 次调用 0.0s 全命中。
- **本地优先去重**：网页结果 URL 含本地文献 DOI 的直接丢弃（模型看本地全文，不看同篇网页摘要）。
- `RAG_TOP_K_PAPERS` 默认 4 → 6，本地文献成为主要证据源。

### 端到端验证（真实调用）
References 共 19 条统一编号：`[1-4]` 本地实验、`[5-10]` 本地文献 chunk、`[11-19]` 联网（全部缓存命中）；
5 条建议的 reasoning 同时引用了本地文献 `[7,8]`、本地实验 `[1,3,4]` 与联网 `[11-13]`。

### 附带
- `.gitignore` 新增 `Solid Sorbent Papers/`（版权文献不入库）。
- 前端/SSE/app.py **零改动**（RAG 输出结构与 Tavily 一致的设计再次兑现）。

---

## 11. 注意事项 / 后续可做

- **密钥安全**：`.env`、`data/rag_chroma/`、`data/papers/` 均被 gitignore 忽略，不会进 git。
  换机器需手动重建 `.env`，并重新 `python -m rag.ingest`。
- **合并 upstream**：若原始库也改了 `llm_service.py`，合并可能冲突（本地已大改）。按需取舍。
- **RAG 二期可扩展**（见 RAG_PLAN.md §11）：加 `bge-reranker` 精排、混合检索、按条件硬过滤、
  实验提交后自动重索引、前端 `[N]` 点击回链 DOI。
- **依赖完整性**：`requirements.txt` 已补全，新环境 `pip install -r requirements.txt` 可直接装齐。
