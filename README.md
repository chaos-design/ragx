# RAGX

RAGX 是一个可离线运行的 Retrieval-Augmented Generation (RAG) 参考实现，覆盖
**版面分析 → 结构感知分块 → 增量同步 → 多路召回 → 排名融合 → 二阶段重排 → 提示词组装 → 生成**
的完整链路。

默认用 `bi_encoder` 做候选搜索：向量召回负责语义匹配，BM25 负责精确词面匹配，
Reciprocal Rank Fusion (RRF) 负责融合候选。融合后可配置选择 Cross-Encoder、
ColBERT 或关闭重排。

## 环境准备

需要 Python 3.9+（本仓库在 3.14 上开发与测试）。源码未使用 3.10+ 专属语法：
`X | Y` 联合类型全部位于 `from __future__ import annotations` 覆盖的注解位置，
因此实际语法下限由 PEP 585 内置泛型（`list[str]`）决定，即 3.9。

```bash
git clone git@github.com:chaos-design/ragx.git
cd ragx

# 最小可运行集：真实 provider + PDF/OCR + CLI
python3 -m pip install -r requirements.txt
```

`requirements.txt` 里 ColBERT（`torch`/`transformers`）与测试工具默认注释掉，
只有需要加载 HuggingFace 模型或跑测试时才安装。

## 快速运行

先配置凭据（provider 走真实 OpenAI 兼容接口）：

```bash
export OPENAI_API_KEY=sk-...
```

然后建库并检索：

```bash
python3 ragx_cli.py index --source data/resources --reset
python3 ragx_cli.py query "RAGX 如何构建索引？" --top-k 3
```

### 无凭据时如何验证链路

不配 `OPENAI_API_KEY` 也可以跑通全链路，但 embedding 与生成都是**本地确定性桩**，
产物不可用于生产。需要同时设置两个开关：

```bash
export RAG_PROVIDER=mock
export RAG_ALLOW_MOCK_PROVIDER=1
python3 ragx_cli.py index --source data/resources --reset
python3 ragx_cli.py query "RAGX 如何构建索引？" --top-k 3
```

双开关是刻意设计：`RAG_PROVIDER=mock` 单独设置会被拒绝，避免误把测试配置带上线。
走 mock 时 CLI 每次都会在 stderr 打印警告，因为「检索真实执行 + 生成是桩」最容易
让人误判系统可用。

## 默认配置

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_SEARCH_BACKEND` | `bi_encoder` | 默认候选搜索配置。 |
| `RAG_RERANKER_BACKEND` | `cross_encoder` | 默认二阶段重排策略。 |
| `RAG_VECTOR_BACKEND` | `sqlite` | 默认本地向量库。 |
| `RAG_TOP_K` | `4` | 默认返回证据数量。 |

## 切换重排策略

Cross-Encoder：

```bash
RAG_RERANKER_BACKEND=cross_encoder \
python3 ragx_cli.py query "报销材料几天内提交？" --top-k 3
```

ColBERT 本地验证：

```bash
RAG_RERANKER_BACKEND=colbert \
python3 ragx_cli.py query "报销材料几天内提交？" --top-k 3
```

ColBERT 加载 HuggingFace 模型：

```bash
python3 -m pip install torch transformers
RAG_RERANKER_BACKEND=colbert \
RAG_COLBERT_MODEL=colbert-ir/colbertv2.0 \
python3 ragx_cli.py query "报销材料几天内提交？" --top-k 3
```

CLI 也支持直接覆盖：

```bash
python3 ragx_cli.py query "RAGX 如何检索？" \
  --reranker-backend colbert \
  --top-k 3
```

## Provider 依赖说明

Provider 抽象优先复用共享 `agent_provider` 包（支持 Azure、Responses 协议等完整能力）。
该包**不是本仓库的必需依赖**：缺失时自动降级到 `rag/providers/_fallback.py` 中的
本地最小实现，保证本仓库可独立导入、跑测试。

降级实现**不具备真实模型能力**，因此在需要真实 provider 时会直接抛错而不是返回桩实现，
错误信息里给出四条可执行出路。离线验证请显式使用上面的双开关。

## 模块入口

四个互相独立的入口，各自也可作为 CLI 子命令使用：

| 入口 | 作用 |
| --- | --- |
| `run_index` | 建库/增量同步，返回 SyncReport。 |
| `run_query` | 检索 + 生成，返回证据与答案。 |
| `run_evaluation` | 计算 Hit@K / MRR / Precision@K / Recall@K。 |
| `RagApplication` | 组合根，直接持有各模块实例。 |

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/interview-bank.html`](docs/interview-bank.html) | 面试题库（62 题 / 13 模块），单文件可直接浏览器打开，含学习路径。 |
| [`docs/reranking-strategies.md`](docs/reranking-strategies.md) | Cross-Encoder 与 ColBERT 的使用方法、配置项、性能对比和取舍。 |
| [`docs/multi-recall.md`](docs/multi-recall.md) | 向量召回、BM25 召回、RRF 融合与重排链路。 |
| [`docs/indexing-and-query-flow.md`](docs/indexing-and-query-flow.md) | 知识库创建、增量同步、查询和生成流程。 |
| [`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md) | 检索评估指标与入口。 |
| [`docs/module-entrypoints.md`](docs/module-entrypoints.md) | 模块入口与职责边界。 |
| [`docs/layout-to-generation-flow.md`](docs/layout-to-generation-flow.md) | 从版面解析到生成的字段流转。 |
| [`docs/hybird-search.md`](docs/hybird-search.md) | 混合检索设计说明。 |
| [`docs/abbreviations.md`](docs/abbreviations.md) | 术语缩写对照。 |

## 验证

```bash
python3 -m pip install pytest ruff
python3 -m ruff check .
python3 -m pytest -q
```

## 已知限制

以下是代码中真实存在的限制（每条都在当前代码里可复现），不是待办清单：

- `sqlite_store.py:106` 用 `meta.get("__chunk_id__", "")` 填 `Document.id`，但全仓从未写入
  `__chunk_id__`，因此 `Document.id` 恒为空串，基于 `relevant_chunk_ids` 的评估口径失效。
- `sqlite_store.py:31` 的 `_cosine` 用 `zip(a, b)` 累乘，维度不一致时静默截断而非报错。
- `run_query` 只构造 `RagApplication` 并调用 `ask()`，不触发词法索引同步；
  而 `RagApplication.__init__` 建的是空 `LexicalDocumentStore`，词法索引只在
  `run_index` 进程内重建。跨进程首次查询会静默退化为单路召回。
- 增量同步采用 delete-then-insert（`rag/ingestion/sync.py`，为规避孤儿向量）。
  代价是从删除到 embedding 完成之间存在检索空窗。
- `HybridSyncer.sync` 先同步向量库、再同步词法索引，两步之间无事务，
  中途失败会留下两侧不一致的状态。
- `HashingColBERTEncoder` 是无依赖的本地哈希实现，仅用于验证 MaxSim 流程本身，无语义能力。
- `build_vector_store` 对 `pgvector` 之外的任何取值都静默落到 SQLite，不校验也不抛错；
  而 `normalize_reranker_backend` 对未知取值显式抛错。两处风格不一致。
