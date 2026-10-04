# RAGX

RAGX 是本仓库的 Retrieval-Augmented Generation (RAG) 示例项目，默认使用 `bi_encoder` 候选搜索：向量召回负责语义匹配，Best Matching 25 (BM25) 负责精确词面匹配，Reciprocal Rank Fusion (RRF) 负责融合候选。融合后可通过配置选择 Cross-Encoder、ColBERT 或关闭二阶段重排。

## 快速运行

```bash
cd agent-library/ragx
python3 ragx_cli.py index --source data/resources --reset
python3 ragx_cli.py query "RAGX 如何构建索引？" --top-k 3
```

默认配置：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_SEARCH_BACKEND` | `bi_encoder` | 默认候选搜索配置。 |
| `RAG_RERANKER_BACKEND` | `cross_encoder` | 默认二阶段重排策略，保留原 cross-encoder 入口。 |
| `RAG_VECTOR_BACKEND` | `sqlite` | 默认本地向量库。 |
| `RAG_TOP_K` | `4` | 默认返回证据数量。 |

## 切换重排策略

Cross-Encoder：

```bash
cd agent-library/ragx
RAG_RERANKER_BACKEND=cross_encoder \
python3 ragx_cli.py query "报销材料几天内提交？" --top-k 3
```

ColBERT 本地验证：

```bash
cd agent-library/ragx
RAG_RERANKER_BACKEND=colbert \
python3 ragx_cli.py query "报销材料几天内提交？" --top-k 3
```

ColBERT 加载 HuggingFace 模型：

```bash
cd agent-library/ragx
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

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/reranking-strategies.md`](docs/reranking-strategies.md) | Cross-Encoder 与 ColBERT 的使用方法、配置项、性能对比和取舍。 |
| [`docs/multi-recall.md`](docs/multi-recall.md) | 向量召回、BM25 召回、RRF 融合与重排链路。 |
| [`docs/indexing-and-query-flow.md`](docs/indexing-and-query-flow.md) | 知识库创建、增量同步、查询和生成流程。 |
| [`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md) | 检索评估指标与入口。 |

## 验证

```bash
cd agent-library/ragx
python3 -m ruff check .
python3 -m pytest -q --cov=. --cov-report=term-missing
```
