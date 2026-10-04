"""检索节点 —— 编排 嵌入服务 + 向量库 完成「query→相关文档」。

职责边界：把 query 向量化后交给向量库检索；自身不持有模型、不存数据。
依赖：rag.interfaces + embedding.service + vectorstore（均通过抽象/注入）。
"""
from __future__ import annotations

from rag.embedding.service import EmbeddingService
from rag.interfaces import Retriever, ScoredDocument, VectorStore


class VectorRetriever(Retriever):
    def __init__(self, embedding_service: EmbeddingService, store: VectorStore) -> None:
        self._embedding = embedding_service
        self._store = store

    def retrieve(self, query: str, top_k: int = 4, metadata_filter: dict | None = None) -> list[ScoredDocument]:
        query_vec = self._embedding.embed_query(query)
        return self._store.search(query_vec, top_k=top_k, metadata_filter=metadata_filter)
