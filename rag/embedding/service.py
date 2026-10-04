"""嵌入节点 —— 把 Document 文本转向量。

职责边界：只做「文本→向量」的编排，真正的算力委托给 EmbeddingProvider。
与具体厂商解耦：构造时注入 EmbeddingProvider 抽象。
依赖：rag.interfaces。
"""
from __future__ import annotations

from collections.abc import Sequence

from rag.interfaces import Document, EmbeddingProvider


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        return self._provider.embed([d.content for d in documents])

    def embed_query(self, query: str) -> list[float]:
        return self._provider.embed([query])[0]
