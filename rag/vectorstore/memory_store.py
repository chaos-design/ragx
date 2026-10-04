"""向量存储节点 —— 内存版向量库 + 余弦检索 + metadata 过滤 + 按 doc 删除。

职责边界：只存向量与文档、做相似度计算与删除；不感知 embedding 如何产生。
生产可替换为 FAISS / Milvus / pgvector / Elasticsearch：
- 删除：Milvus 用 `delete(expr="doc_id in [...]")`；pgvector 用 SQL DELETE WHERE doc_id=...
- 过滤：检索前下推 metadata_filter(权限 acl / is_latest / 生效时间)，先过滤再算分。
依赖：rag.interfaces。
"""
from __future__ import annotations

import math
from typing import Sequence

from rag.interfaces import Document, ScoredDocument, VectorStore


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _match(meta: dict, flt: dict) -> bool:
    """简化版 metadata 过滤：支持等值匹配与 in 列表匹配。"""
    for k, v in flt.items():
        mv = meta.get(k)
        if isinstance(v, (list, tuple, set)):
            if mv not in v:
                return False
        elif mv != v:
            return False
    return True


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        # 以 chunk_id 为主键存储，便于按 doc 删除与差分
        self._docs: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    def add(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        for doc, vec in zip(documents, vectors):
            self._docs[doc.id] = doc
            self._vectors[doc.id] = list(vec)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[ScoredDocument]:
        scored = [
            ScoredDocument(document=doc, score=_cosine(query_vector, self._vectors[cid]))
            for cid, doc in self._docs.items()
            if not metadata_filter or _match(doc.metadata, metadata_filter)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def delete_by_doc(self, doc_id: str) -> int:
        targets = [cid for cid, d in self._docs.items() if d.metadata.get("doc_id") == doc_id]
        for cid in targets:
            self._docs.pop(cid, None)
            self._vectors.pop(cid, None)
        return len(targets)

    def list_hashes(self, doc_id: str) -> dict[str, str]:
        return {
            cid: d.metadata.get("content_hash", "")
            for cid, d in self._docs.items()
            if d.metadata.get("doc_id") == doc_id
        }
