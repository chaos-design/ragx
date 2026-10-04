"""Vector store factory and in-memory vector store edge tests."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from rag.interfaces import Document
from rag.vectorstore.factory import build_vector_store
from rag.vectorstore.memory_store import InMemoryVectorStore
from rag.vectorstore.sqlite_store import SQLiteVectorStore


def _document(chunk_id: str, doc_id: str, content: str, **metadata) -> Document:
    """Build a document with stable vector-store metadata."""
    return Document(
        id=chunk_id,
        content=content,
        metadata={
            "doc_id": doc_id,
            "source": doc_id,
            "content_hash": f"hash-{chunk_id}",
            **metadata,
        },
    )


def test_in_memory_vector_store_search_orders_and_filters():
    """目标：内存向量库应按相似度排序，并在检索前应用 metadata 过滤。"""
    store = InMemoryVectorStore()
    documents = [
        _document("a::0", "a.md", "报销规则", acl="public", is_latest=True),
        _document("b::0", "b.md", "差旅制度", acl="private", is_latest=True),
        _document("c::0", "c.md", "旧版制度", acl="public", is_latest=False),
    ]
    store.add(documents, [[1.0, 0.0], [0.4, 0.9], [0.9, 0.1]])

    hits = store.search(
        [1.0, 0.0],
        top_k=5,
        metadata_filter={"acl": ["public"], "is_latest": True},
    )

    assert [hit.document.id for hit in hits] == ["a::0"]
    assert hits[0].score == pytest.approx(1.0)
    assert store.list_hashes("a.md") == {"a::0": "hash-a::0"}


def test_in_memory_vector_store_delete_by_doc_cleans_vectors():
    """目标：按 doc_id 删除应同步清理文档、向量和 hash 视图。"""
    store = InMemoryVectorStore()
    documents = [
        _document("a::0", "a.md", "第一块"),
        _document("a::1", "a.md", "第二块"),
        _document("b::0", "b.md", "保留块"),
    ]
    store.add(documents, [[1.0], [0.8], [0.1]])

    assert store.delete_by_doc("a.md") == 2
    assert store.delete_by_doc("missing.md") == 0
    assert store.list_hashes("a.md") == {}
    assert [hit.document.id for hit in store.search([1.0], top_k=5)] == ["b::0"]


def test_in_memory_vector_store_top_k_zero_returns_empty():
    """目标：top_k=0 时不返回候选，空向量也不应抛错。"""
    store = InMemoryVectorStore()
    store.add([_document("zero::0", "zero.md", "空向量")], [[0.0, 0.0]])

    assert store.search([0.0, 0.0], top_k=0) == []
    hits = store.search([0.0, 0.0], top_k=1)
    assert hits[0].score == 0.0


def test_build_vector_store_defaults_to_sqlite_for_unknown_backend(tmp_path):
    """目标：除显式 pgvector 外，工厂按生产默认策略回落 SQLite。"""
    cfg = SimpleNamespace(
        vector_backend="unknown",
        sqlite_path=str(tmp_path / "vectors.db"),
        pg_dsn="postgres://unused",
        embedding_dim=4,
    )

    store = build_vector_store(cfg)

    try:
        assert isinstance(store, SQLiteVectorStore)
    finally:
        store.close()


def test_build_vector_store_uses_pgvector_when_explicit(monkeypatch):
    """目标：显式配置 pgvector 时延迟导入 PgVectorStore 并传入 DSN 与维度。"""

    class FakePgVectorStore:
        def __init__(self, dsn, dim):
            self.dsn = dsn
            self.dim = dim

    fake_module = types.ModuleType("rag.vectorstore.pgvector_store")
    fake_module.PgVectorStore = FakePgVectorStore
    monkeypatch.setitem(sys.modules, "rag.vectorstore.pgvector_store", fake_module)
    cfg = SimpleNamespace(
        vector_backend=" PGVECTOR ",
        sqlite_path=":memory:",
        pg_dsn="postgres://example/db",
        embedding_dim=1536,
    )

    store = build_vector_store(cfg)

    assert isinstance(store, FakePgVectorStore)
    assert store.dsn == "postgres://example/db"
    assert store.dim == 1536
