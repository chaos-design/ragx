"""Manifest and parent-child retrieval edge tests."""
from __future__ import annotations

from rag.ingestion.manifest import JsonFileManifest
from rag.ingestion.parent_child import ParentExpandingRetriever, ParentStore
from rag.interfaces import Document, ScoredDocument


class _StaticRetriever:
    """Retriever test double for parent expansion tests."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, top_k=4, metadata_filter=None):
        self.calls.append((query, top_k, metadata_filter))
        return self.hits[:top_k]


def _scored(chunk_id: str, score: float, **metadata) -> ScoredDocument:
    """Build a scored child document with optional parent metadata."""
    return ScoredDocument(
        document=Document(id=chunk_id, content=f"content:{chunk_id}", metadata=metadata),
        score=score,
    )


def test_json_file_manifest_handles_missing_invalid_and_save(tmp_path):
    """目标：JSON manifest 缺失/损坏时安全回空，保存时自动建目录并可重载。"""
    path = tmp_path / "state" / "manifest.json"
    manifest = JsonFileManifest(str(path))

    assert manifest.load() == {}

    path.parent.mkdir()
    path.write_text("{broken json", encoding="utf-8")
    assert manifest.load() == {}

    expected = {"doc.md": {"source_hash": "h2", "version": 2}}
    manifest.save(expected)

    assert JsonFileManifest(str(path)).load() == expected


def test_parent_store_delete_by_doc_removes_only_matching_parents():
    """目标：ParentStore 按 doc_id 级联删除时不影响其他文档的父块。"""
    store = ParentStore()
    store.put(Document("p1", "父块 1", {"doc_id": "a.md"}))
    store.put(Document("p2", "父块 2", {"doc_id": "a.md"}))
    store.put(Document("p3", "父块 3", {"doc_id": "b.md"}))

    assert store.delete_by_doc("a.md") == 2
    assert store.get("p1") is None
    assert store.get("p2") is None
    assert store.get("p3").content == "父块 3"
    assert store.delete_by_doc("missing.md") == 0


def test_parent_expanding_retriever_keeps_child_when_parent_id_missing():
    """目标：命中缺少 parent_id 的子块时应降级返回原始子块。"""
    child = _scored("orphan::0", 0.9, doc_id="doc.md")
    base = _StaticRetriever([child])
    retriever = ParentExpandingRetriever(base, ParentStore(), child_top_k=8)

    hits = retriever.retrieve("报销", top_k=1, metadata_filter={"is_latest": True})

    assert hits == [child]
    assert base.calls == [("报销", 8, {"is_latest": True})]


def test_parent_expanding_retriever_dedupes_by_best_parent_score():
    """目标：同一父块多次命中时保留最高分，并替换为父块内容。"""
    parent_store = ParentStore()
    parent_store.put(Document("parent-1", "完整父块内容", {"doc_id": "doc.md"}))
    hits = [
        _scored("child-low", 0.2, parent_id="parent-1", doc_id="doc.md"),
        _scored("child-high", 0.9, parent_id="parent-1", doc_id="doc.md"),
        _scored("missing-parent", 0.8, parent_id="parent-missing", doc_id="doc.md"),
    ]
    retriever = ParentExpandingRetriever(_StaticRetriever(hits), parent_store, child_top_k=3)

    expanded = retriever.retrieve("报销", top_k=2)

    assert [hit.document.id for hit in expanded] == ["parent-1", "missing-parent"]
    assert expanded[0].score == 0.9
    assert expanded[0].document.content == "完整父块内容"
    assert expanded[1].score == 0.8
