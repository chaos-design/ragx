"""RagApplication filtering and evaluation edge tests."""
from __future__ import annotations

from rag.app import DEFAULT_RERANK_TOP_K, DEFAULT_RETRIEVAL_CANDIDATE_K, RagApplication
from rag.interfaces import Document, ScoredDocument


def _hit(chunk_id: str = "doc.md::0", score: float = 0.7) -> ScoredDocument:
    """Build a scored document suitable for app-level orchestration tests."""
    return ScoredDocument(
        document=Document(
            id=chunk_id,
            content="报销材料需要在七天内提交。",
            metadata={
                "source": "doc.md",
                "doc_id": "doc.md",
                "heading_path": ["制度", "报销"],
                "version": 2,
                "is_latest": True,
            },
        ),
        score=score,
    )


class _RecordingRetriever:
    """Retriever test double that records the exact metadata filter."""

    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=4, metadata_filter=None):
        self.calls.append((query, top_k, metadata_filter))
        return [_hit()]


class _RecordingReranker:
    """Reranker test double that records candidate count and top_k."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, contexts, top_k):
        self.calls.append((query, list(contexts), top_k))
        return list(contexts)[:top_k]


def test_ask_merges_latest_filter_with_user_metadata_filter():
    """目标：ask 默认添加 is_latest，并允许调用方叠加 ACL 等 metadata 过滤。"""
    app = object.__new__(RagApplication)
    retriever = _RecordingRetriever()
    reranker = _RecordingReranker()
    app._retriever = retriever
    app._reranker = reranker
    app._pipeline_ask_with_contexts = lambda query, contexts: {
        "answer": "ok",
        "context_count": len(contexts),
    }

    result = app.ask("报销材料", metadata_filter={"acl": "finance"})

    assert result == {"answer": "ok", "context_count": 1}
    assert retriever.calls == [
        (
            "报销材料",
            DEFAULT_RETRIEVAL_CANDIDATE_K,
            {"is_latest": True, "acl": "finance"},
        )
    ]
    assert reranker.calls[0][2] == DEFAULT_RERANK_TOP_K


def test_ask_uses_latest_filter_when_user_filter_is_empty():
    """目标：metadata_filter 为空时仍只检索最新版本。"""
    app = object.__new__(RagApplication)
    retriever = _RecordingRetriever()
    app._retriever = retriever
    app._reranker = _RecordingReranker()
    app._pipeline_ask_with_contexts = lambda query, contexts: {
        "answer": query,
        "contexts": contexts,
    }

    app.ask("报销材料")

    assert retriever.calls[0][2] == {"is_latest": True}


def test_evaluate_retrieval_merges_filter_and_forwards_relevance(monkeypatch):
    """目标：evaluate_retrieval 应复用过滤策略，并把相关性标注传给评估函数。"""
    app = object.__new__(RagApplication)
    retriever = _RecordingRetriever()
    reranker = _RecordingReranker()
    app._retriever = retriever
    app._reranker = reranker
    captured = {}

    class FakeReport:
        def to_dict(self):
            return {"hit_at_k": {2: True}, "captured": captured}

    def fake_evaluate(query, contexts, **kwargs):
        captured["query"] = query
        captured["contexts"] = list(contexts)
        captured.update(kwargs)
        return FakeReport()

    monkeypatch.setattr("rag.app.evaluate_retrieval_effect", fake_evaluate)

    result = app.evaluate_retrieval(
        "报销材料",
        metadata_filter={"acl": ["finance", "public"]},
        relevant_sources=["doc.md"],
        relevant_doc_ids=["doc.md"],
        relevant_chunk_ids=["doc.md::0"],
        top_k_values=(2, 4),
    )

    assert retriever.calls == [
        (
            "报销材料",
            DEFAULT_RETRIEVAL_CANDIDATE_K,
            {"is_latest": True, "acl": ["finance", "public"]},
        )
    ]
    assert reranker.calls[0][2] == DEFAULT_RETRIEVAL_CANDIDATE_K
    assert captured["relevant_sources"] == ["doc.md"]
    assert captured["relevant_doc_ids"] == ["doc.md"]
    assert captured["relevant_chunk_ids"] == ["doc.md::0"]
    assert captured["config"].top_k_values == (2, 4)
    assert result["hit_at_k"] == {2: True}
