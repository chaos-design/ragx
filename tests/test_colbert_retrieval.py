"""ColBERT late-interaction reranking tests."""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from rag.interfaces import Document, ScoredDocument
from rag.retrieval.colbert import (
    ColBERTEncoder,
    ColBERTRerankConfig,
    ColBERTReranker,
    HashingColBERTEncoder,
    TransformersColBERTEncoder,
    rank_by_colbert,
)
from rag.retrieval.reranker import (
    NoOpReranker,
    build_reranker,
    normalize_reranker_backend,
)


def _hit(
    chunk_id: str,
    content: str,
    score: float,
    *,
    source: str = "doc.md",
) -> ScoredDocument:
    """Build a scored document.

    Example Input:
        _hit("a::0", "content", 0.9)

    Example Output:
        ScoredDocument(...)
    """
    return ScoredDocument(
        document=Document(
            id=chunk_id,
            content=content,
            metadata={"source": source, "doc_id": source, "is_latest": True},
        ),
        score=score,
    )


class _StaticColBERTEncoder:
    """Deterministic encoder with explicit token vectors."""

    def encode_queries(self, queries: Sequence[str]) -> list[list[list[float]]]:
        """Return two orthogonal query token vectors.

        Example Input:
            encoder.encode_queries(["q"])

        Example Output:
            [[[1.0, 0.0], [0.0, 1.0]]]
        """
        return [[[1.0, 0.0], [0.0, 1.0]] for _ in queries]

    def encode_documents(self, documents: Sequence[Document]) -> list[list[list[float]]]:
        """Return token vectors by document id.

        Example Input:
            encoder.encode_documents([Document("exact", "", {})])

        Example Output:
            [[[1.0, 0.0], [0.0, 1.0]]]
        """
        vectors = {
            "exact": [[1.0, 0.0], [0.0, 1.0]],
            "partial": [[1.0, 0.0], [-1.0, 0.0]],
            "noise": [[-1.0, 0.0]],
        }
        return [vectors[document.id] for document in documents]


def test_hashing_colbert_encoder_returns_token_vectors():
    """验证本地 ColBERT encoder 输出 token-level vectors 并遵守 token 上限。"""
    encoder = HashingColBERTEncoder(
        query_max_tokens=2,
        document_max_tokens=3,
        dimension=8,
    )

    query_vectors = encoder.encode_queries(["RAGX 支持 ColBERT 检索"])[0]
    doc_vectors = encoder.encode_documents(
        [Document("doc", "ColBERT late interaction 检索", {"source": "a.md"})]
    )[0]

    assert 0 < len(query_vectors) <= 2
    assert 0 < len(doc_vectors) <= 3
    assert len(query_vectors[0]) == 8
    assert len(doc_vectors[0]) == 8
    print("✓ colbert: hashing encoder returns bounded token vectors")


def test_colbert_reranker_uses_maxsim_before_retrieval_score():
    """验证 ColBERT MaxSim 能压过原始召回分中的噪声候选。"""
    contexts = [
        _hit("noise", "unrelated", 0.99),
        _hit("partial", "partial match", 0.2),
        _hit("exact", "exact match", 0.1),
    ]
    reranker = ColBERTReranker(
        encoder=_StaticColBERTEncoder(),
        config=ColBERTRerankConfig(
            interaction_weight=0.95,
            retrieval_score_weight=0.05,
        ),
    )

    ranked = reranker.rerank("query", contexts, top_k=2)

    assert [hit.document.id for hit in ranked] == ["exact", "partial"]
    assert ranked[0].score > ranked[1].score
    print("✓ colbert: MaxSim interaction drives reranking")


def test_rank_by_colbert_compatibility_entrypoint():
    """验证 ColBERT 兼容入口不影响 cross-encoder 旧入口。"""
    contexts = [_hit("exact", "exact match", 0.1), _hit("noise", "noise", 0.9)]
    reranker = ColBERTReranker(encoder=_StaticColBERTEncoder())

    ranked = rank_by_colbert("query", contexts, top_k=1, reranker=reranker)

    assert ranked[0].document.id == "exact"
    print("✓ colbert: compatibility entrypoint ok")


def test_reranker_factory_selects_cross_encoder_colbert_and_none():
    """验证配置可独立切换不同 reranker backend。"""
    cross_encoder = build_reranker(Settings(reranker_backend="cross_encoder"))
    colbert = build_reranker(Settings(reranker_backend="colbert"))
    no_op = build_reranker(Settings(reranker_backend="none"))

    assert cross_encoder.__class__.__name__ == "CrossEncoderReranker"
    assert colbert.__class__.__name__ == "ColBERTReranker"
    assert isinstance(no_op, NoOpReranker)
    assert normalize_reranker_backend("cross-encoder") == "cross_encoder"
    assert normalize_reranker_backend("bi_encoder") == "none"
    with pytest.raises(ValueError, match="Unsupported reranker backend"):
        normalize_reranker_backend("bad")
    print("✓ reranker factory: selectable backends ok")


def test_transformers_colbert_encoder_reports_missing_optional_dependency(monkeypatch):
    """验证真实 ColBERT 模型加载缺少可选依赖时给出明确安装指引。"""
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        """Block torch import.

        Example Input:
            fake_import("torch")

        Example Output:
            raises ModuleNotFoundError
        """
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="torch transformers"):
        TransformersColBERTEncoder("colbert-ir/colbertv2.0")
    print("✓ colbert: optional dependency error is explicit")


def test_app_colbert_backend_indexes_and_retrieves(tmp_path):
    """功能验证：ColBERT 配置下可独立建库并输出检索结果。"""
    from rag.app import RagApplication

    source = tmp_path / "resources"
    source.mkdir()
    (source / "finance.md").write_text(
        "# 报销制度\n\n"
        + (
            "报销材料需要在七天内提交，审批通过后打款。"
            "申请人需要提供发票、审批单和费用说明。"
            * 6
        ),
        encoding="utf-8",
    )
    (source / "menu.md").write_text(
        "# 食堂菜单\n\n"
        + (
            "本周菜单包含面条、米饭和沙拉，午餐窗口会按时开放。"
            "菜品信息用于餐饮通知，不包含费用报销制度。"
            * 6
        ),
        encoding="utf-8",
    )

    app = RagApplication(
        Settings(
            provider="mock",
            retrieval_backend="hybrid",
            reranker_backend="colbert",
            vector_backend="sqlite",
            sqlite_path=":memory:",
            manifest_path="",
            top_k=2,
            chunk_size=80,
            chunk_overlap=10,
        )
    )
    app.index(str(source))
    result = app.ask("报销材料几天内提交？", top_k=2)

    assert app._search_backend == "bi_encoder"
    assert app._reranker_backend == "colbert"
    assert result["contexts"]
    assert result["contexts"][0]["source"] == "finance.md"
    print("✓ app colbert: functional retrieval ok")


def test_colbert_encoder_protocol_is_structural():
    """验证测试 encoder 满足 ColBERTEncoder 协议的结构形态。"""
    encoder: ColBERTEncoder = _StaticColBERTEncoder()

    assert encoder.encode_queries(["q"])
    assert encoder.encode_documents([Document("exact", "", {})])
    print("✓ colbert: encoder protocol is structural")
