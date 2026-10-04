"""混合检索验证：向量 + 词法召回、RRF 融合、去重和降级路径。"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from rag.interfaces import Document, ScoredDocument
from rag.retrieval.fusion import (
    FusionConfig,
    reciprocal_rank_fusion,
)
from rag.retrieval.hybrid import HybridRetriever


def _hit(
    chunk_id: str,
    source: str,
    score: float,
    content: str = "content",
) -> ScoredDocument:
    """构造检索命中。

    输入 (Input):
        chunk_id: chunk ID。
        source: 来源文档。
        score: 原始检索分。
        content: 文档内容。

    输出 (Output):
        ScoredDocument 实例。

    示例 (Example):
        _hit("a::0", "a.md", 1.0)
    """
    return ScoredDocument(
        document=Document(
            id=chunk_id,
            content=content,
            metadata={"source": source, "doc_id": source, "is_latest": True},
        ),
        score=score,
    )


class _StaticRetriever:
    """固定结果 Retriever，用于验证 HybridRetriever 编排。"""

    def __init__(self, hits: list[ScoredDocument]) -> None:
        """初始化固定结果 Retriever。

        输入 (Input):
            hits: 待返回的命中列表。

        输出 (Output):
            None。

        示例 (Example):
            _StaticRetriever([_hit("a::0", "a.md", 1.0)])
        """
        self.hits = hits
        self.calls: list[tuple[str, int, dict | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict | None = None,
    ) -> list[ScoredDocument]:
        """返回固定命中，并记录调用参数。

        输入 (Input):
            query: 用户查询。
            top_k: 返回数量。
            metadata_filter: metadata 过滤条件。

        输出 (Output):
            固定命中列表的前 top_k 条。

        示例 (Example):
            retriever.retrieve("q", top_k=2)
        """
        self.calls.append((query, top_k, metadata_filter))
        return self.hits[:top_k]


def test_rrf_promotes_cross_channel_hit():
    vector_hits = [_hit("semantic::0", "semantic.md", 0.99), _hit("shared::0", "shared.md", 0.8)]
    lexical_hits = [_hit("shared::0", "shared.md", 12.0), _hit("exact::0", "exact.md", 10.0)]

    fused = reciprocal_rank_fusion(
        [vector_hits, lexical_hits],
        weights=[1.0, 1.0],
        top_k=3,
    )

    assert fused[0].document.id == "shared::0"
    assert [hit.document.id for hit in fused].count("shared::0") == 1
    print("✓ hybrid rrf: cross-channel hit promoted and deduped")


def test_rrf_default_k_is_60():
    assert FusionConfig().rrf_k == 60
    print("✓ hybrid rrf: default K is 60")


def test_hybrid_retriever_forwards_filter_and_expands_candidate_k():
    vector = _StaticRetriever([_hit("vector::0", "vector.md", 0.9)])
    lexical = _StaticRetriever([_hit("lexical::0", "lexical.md", 8.0)])
    retriever = HybridRetriever(
        vector,
        lexical,
        config=FusionConfig(candidate_k=20, source_diversity_penalty=0.0),
    )

    hits = retriever.retrieve("报销材料", top_k=3, metadata_filter={"is_latest": True})

    assert hits
    assert vector.calls == [("报销材料", 20, {"is_latest": True})]
    assert lexical.calls == [("报销材料", 20, {"is_latest": True})]
    print("✓ hybrid retriever: filter forwarded and candidate_k expanded")


def test_hybrid_retriever_does_not_over_expand_large_candidate_request():
    vector = _StaticRetriever([_hit("vector::0", "vector.md", 0.9)])
    lexical = _StaticRetriever([_hit("lexical::0", "lexical.md", 8.0)])
    retriever = HybridRetriever(
        vector,
        lexical,
        config=FusionConfig(candidate_k=20, source_diversity_penalty=0.0),
    )

    retriever.retrieve("报销材料", top_k=100, metadata_filter={"is_latest": True})

    assert vector.calls == [("报销材料", 100, {"is_latest": True})]
    assert lexical.calls == [("报销材料", 100, {"is_latest": True})]
    print("✓ hybrid retriever: large candidate request is not multiplied")


def test_rrf_applies_source_diversity():
    same_source_a = _hit("a::0", "same.md", 10.0)
    same_source_b = _hit("a::1", "same.md", 9.5)
    other_source = _hit("b::0", "other.md", 9.0)

    fused = reciprocal_rank_fusion(
        [[same_source_a, same_source_b, other_source]],
        weights=[1.0],
        top_k=2,
        source_diversity_penalty=1.0,
    )

    assert [hit.document.metadata["source"] for hit in fused] == ["same.md", "other.md"]
    print("✓ hybrid rrf: source diversity penalty ok")


def test_app_hybrid_backend_indexes_and_retrieves(tmp_path):
    from rag.app import RagApplication

    text = "# AI Agent 记忆系统\n\n" + (
        "AI Agent 的记忆系统包括短期记忆、长期记忆和检索增强，"
        "用于保存用户偏好、任务上下文和历史事实。"
        * 8
    )
    source = tmp_path / "memory.md"
    source.write_text(text, encoding="utf-8")
    store_dir = tmp_path / "stores"
    store_dir.mkdir()

    app = RagApplication(
        Settings(
            provider="mock",
            retrieval_backend="hybrid",
            vector_backend="sqlite",
            sqlite_path=str(store_dir / "vectors.db"),
            manifest_path="",
            top_k=3,
        )
    )
    report = app.index(str(tmp_path))
    result = app.ask("AI Agent 记忆系统是什么？")

    assert app._retrieval_backend == "hybrid"
    assert app._search_backend == "bi_encoder"
    assert app._reranker_backend == "cross_encoder"
    assert report.added == 1
    assert result["contexts"]
    assert result["contexts"][0]["source"] == "memory.md"
    print("✓ app hybrid: explicit backend indexes and retrieves")


def test_hybrid_surfaces_embedding_provider_errors(monkeypatch):
    import rag.app as app_module

    class FakeLLM:
        def chat(self, messages, **kwargs):
            """返回固定回答。

            输入 (Input):
                messages: prompt 消息。
                kwargs: 生成参数。

            输出 (Output):
                固定回答文本。

            示例 (Example):
                FakeLLM().chat([])
            """
            return "fake answer"

    def fail_embedding_provider(cfg):
        """模拟 embedding provider 配置或连通性失败。

        输入 (Input):
            cfg: 应用配置。

        输出 (Output):
            抛出 AssertionError。

        示例 (Example):
            fail_embedding_provider(Settings())
        """
        raise AssertionError("embedding provider is required for hybrid retrieval")

    monkeypatch.setattr(app_module, "build_llm_provider", lambda cfg: FakeLLM())
    monkeypatch.setattr(app_module, "build_embedding_provider", fail_embedding_provider)

    with pytest.raises(AssertionError, match="embedding provider is required"):
        app_module.RagApplication(
            Settings(
                provider="openai",
                api_key="fake",
                endpoint="https://example.com/responses",
                retrieval_backend="hybrid",
                vector_backend="sqlite",
                sqlite_path=":memory:",
                manifest_path="",
            )
        )
    print("✓ app hybrid: embedding provider errors are surfaced")


def test_hybrid_functional_retrieval_on_sample_resources(tmp_path):
    """端到端验证混合检索：语义类 query 命中向量通道，字面类 query 命中词法通道。

    夹具在 tmp_path 内自建，不依赖 data/resources 下的外部样本文件——
    否则仓库裁剪样本时测试会静默失效。
    """
    from rag.app import RagApplication

    source_dir = tmp_path / "resources"
    source_dir.mkdir()
    # 每段正文需超过 ChunkPolicy.min_chunk(=40 token)，否则会被去噪丢弃。
    (source_dir / "governance.md").write_text(
        "# CPO 体验治理\n\n"
        "## 目标\n\n"
        "CPO 体验治理的目标是降低商家接入成本，统一工单流转链路，"
        "让商家在接入、履约、售后各环节都有一致的入口与状态反馈机制。\n\n"
        "## 商家反馈\n\n"
        "商家反馈通过治理看板提交，按日汇总给 CPO 团队，"
        "高优问题会进入当周迭代评审，并在下一个发布窗口内反馈处理结论。\n",
        encoding="utf-8",
    )
    (source_dir / "retrieval.md").write_text(
        "# 混合检索\n\n"
        "## 融合策略\n\n"
        "混合检索通过向量召回与 BM25 词法召回并行执行，"
        "再用 Reciprocal Rank Fusion 按各通道内部排名融合候选，避免跨通道比较原始分数。\n\n"
        "## 分块\n\n"
        "表格按行切分时必须把表头复制到每个分片，"
        "否则分片脱离列定义后无法被正确理解，列名也无法成为可检索的语义锚点。\n\n"
        "## PDF 版面\n\n"
        "PDF 双栏论文需要先做投影分栏再排序，"
        "否则左右栏的文本会逐行交错，导致标题路径计算与语义 embedding 全部失真。\n",
        encoding="utf-8",
    )

    app = RagApplication(
        Settings(
            provider="mock",
            retrieval_backend="hybrid",
            vector_backend="sqlite",
            sqlite_path=":memory:",
            manifest_path="",
            top_k=5,
            chunk_size=300,
            chunk_overlap=50,
        )
    )
    app.index(str(source_dir))
    #夹具必须真正产出 chunk，否则后续断言无意义
    assert app._store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0, (
        "夹具未产出任何 chunk：请检查测试文档是否低于 ChunkPolicy.min_chunk 被去噪"
    )

    checks = {
        "CPO 体验治理的目标是什么？": "governance.md",
        "混合检索怎么做？": "retrieval.md",
        "表格如何切分？": "retrieval.md",
        "PDF 双栏论文怎么处理？": "retrieval.md",
        "商家反馈如何获得？": "governance.md",
    }
    for query, expected_source in checks.items():
        result = app.ask(query)
        sources = [context["source"] for context in result["contexts"]]
        assert sources, f"query={query!r} 未召回任何证据"
        assert expected_source in sources, (
            f"query={query!r} 期望命中 {expected_source}，实际 {sources}"
        )
    print("✓ app hybrid: sample resources hit@k checks ok")
