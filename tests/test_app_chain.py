"""完整生产链路验证：RagApplication 建库(含增量) + metadata 过滤问答。"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from rag.app import (
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_CANDIDATE_K,
    RagApplication,
)
from rag.ingestion.layout import LayoutAwareChunker
from rag.interfaces import (
    ChatMessage,
    Document,
    ScoredDocument,
)


def _write(d: str, name: str, text: str) -> None:
    with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def test_full_chain_with_incremental():
    with tempfile.TemporaryDirectory() as d:
        store_dir = os.path.join(d, "stores")
        os.makedirs(store_dir)
        _write(d, "rule.md", "# 制度\n\n## 报销\n出差报销需在 7 天内提交相关凭证，"
                             "逾期未提交的将无法报销，请务必在规定时间内完成提交流程。\n")
        app = RagApplication(
            Settings(
                provider="mock",
                vector_backend="sqlite",
                sqlite_path=os.path.join(store_dir, "vectors.db"),
                manifest_path="",
            )
        )
        assert isinstance(app._chunker, LayoutAwareChunker)

        r1 = app.index(d)
        assert r1.added == 1, r1

        # 再次 index 未改 → 跳过(规避重复写入/孤儿)
        r2 = app.index(d)
        assert r2.skipped == 1, r2

        # 修改文档 → delete-then-insert，版本递增
        _write(d, "rule.md", "# 制度\n\n## 报销\n出差报销需在 3 天内提交相关凭证，"
                             "逾期未提交的将无法报销，请务必在规定时间内完成提交流程。\n")
        r3 = app.index(d)
        assert r3.updated == 1 and r3.deleted > 0, r3

        # 问答：默认 is_latest 过滤，只命中最新版本
        res = app.ask("报销要几天内提交？")
        assert res["answer"], res
        assert res["contexts"], res
        # 命中的是最新版本 v2
        assert all(c["version"] == 2 for c in res["contexts"]), res
        # 溯源字段齐全
        assert "heading_path" in res["contexts"][0]
        print(f"✓ full chain: add={r1.added} skip={r2.skipped} upd={r3.updated} "
              f"del={r3.deleted}; answer hits v{res['contexts'][0]['version']}")


def test_hybrid_retrieval_uses_embedding_for_responses_endpoint(monkeypatch):
    from rag import app as app_module

    embed_calls = []

    class FakeLLM:
        def chat(self, messages, **kwargs):
            assert any(isinstance(message, ChatMessage) for message in messages)
            return "真实 LLM 占位响应"

    class FakeEmbedding:
        def embed(self, texts):
            embed_calls.append(list(texts))
            return [[float(len(text) or 1), 1.0, 0.0] for text in texts]

    monkeypatch.setattr(app_module, "build_llm_provider", lambda cfg: FakeLLM())
    monkeypatch.setattr(app_module, "build_embedding_provider", lambda cfg: FakeEmbedding())

    with tempfile.TemporaryDirectory() as d:
        store_dir = os.path.join(d, "stores")
        os.makedirs(store_dir)
        _write(
            d,
            "rule.md",
            "# 制度\n\n## 报销\n报销材料需要在 3 天内提交，"
            "申请人必须上传发票、审批单和付款证明，逾期提交会影响费用审核，"
            "财务部门会根据最新制度完成校验并反馈处理结果。\n",
        )
        app = RagApplication(
            Settings(
                provider="openai",
                api_key="fake-key",
                endpoint="https://example.test/v1/responses",
                llm_model="gpt",
                retrieval_backend="hybrid",
                vector_backend="sqlite",
                sqlite_path=os.path.join(store_dir, "vectors.db"),
                manifest_path="",
            )
        )

        report = app.index(d)
        result = app.ask("报销材料几天内提交？")

    assert report.added == 1
    assert app._retrieval_backend == "hybrid"
    assert embed_calls
    assert result["answer"] == "真实 LLM 占位响应"
    assert result["contexts"]
    assert result["contexts"][0]["source"] == "rule.md"
    print("✓ hybrid retrieval: responses endpoint still uses embedding provider")


def test_app_uses_relevance_evaluation_before_returning_contexts():
    class StaticRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=4, metadata_filter=None):
            self.calls.append((query, top_k, metadata_filter))
            return [
                ScoredDocument(
                    document=Document(
                        id="deploy.md::0",
                        content="部署日志包含服务启动和端口检查。",
                        metadata={
                            "source": "deploy.md",
                            "doc_id": "deploy.md",
                            "is_latest": True,
                        },
                    ),
                    score=0.99,
                ),
                ScoredDocument(
                    document=Document(
                        id="rule.md::0",
                        content=(
                            "报销材料需要在 3 天内提交，申请人必须上传发票和审批单。"
                            "完整片段结束TAIL"
                        ),
                        metadata={
                            "source": "rule.md",
                            "doc_id": "rule.md",
                            "heading_path": "制度 > 报销材料",
                            "is_latest": True,
                            "version": 1,
                        },
                    ),
                    score=0.01,
                ),
            ]

    app = RagApplication(
        Settings(
            provider="mock",
            vector_backend="sqlite",
            sqlite_path=":memory:",
            manifest_path="",
            top_k=1,
        )
    )
    retriever = StaticRetriever()
    app._retriever = retriever

    result = app.ask("报销材料几天内提交？")

    assert retriever.calls[0][1] == DEFAULT_RETRIEVAL_CANDIDATE_K
    assert result["contexts"][0]["source"] == "rule.md"
    assert result["contexts"][0]["score"] > 0.5
    assert "完整片段结束TAIL" in result["contexts"][0]["preview"]
    print("✓ app: cross encoder reranks and keeps full preview")


def test_app_limits_reranked_contexts_to_twenty():
    class ManyHitsRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=4, metadata_filter=None):
            self.calls.append((query, top_k, metadata_filter))
            return [
                ScoredDocument(
                    document=Document(
                        id=f"rule.md::{index}",
                        content=f"报销材料需要在 3 天内提交，第 {index} 个候选。",
                        metadata={
                            "source": "rule.md",
                            "doc_id": "rule.md",
                            "heading_path": "制度 > 报销材料",
                            "is_latest": True,
                            "version": 1,
                        },
                    ),
                    score=float(100 - index),
                )
                for index in range(25)
            ]

    app = RagApplication(
        Settings(
            provider="mock",
            vector_backend="sqlite",
            sqlite_path=":memory:",
            manifest_path="",
        )
    )
    retriever = ManyHitsRetriever()
    app._retriever = retriever

    result = app.ask("报销材料几天内提交？")

    assert retriever.calls[0][1] == DEFAULT_RETRIEVAL_CANDIDATE_K
    assert len(result["contexts"]) == DEFAULT_RERANK_TOP_K
    print("✓ app: reranked contexts capped at 20")


def test_app_evaluate_retrieval_returns_effect_metrics():
    with tempfile.TemporaryDirectory() as d:
        store_dir = os.path.join(d, "stores")
        os.makedirs(store_dir)
        _write(
            d,
            "rule.md",
            "# 制度\n\n## 报销\n报销材料需要在 3 天内提交，"
            "申请人必须上传发票和审批单。"
            "财务部门会根据最新制度完成校验并反馈处理结果，"
            "请务必在规定时间内完成提交流程，避免影响费用审核。\n",
        )
        app = RagApplication(
            Settings(
                provider="mock",
                vector_backend="sqlite",
                sqlite_path=os.path.join(store_dir, "vectors.db"),
                manifest_path="",
            )
        )
        app.index(d)

        report = app.evaluate_retrieval(
            "报销材料几天内提交？",
            relevant_sources=["rule.md"],
            top_k_values=(1, 3),
        )
        empty_k_report = app.evaluate_retrieval(
            "报销材料几天内提交？",
            relevant_sources=["rule.md"],
            top_k_values=(),
        )

    assert report["labelled"] is True
    assert report["hit_at_k"][1] is True
    assert report["mrr"] == 1.0
    assert report["source_diversity"] == 1.0
    assert empty_k_report["hit_at_k"] == {}
    print("✓ app: retrieval effect metrics returned")


if __name__ == "__main__":
    test_full_chain_with_incremental()
    test_app_uses_relevance_evaluation_before_returning_contexts()
    test_app_limits_reranked_contexts_to_twenty()
    test_app_evaluate_retrieval_returns_effect_metrics()
    print("\n✓✓✓ full production chain passed")
