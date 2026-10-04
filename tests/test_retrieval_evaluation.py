"""检索证据评估重排测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.interfaces import Document, ScoredDocument  # noqa: E402
from rag.retrieval.evaluation import (  # noqa: E402
    EvaluationConfig,
    evaluate_retrieval_effect,
)
from rag.retrieval.cross_encoder import (  # noqa: E402
    CrossEncoderReranker,
    CrossEncoderScorer,
    rank_by_relevance,
)


def _scored(source: str, content: str, raw_score: float, heading: str = ""):
    """构造测试命中。

    Example Input:
        _scored("rule.md", "报销材料需要 3 天内提交", 0.1)

    Example Output:
        ScoredDocument(document=Document(...), score=0.1)
    """
    return ScoredDocument(
        document=Document(
            id=f"{source}::0",
            content=content,
            metadata={
                "source": source,
                "doc_id": source,
                "heading_path": heading,
                "is_latest": True,
            },
        ),
        score=raw_score,
    )


def test_relevance_rank_promotes_important_low_raw_score_context():
    hits = [
        _scored(
            "deploy.md",
            "系统部署日志包含服务启动、端口检查和健康检查命令。",
            0.98,
            "部署手册",
        ),
        _scored(
            "rule.md",
            "报销材料需要在 3 天内提交，申请人必须上传发票和审批单。",
            0.05,
            "制度 > 报销材料",
        ),
    ]

    ranked = rank_by_relevance("报销材料几天内提交？", hits, top_k=2)

    assert ranked[0].document.metadata["source"] == "rule.md"
    assert ranked[0].score > ranked[1].score
    assert all(0.0 <= hit.score <= 1.0 for hit in ranked)
    print("✓ retrieval evaluation: important low raw score context promoted")


def test_relevance_rank_respects_top_k_and_empty_boundary():
    hits = [
        _scored("a.md", "报销材料需要在 3 天内提交。", 0.2),
        _scored("b.md", "报销材料需要上传发票。", 0.1),
    ]

    assert rank_by_relevance("报销材料", hits, top_k=0) == []
    ranked = rank_by_relevance("报销材料", hits, top_k=1)

    assert len(ranked) == 1
    assert ranked[0].document.metadata["source"] in {"a.md", "b.md"}
    print("✓ retrieval evaluation: top_k and empty boundaries ok")


def test_cross_encoder_reranker_accepts_injected_scorer():
    class PreferSecondScorer(CrossEncoderScorer):
        def score(self, query, documents):
            return [0.1, 0.9]

    hits = [
        _scored("a.md", "泛化内容", 0.99),
        _scored("b.md", "目标内容", 0.01),
    ]
    reranker = CrossEncoderReranker(scorer=PreferSecondScorer())

    ranked = reranker.rerank("目标", hits, top_k=2)

    assert ranked[0].document.metadata["source"] == "b.md"
    assert ranked[0].score > ranked[1].score
    print("✓ cross encoder reranker: injected scorer controls ranking")


def test_retrieval_effect_metrics_with_labels():
    hits = [
        _scored("deploy.md", "部署日志", 0.9),
        _scored("rule.md", "报销材料需要在 3 天内提交。", 0.8),
        _scored("faq.md", "发票 FAQ", 0.3),
    ]

    report = evaluate_retrieval_effect(
        "报销材料几天内提交？",
        hits,
        relevant_sources=["rule.md"],
        config=EvaluationConfig(top_k_values=(1, 2, 3)),
    )

    assert report.labelled is True
    assert report.best_rank == 2
    assert report.mrr == 0.5
    assert report.hit_at_k == {1: False, 2: True, 3: True}
    assert report.precision_at_k[2] == 0.5
    assert report.recall_at_k[2] == 1.0
    assert report.to_dict()["source_diversity"] == 1.0
    print("✓ retrieval evaluation: labelled hit/precision/recall/mrr ok")


def test_retrieval_effect_diagnostics_without_labels():
    hits = [
        _scored("a.md", "A", 0.2),
        _scored("a.md", "B", 0.4),
    ]

    report = evaluate_retrieval_effect("无标注 query", hits)

    assert report.labelled is False
    assert report.hit_at_k == {}
    assert report.mean_score == 0.3
    assert report.max_score == 0.4
    assert report.source_diversity == 0.5
    print("✓ retrieval evaluation: unlabelled diagnostics ok")
