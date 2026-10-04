"""词法检索排序验证：BM25、中文 n-gram、短语和 metadata 加权。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.interfaces import Document
from rag.retrieval.lexical import (
    LexicalDocumentStore,
    LexicalRetriever,
    _tokenize,
)


def _doc(
    doc_id: str,
    content: str,
    *,
    heading_path: str = "",
    is_latest: bool = True,
) -> Document:
    """构造测试 Document。

    输入 (Input):
        doc_id: 文档 ID。
        content: 分块正文。
        heading_path: 章节路径。
        is_latest: 是否为最新版本。

    输出 (Output):
        Document 实例。

    示例 (Example):
        _doc("rule.md", "报销规则")
    """
    return Document(
        id=f"{doc_id}::0",
        content=content,
        metadata={
            "doc_id": doc_id,
            "source": doc_id,
            "heading_path": heading_path,
            "is_latest": is_latest,
        },
    )


def test_tokenize_uses_cjk_ngrams_and_filters_stopwords():
    terms = _tokenize("CPO 体验治理的目标是什么？")

    assert "cpo" in terms
    assert "体验" in terms
    assert "治理" in terms
    assert "体验治" in terms
    assert "的" not in terms
    assert "什么" not in terms
    print("✓ lexical tokenize: cjk ngram + stopwords ok")


def test_bm25_prefers_domain_entity_over_common_overlap():
    store = LexicalDocumentStore()
    store.add(
        [
            _doc(
                "douyin-shop-cpo-experience-governance.md",
                "[抖店 CPO 体验治理解决方案]\n"
                "通过 CPO 工具定位高频问题，指引商家解决体验治理问题。",
                heading_path="抖店 CPO 体验治理解决方案",
            ),
            _doc(
                "ai-agent-engineer.docx",
                "目标是什么 体验是什么 事实是什么 目标是什么 "
                "知识库回答事实是什么，长期记忆回答状态是什么。",
                heading_path="ai-agent-engineer.docx",
            ),
        ]
    )

    hits = LexicalRetriever(store).retrieve(
        "CPO 体验治理的目标是什么？",
        top_k=2,
        metadata_filter={"is_latest": True},
    )

    assert hits[0].document.metadata["source"] == (
        "douyin-shop-cpo-experience-governance.md"
    )
    assert hits[0].score > hits[1].score
    print("✓ lexical bm25: domain entity outranks common overlap")


def test_bm25_prefers_explanatory_heading_over_command_snippet():
    store = LexicalDocumentStore()
    store.add(
        [
            _doc(
                "rag-sample.md",
                "[文档 RAG 处理实战样例 > 四、检索阶段]\n"
                "混合检索怎么做？同时跑向量召回和 BM25 关键词召回两路，"
                "再用 RRF 融合。",
                heading_path="文档 RAG 处理实战样例 > 四、检索阶段",
            ),
            _doc(
                "ai-agent-engineer.docx",
                "python3 -m docrag.cli query \"混合检索怎么做\"",
                heading_path="ai-agent-engineer.docx",
            ),
        ]
    )

    hits = LexicalRetriever(store).retrieve("混合检索怎么做？", top_k=2)

    assert hits[0].document.metadata["source"] == "rag-sample.md"
    assert "RRF" in hits[0].document.content
    print("✓ lexical bm25: explanatory heading beats command snippet")


def test_search_respects_filter_top_k_boundary_and_delete():
    store = LexicalDocumentStore()
    store.add(
        [
            _doc("current.md", "报销材料需要在 3 天内提交", is_latest=True),
            _doc("stale.md", "报销材料需要在 7 天内提交", is_latest=False),
        ]
    )

    assert store.search("报销材料提交", top_k=0) == []
    hits = store.search(
        "报销材料提交",
        top_k=3,
        metadata_filter={"is_latest": True},
    )
    assert len(hits) == 1
    assert hits[0].document.metadata["source"] == "current.md"

    assert store.delete_by_doc("current.md") == 1
    hits_after_delete = store.search(
        "报销材料提交",
        top_k=3,
        metadata_filter={"is_latest": True},
    )
    assert hits_after_delete == []
    print("✓ lexical search: filter/top_k/delete boundaries ok")
