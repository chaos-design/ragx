"""Cross-Encoder 重排。

Cross-Encoder 阶段接收双路召回 + RRF 融合后的候选池，逐个对
`(query, document)` 做联合相关性打分，再结合少量召回原始分作为稳定项。
默认实现为本地 deterministic scorer，生产可注入真实 Cross-Encoder 模型适配器。
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag.interfaces import Document, ScoredDocument
from rag.retrieval.lexical import _query_phrases, _tokenize

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_QUESTION_FRAGMENTS = (
    "什么",
    "怎么",
    "如何",
    "是否",
    "几",
    "多少",
    "多久",
    "哪个",
    "哪些",
    "吗",
    "呢",
)
_TERM_ALIASES = {
    "记忆": ("memory",),
    "系统": ("system",),
    "智能体": ("agent",),
    "代理": ("agent",),
}


class CrossEncoderScorer(Protocol):
    """Cross-Encoder scorer interface."""

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        """Return one 0-1 relevance score per document."""
        ...


@dataclass(frozen=True)
class CrossEncoderRerankConfig:
    """Cross-Encoder 重排参数。"""

    cross_encoder_weight: float = 0.95
    retrieval_score_weight: float = 0.05


@dataclass(frozen=True)
class _QueryProfile:
    """Cross-Encoder 本地 scorer 使用的查询特征。"""

    terms: tuple[str, ...]
    phrases: tuple[str, ...]
    compact: str


class HeuristicCrossEncoderScorer(CrossEncoderScorer):
    """Deterministic local Cross-Encoder scorer.

    该实现不加载外部模型，但按 Cross-Encoder 的调用形态对 query-document pair
    做联合打分，便于测试和无模型环境运行。生产接入真实模型时只需实现
    `CrossEncoderScorer.score`。
    """

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        profile = _build_query_profile(query)
        return [_score_pair(profile, document) for document in documents]


class CrossEncoderReranker:
    """对候选池执行 Cross-Encoder 重排。"""

    def __init__(
        self,
        scorer: CrossEncoderScorer | None = None,
        config: CrossEncoderRerankConfig | None = None,
    ) -> None:
        self._scorer = scorer or HeuristicCrossEncoderScorer()
        self._config = config or CrossEncoderRerankConfig()

    def rerank(
        self,
        query: str,
        contexts: Sequence[ScoredDocument],
        *,
        top_k: int,
    ) -> list[ScoredDocument]:
        """按 Cross-Encoder 分数重排候选片段。"""
        if top_k <= 0 or not contexts:
            return []

        documents = [context.document for context in contexts]
        ce_scores = self._scorer.score(query, documents)
        raw_scores = _normalize_raw_scores([context.score for context in contexts])
        ranked: list[tuple[int, float, ScoredDocument]] = []
        for index, context in enumerate(contexts):
            score = _combined_score(
                ce_scores[index] if index < len(ce_scores) else 0.0,
                raw_scores[index],
                self._config,
            )
            ranked.append(
                (
                    index,
                    score,
                    ScoredDocument(document=context.document, score=score),
                )
            )

        ranked.sort(key=lambda item: (item[1], raw_scores[item[0]], -item[0]), reverse=True)
        return [item[2] for item in ranked[:top_k]]


def rank_by_relevance(
    query: str,
    contexts: Sequence[ScoredDocument],
    *,
    top_k: int,
    reranker: CrossEncoderReranker | None = None,
) -> list[ScoredDocument]:
    """兼容入口：用 Cross-Encoder reranker 重排候选。"""
    return (reranker or CrossEncoderReranker()).rerank(query, contexts, top_k=top_k)


def _combined_score(
    cross_encoder_score: float,
    raw_score: float,
    config: CrossEncoderRerankConfig,
) -> float:
    score = (
        _clamp(cross_encoder_score) * config.cross_encoder_weight
        + _clamp(raw_score) * config.retrieval_score_weight
    )
    return round(_clamp(score), 6)


def _build_query_profile(query: str) -> _QueryProfile:
    """构建查询评估特征。"""
    base_terms = (term for term in _tokenize(query) if not _is_query_noise(term))
    terms = tuple(_unique(_expand_aliases(base_terms)))
    phrases = tuple(
        _unique(phrase for phrase in _query_phrases(query) if not _is_query_noise(phrase))
    )
    return _QueryProfile(terms=terms, phrases=phrases, compact=_compact(query))


def _expand_aliases(terms: Iterable[str]) -> Iterable[str]:
    """扩展少量领域术语别名，补齐中文 query 与英文文件名的 metadata 匹配。"""
    for term in terms:
        yield term
        for alias in _TERM_ALIASES.get(term, ()):
            yield alias


def _is_query_noise(term: str) -> bool:
    """过滤疑问句语气片段，避免“是什么”等泛化词干扰证据评估。"""
    if not term:
        return True
    if any(fragment in term for fragment in _QUESTION_FRAGMENTS):
        return True
    return bool(_CJK_RE.search(term) and "是" in term and len(term) > 1)


def _score_pair(profile: _QueryProfile, document: Document) -> float:
    """计算 query-document pair 的联合相关度。"""
    content = str(document.content or "").lower()
    heading = _metadata_text(document.metadata.get("heading_path")).lower()
    source = _metadata_text(document.metadata.get("source")).lower()
    metadata_text = f"{heading} {source}".strip()
    searchable_text = f"{content} {metadata_text}".strip()

    term_score = _term_coverage(profile.terms, searchable_text, metadata_text)
    phrase_score = _phrase_coverage(profile.phrases, content, heading, source)
    metadata_score = _metadata_coverage(profile.terms, profile.phrases, metadata_text)
    source_score = _source_coverage(profile.terms, source)
    exact_score = _exact_query_score(profile.compact, searchable_text)

    score = (
        term_score * 0.41
        + phrase_score * 0.30
        + metadata_score * 0.14
        + source_score * 0.10
        + exact_score * 0.05
        + _coverage_confidence_bonus(term_score, phrase_score)
    )
    return round(_clamp(score), 6)


def _coverage_confidence_bonus(term_score: float, phrase_score: float) -> float:
    """对高覆盖候选追加置信加权，避免正确证据分数被压低。"""
    bonus = 0.0
    if term_score >= 0.60:
        bonus += 0.06
    if phrase_score >= 0.40:
        bonus += 0.04
    return bonus


def _normalize_raw_scores(scores: Sequence[float]) -> list[float]:
    """把任意后端原始分归一化到 0-1。"""
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [1.0 if score > 0 else 0.0 for score in scores]
    span = maximum - minimum
    return [_clamp((score - minimum) / span) for score in scores]


def _term_coverage(
    query_terms: Sequence[str],
    searchable_text: str,
    metadata_text: str,
) -> float:
    """计算查询词项在候选片段中的加权覆盖率。"""
    if not query_terms:
        return 0.0

    searchable_terms = set(_tokenize(searchable_text))
    metadata_terms = set(_tokenize(metadata_text))
    total_weight = 0.0
    matched_weight = 0.0
    for term in query_terms:
        weight = _term_weight(term)
        total_weight += weight
        if term in searchable_terms or term in searchable_text:
            matched_weight += weight
        elif term in metadata_terms or term in metadata_text:
            matched_weight += weight * 0.85
    return _safe_ratio(matched_weight, total_weight)


def _phrase_coverage(
    query_phrases: Sequence[str],
    content: str,
    heading: str,
    source: str,
) -> float:
    """计算查询短语在正文、标题和来源中的命中比例。"""
    if not query_phrases:
        return 0.0

    total_weight = 0.0
    matched_weight = 0.0
    for phrase in query_phrases:
        weight = _term_weight(phrase)
        total_weight += weight
        if phrase in content:
            matched_weight += weight
            continue
        if phrase in heading:
            matched_weight += weight * 0.95
            continue
        if phrase in source:
            matched_weight += weight * 0.55
    return _safe_ratio(matched_weight, total_weight)


def _metadata_coverage(
    query_terms: Sequence[str],
    query_phrases: Sequence[str],
    metadata_text: str,
) -> float:
    """计算标题与来源 metadata 对查询的支持度。"""
    if not metadata_text:
        return 0.0

    metadata_terms = set(_tokenize(metadata_text))
    term_hits = sum(
        _term_weight(term)
        for term in query_terms
        if term in metadata_terms or term in metadata_text
    )
    term_total = sum(_term_weight(term) for term in query_terms)
    phrase_hits = sum(
        _term_weight(phrase) for phrase in query_phrases if phrase in metadata_text
    )
    phrase_total = sum(_term_weight(phrase) for phrase in query_phrases)

    term_score = _safe_ratio(term_hits, term_total)
    phrase_score = _safe_ratio(phrase_hits, phrase_total)
    return max(term_score, phrase_score)


def _source_coverage(query_terms: Sequence[str], source: str) -> float:
    """计算文件名/来源对查询核心词的支持度。"""
    if not query_terms or not source:
        return 0.0
    source_terms = set(_tokenize(source.replace("-", " ")))
    matched = 0.0
    total = 0.0
    for term in query_terms:
        if _CJK_RE.search(term):
            continue
        weight = _term_weight(term)
        total += weight
        if term in source_terms or term in source:
            matched += weight
    return _safe_ratio(matched, total)


def _exact_query_score(compact_query: str, searchable_text: str) -> float:
    """判断候选片段是否包含去标点后的完整查询表达。"""
    if len(compact_query) < 4:
        return 0.0
    return 1.0 if compact_query in _compact(searchable_text) else 0.0


def _metadata_text(value: Any) -> str:
    """把 metadata 值转换为可检索文本。"""
    if isinstance(value, (list, tuple)):
        return " > ".join(str(item) for item in value)
    return str(value or "")


def _term_weight(term: str) -> float:
    """按词项长度给实体短语更高权重。"""
    return 1.0 + min(len(term), 8) * 0.12


def _safe_ratio(numerator: float, denominator: float) -> float:
    """安全计算 0-1 比率。"""
    if denominator <= 0:
        return 0.0
    return _clamp(numerator / denominator)


def _compact(text: str) -> str:
    """移除空白和标点，生成完整查询匹配键。"""
    return _PUNCT_RE.sub("", str(text or "").lower())


def _clamp(value: float) -> float:
    """把数值限制在 0-1 区间。"""
    return max(0.0, min(1.0, float(value)))


def _unique(values: Iterable[str]) -> list[str]:
    """按输入顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
