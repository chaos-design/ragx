"""检索效果评估。

本模块只负责评估已经排序好的检索结果，不参与召回、融合或 reranking。
有标注数据时输出 Hit@K、Precision@K、Recall@K 和 MRR；没有标注数据时
输出候选数量、平均分、最高分和 source 多样性等诊断指标。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.interfaces import ScoredDocument


@dataclass(frozen=True)
class EvaluationConfig:
    """效果评估配置。"""

    top_k_values: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """单次 query 的检索效果评估结果。"""

    query: str
    candidate_count: int
    labelled: bool
    best_rank: int | None
    mrr: float
    hit_at_k: dict[int, bool]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mean_score: float
    max_score: float
    source_diversity: float

    def to_dict(self) -> dict:
        """转换为可 JSON 序列化的字典。"""
        return {
            "query": self.query,
            "candidate_count": self.candidate_count,
            "labelled": self.labelled,
            "best_rank": self.best_rank,
            "mrr": self.mrr,
            "hit_at_k": self.hit_at_k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mean_score": self.mean_score,
            "max_score": self.max_score,
            "source_diversity": self.source_diversity,
        }


def evaluate_retrieval_effect(
    query: str,
    ranked_contexts: Sequence[ScoredDocument],
    *,
    relevant_sources: Sequence[str] = (),
    relevant_doc_ids: Sequence[str] = (),
    relevant_chunk_ids: Sequence[str] = (),
    config: EvaluationConfig | None = None,
) -> RetrievalEvaluationReport:
    """评估一次 query 的排序结果。

    输入 (Input):
        query: 用户查询。
        ranked_contexts: 已完成融合和 reranking 的结果，按相关性降序排列。
        relevant_sources: 可选标注，认为相关的 source 文件名集合。
        relevant_doc_ids: 可选标注，认为相关的 doc_id 集合。
        relevant_chunk_ids: 可选标注，认为相关的 chunk id 集合。
        config: top_k 指标配置。

    输出 (Output):
        RetrievalEvaluationReport。

    示例 (Example):
        evaluate_retrieval_effect("报销材料", hits, relevant_sources=["rule.md"])
    """
    cfg = config or EvaluationConfig()
    contexts = list(ranked_contexts)
    relevant = _RelevantSets(
        sources=frozenset(relevant_sources),
        doc_ids=frozenset(relevant_doc_ids),
        chunk_ids=frozenset(relevant_chunk_ids),
    )
    labelled = relevant.has_labels
    relevant_flags = [_is_relevant(hit, relevant) for hit in contexts]
    best_rank = _best_rank(relevant_flags) if labelled else None

    return RetrievalEvaluationReport(
        query=query,
        candidate_count=len(contexts),
        labelled=labelled,
        best_rank=best_rank,
        mrr=round(1.0 / best_rank, 6) if best_rank else 0.0,
        hit_at_k=_hit_at_k(relevant_flags, cfg.top_k_values) if labelled else {},
        precision_at_k=(
            _precision_at_k(relevant_flags, cfg.top_k_values) if labelled else {}
        ),
        recall_at_k=(
            _recall_at_k(relevant_flags, cfg.top_k_values) if labelled else {}
        ),
        mean_score=round(_mean([hit.score for hit in contexts]), 6),
        max_score=round(max((hit.score for hit in contexts), default=0.0), 6),
        source_diversity=round(_source_diversity(contexts), 6),
    )


@dataclass(frozen=True)
class _RelevantSets:
    """相关性标注集合。"""

    sources: frozenset[str]
    doc_ids: frozenset[str]
    chunk_ids: frozenset[str]

    @property
    def has_labels(self) -> bool:
        return bool(self.sources or self.doc_ids or self.chunk_ids)


def _is_relevant(hit: ScoredDocument, relevant: _RelevantSets) -> bool:
    """判断单条命中是否匹配相关性标注。"""
    metadata = hit.document.metadata
    source = str(metadata.get("source") or "")
    doc_id = str(metadata.get("doc_id") or "")
    chunk_id = str(hit.document.id or "")
    return (
        source in relevant.sources
        or doc_id in relevant.doc_ids
        or chunk_id in relevant.chunk_ids
    )


def _best_rank(relevant_flags: Sequence[bool]) -> int | None:
    """返回第一条相关结果的 1-based rank。"""
    for index, relevant in enumerate(relevant_flags, 1):
        if relevant:
            return index
    return None


def _hit_at_k(
    relevant_flags: Sequence[bool],
    top_k_values: Sequence[int],
) -> dict[int, bool]:
    """计算 Hit@K。"""
    return {
        k: any(relevant_flags[:k])
        for k in _valid_top_k_values(top_k_values)
    }


def _precision_at_k(
    relevant_flags: Sequence[bool],
    top_k_values: Sequence[int],
) -> dict[int, float]:
    """计算 Precision@K。"""
    out: dict[int, float] = {}
    for k in _valid_top_k_values(top_k_values):
        window = relevant_flags[:k]
        denominator = min(k, len(relevant_flags))
        out[k] = round(sum(window) / denominator, 6) if denominator else 0.0
    return out


def _recall_at_k(
    relevant_flags: Sequence[bool],
    top_k_values: Sequence[int],
) -> dict[int, float]:
    """计算 Recall@K。"""
    total_relevant = sum(relevant_flags)
    if total_relevant == 0:
        return {k: 0.0 for k in _valid_top_k_values(top_k_values)}
    return {
        k: round(sum(relevant_flags[:k]) / total_relevant, 6)
        for k in _valid_top_k_values(top_k_values)
    }


def _valid_top_k_values(values: Sequence[int]) -> tuple[int, ...]:
    """过滤非法 K，并去重排序。"""
    return tuple(sorted({int(value) for value in values if int(value) > 0}))


def _mean(values: Sequence[float]) -> float:
    """计算均值。"""
    return sum(values) / len(values) if values else 0.0


def _source_diversity(contexts: Sequence[ScoredDocument]) -> float:
    """计算 source 多样性，范围 0-1。"""
    if not contexts:
        return 0.0
    sources = {str(hit.document.metadata.get("source") or "") for hit in contexts}
    return len(sources) / len(contexts)
