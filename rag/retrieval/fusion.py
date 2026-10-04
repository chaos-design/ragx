"""检索融合策略。

当前生产策略是 RRF(Reciprocal Rank Fusion)：对向量召回和 BM25 词法召回
分别保留排序位置，再按 1 / (K + rank) 融合。默认 K=60。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.interfaces import Document, ScoredDocument


@dataclass(frozen=True)
class FusionConfig:
    """RRF 融合参数。"""

    candidate_k: int = 100
    rrf_k: int = 60
    vector_weight: float = 1.0
    lexical_weight: float = 1.5
    raw_score_weight: float = 0.05
    source_diversity_penalty: float = 0.01


@dataclass
class _FusionEntry:
    """融合过程中的内部候选项。"""

    document: Document
    score: float = 0.0
    best_raw_score: float = 0.0


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredDocument]],
    *,
    weights: Sequence[float],
    top_k: int,
    rrf_k: int = 60,
    raw_score_weight: float = 0.05,
    source_diversity_penalty: float = 0.01,
) -> list[ScoredDocument]:
    """用 RRF 融合多路检索结果。

    输入 (Input):
        ranked_lists: 多个已按相关性降序排列的检索结果列表。
        weights: 与 ranked_lists 对齐的通道权重。
        top_k: 最终返回数量。
        rrf_k: RRF 平滑常量，默认 60。
        raw_score_weight: 原始分归一化后的轻量补偿权重。
        source_diversity_penalty: 同一 source 连续占位时的降权。

    输出 (Output):
        融合、去重并做 source 多样性控制后的 ScoredDocument 列表。

    示例 (Example):
        reciprocal_rank_fusion([[hit_a], [hit_b]], weights=[1.0, 1.0], top_k=2)
    """
    if top_k <= 0:
        return []

    entries: dict[str, _FusionEntry] = {}
    for channel_index, ranked in enumerate(ranked_lists):
        weight = weights[channel_index] if channel_index < len(weights) else 1.0
        max_abs_score = max((abs(hit.score) for hit in ranked), default=0.0) or 1.0
        for rank, hit in enumerate(ranked, 1):
            chunk_id = _chunk_id(hit.document)
            entry = entries.setdefault(chunk_id, _FusionEntry(document=hit.document))
            entry.score += weight / (rrf_k + rank)
            entry.score += weight * raw_score_weight * (hit.score / max_abs_score)
            entry.best_raw_score = max(entry.best_raw_score, hit.score)

    ordered = sorted(entries.values(), key=lambda item: item.score, reverse=True)
    return _apply_source_diversity(
        ordered,
        top_k=top_k,
        source_diversity_penalty=source_diversity_penalty,
    )


def _apply_source_diversity(
    entries: Sequence[_FusionEntry],
    *,
    top_k: int,
    source_diversity_penalty: float,
) -> list[ScoredDocument]:
    """对融合候选做轻量 source 多样性重排。"""
    remaining = list(entries)
    selected: list[ScoredDocument] = []
    source_counts: dict[str, int] = {}

    while remaining and len(selected) < top_k:
        best_index = 0
        best_adjusted_score = float("-inf")
        for index, entry in enumerate(remaining):
            source = str(entry.document.metadata.get("source") or "")
            adjusted_score = (
                entry.score - source_diversity_penalty * source_counts.get(source, 0)
            )
            if adjusted_score > best_adjusted_score:
                best_index = index
                best_adjusted_score = adjusted_score
        chosen = remaining.pop(best_index)
        source = str(chosen.document.metadata.get("source") or "")
        source_counts[source] = source_counts.get(source, 0) + 1
        selected.append(ScoredDocument(document=chosen.document, score=chosen.score))
    return selected


def _chunk_id(document: Document) -> str:
    """返回可用于去重的 chunk 标识。"""
    if document.id:
        return document.id
    doc_id = document.metadata.get("doc_id") or document.metadata.get("source") or ""
    chunk_index = document.metadata.get("chunk_index", "")
    return f"{doc_id}::{chunk_index}"
