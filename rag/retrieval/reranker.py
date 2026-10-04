"""Backward-compatible reranker exports and strategy selection."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.retrieval.cross_encoder import (
    CrossEncoderRerankConfig,
    CrossEncoderReranker,
    CrossEncoderScorer,
    HeuristicCrossEncoderScorer,
    rank_by_relevance,
)

RERANKER_BACKEND_CROSS_ENCODER = "cross_encoder"
RERANKER_BACKEND_COLBERT = "colbert"
RERANKER_BACKEND_NONE = "none"


class CandidateReranker(Protocol):
    """Common interface for candidate rerankers."""

    def rerank(self, query: str, contexts: Sequence, *, top_k: int) -> list:
        """Return reranked contexts.

        Example Input:
            reranker.rerank("query", contexts, top_k=3)

        Example Output:
            A list of scored contexts ordered by relevance.
        """
        ...


class NoOpReranker:
    """Return fused retrieval candidates without a second-stage reranker."""

    def rerank(self, query: str, contexts: Sequence, *, top_k: int) -> list:
        """Return the first `top_k` contexts unchanged.

        Example Input:
            NoOpReranker().rerank("query", contexts, top_k=3)

        Example Output:
            The first three contexts from the existing fused order.
        """
        del query
        if top_k <= 0:
            return []
        return list(contexts)[:top_k]


def normalize_reranker_backend(value: str | None) -> str:
    """Normalize reranker backend names and aliases.

    Example Input:
        normalize_reranker_backend("cross-encoder")

    Example Output:
        "cross_encoder"
    """
    normalized = (value or RERANKER_BACKEND_CROSS_ENCODER).strip().lower()
    normalized = normalized.replace("-", "_")
    aliases = {
        "ce": RERANKER_BACKEND_CROSS_ENCODER,
        "crossencoder": RERANKER_BACKEND_CROSS_ENCODER,
        "cross_encoder": RERANKER_BACKEND_CROSS_ENCODER,
        "colbert": RERANKER_BACKEND_COLBERT,
        "off": RERANKER_BACKEND_NONE,
        "none": RERANKER_BACKEND_NONE,
        "no_rerank": RERANKER_BACKEND_NONE,
        "bi_encoder": RERANKER_BACKEND_NONE,
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported reranker backend: "
            f"{value}. Expected cross_encoder, colbert, or none."
        )
    return aliases[normalized]


def build_reranker(cfg) -> CandidateReranker:
    """Build the configured reranker.

    Example Input:
        build_reranker(Settings(reranker_backend="colbert"))

    Example Output:
        A ColBERTReranker instance.
    """
    backend = normalize_reranker_backend(getattr(cfg, "reranker_backend", None))
    if backend == RERANKER_BACKEND_COLBERT:
        from rag.retrieval.colbert import ColBERTRerankConfig, ColBERTReranker

        return ColBERTReranker(
            config=ColBERTRerankConfig(
                model_name=getattr(cfg, "colbert_model", ""),
                query_max_tokens=getattr(cfg, "colbert_query_max_tokens", 32),
                document_max_tokens=getattr(cfg, "colbert_document_max_tokens", 180),
                batch_size=getattr(cfg, "colbert_batch_size", 8),
                interaction_weight=getattr(cfg, "colbert_interaction_weight", 0.95),
                retrieval_score_weight=getattr(
                    cfg,
                    "colbert_retrieval_score_weight",
                    0.05,
                ),
                hashing_dim=getattr(cfg, "colbert_hashing_dim", 64),
            )
        )
    if backend == RERANKER_BACKEND_NONE:
        return NoOpReranker()
    return CrossEncoderReranker(
        config=CrossEncoderRerankConfig(
            cross_encoder_weight=getattr(cfg, "cross_encoder_weight", 0.95),
            retrieval_score_weight=getattr(
                cfg,
                "cross_encoder_retrieval_score_weight",
                0.05,
            ),
        )
    )


__all__ = [
    "CandidateReranker",
    "CrossEncoderRerankConfig",
    "CrossEncoderReranker",
    "CrossEncoderScorer",
    "HeuristicCrossEncoderScorer",
    "NoOpReranker",
    "RERANKER_BACKEND_COLBERT",
    "RERANKER_BACKEND_CROSS_ENCODER",
    "RERANKER_BACKEND_NONE",
    "build_reranker",
    "normalize_reranker_backend",
    "rank_by_relevance",
]
