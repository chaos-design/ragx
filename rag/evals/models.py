"""Typed contracts for repeatable RAGX quality evaluations."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class EvalDocument:
    """One synthetic source document owned by an evaluation case."""

    source: str
    content: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EvalDocument:
        source = str(payload.get("source", "")).strip()
        content = str(payload.get("content", "")).strip()
        if (
            not source
            or not content
            or Path(source).name != source
            or source in {".", ".."}
        ):
            raise ValueError(
                "eval documents require a safe basename source and non-empty content"
            )
        return cls(source=source, content=content)


@dataclass(frozen=True)
class EvalCase:
    """An isolated indexing, retrieval, and grounded-generation scenario."""

    case_id: str
    category: str
    description: str
    documents: tuple[EvalDocument, ...]
    query: str
    relevant_sources: tuple[str, ...]
    expected_answer_terms: tuple[str, ...]
    forbidden_answer_terms: tuple[str, ...] = field(default_factory=tuple)
    top_k: int = 3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EvalCase:
        case_id = str(payload.get("case_id", "")).strip()
        category = str(payload.get("category", "general")).strip()
        query = str(payload.get("query", "")).strip()
        documents = tuple(
            EvalDocument.from_mapping(item)
            for item in _mapping_sequence(payload.get("documents", []), "documents")
        )
        relevant_sources = _text_tuple(
            payload.get("relevant_sources", []),
            "relevant_sources",
        )
        expected_terms = _text_tuple(
            payload.get("expected_answer_terms", []),
            "expected_answer_terms",
        )
        forbidden_terms = _text_tuple(
            payload.get("forbidden_answer_terms", []),
            "forbidden_answer_terms",
            allow_empty=True,
        )
        if not case_id or not query or not documents:
            raise ValueError(
                "eval cases require case_id, query, and non-empty documents"
            )
        if not _SLUG_RE.fullmatch(case_id) or not _SLUG_RE.fullmatch(category):
            raise ValueError("eval case_id and category must be lowercase slugs")
        if not relevant_sources or not expected_terms:
            raise ValueError(
                "eval cases require relevant_sources and expected_answer_terms"
            )
        sources = [document.source for document in documents]
        if len(sources) != len(set(sources)):
            raise ValueError(f"eval case contains duplicate source: {case_id}")
        if len(relevant_sources) != len(set(relevant_sources)):
            raise ValueError(f"eval case contains duplicate relevant source: {case_id}")
        if len(expected_terms) != len(set(expected_terms)):
            raise ValueError(f"eval case contains duplicate expected term: {case_id}")
        if len(forbidden_terms) != len(set(forbidden_terms)):
            raise ValueError(f"eval case contains duplicate forbidden term: {case_id}")
        conflicting_terms = sorted(
            {term.casefold() for term in expected_terms}
            & {term.casefold() for term in forbidden_terms}
        )
        if conflicting_terms:
            raise ValueError(
                f"eval case contains expected/forbidden term conflicts: {case_id}"
            )
        missing_sources = sorted(set(relevant_sources) - set(sources))
        if missing_sources:
            raise ValueError(
                f"eval case relevant_sources are missing documents: {missing_sources}"
            )
        top_k = int(payload.get("top_k", 3))
        if top_k <= 0:
            raise ValueError("eval case top_k must be positive")
        return cls(
            case_id=case_id,
            category=category,
            description=str(payload.get("description", "")).strip(),
            documents=documents,
            query=query,
            relevant_sources=relevant_sources,
            expected_answer_terms=expected_terms,
            forbidden_answer_terms=forbidden_terms,
            top_k=top_k,
        )


@dataclass(frozen=True)
class EvalThresholds:
    """Release gates applied to aggregate RAG quality metrics."""

    min_case_pass_rate: float = 1.0
    min_hit_rate_at_1: float = 0.75
    min_hit_rate_at_3: float = 1.0
    min_mrr: float = 0.85
    min_answer_term_recall: float = 1.0
    min_grounded_answer_rate: float = 1.0
    max_forbidden_term_violation_rate: float = 0.0
    max_p95_case_latency_ms: float = 250.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EvalThresholds:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown eval thresholds: {unknown}")
        thresholds = cls(**{key: float(value) for key, value in payload.items()})
        for name, value in asdict(thresholds).items():
            if name == "max_p95_case_latency_ms":
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
            elif not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        return thresholds


@dataclass(frozen=True)
class EvalCaseResult:
    """Measured result for one isolated RAG evaluation case."""

    case_id: str
    passed: bool
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    answer_term_hits: int
    answer_term_total: int
    grounded_answer: bool
    forbidden_term_hits: int
    case_latency_ms: float
    index_latency_ms: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalReport:
    """Complete suite result with reproducibility metadata and gate failures."""

    suite: str
    dataset: str
    dataset_sha256: str
    thresholds_sha256: str
    evaluator_version: str
    generated_at: str
    passed: bool
    coverage: dict[str, int]
    metrics: dict[str, float]
    thresholds: dict[str, float]
    failed_gates: tuple[str, ...]
    cases: tuple[EvalCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        payload["cases"] = [item.to_dict() for item in self.cases]
        return payload


def _mapping_sequence(value: object, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")  # noqa: TRY004
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{field_name} must contain JSON objects")
    return tuple(value)


def _text_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")  # noqa: TRY004
    items = tuple(str(item).strip() for item in value)
    if any(not item for item in items) or (not items and not allow_empty):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return items
