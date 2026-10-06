"""Offline dataset runner, metrics, reports, and release gates for RAGX."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from config.settings import PROJECT_ROOT, Settings
from rag.app import RagApplication
from rag.evals.models import EvalCase, EvalCaseResult, EvalReport, EvalThresholds
from rag.providers import ProviderBundle

EVALUATOR_VERSION = "2"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_FIRST_CONTEXT_RE = re.compile(
    r"\[片段1[^\]]*\]\n(.*?)(?=\n\n\[片段2|\n\n【问题】)",
    re.DOTALL,
)


def default_dataset_path() -> Path:
    """Return the repository's deterministic core eval dataset."""
    return PROJECT_ROOT / "evals" / "datasets" / "core-rag.jsonl"


def default_thresholds_path() -> Path:
    """Return the repository's default release gates."""
    return PROJECT_ROOT / "evals" / "thresholds.json"


def load_eval_cases(path: str | Path | None = None) -> list[EvalCase]:
    """Load and validate JSONL eval cases."""
    dataset_path = Path(path) if path is not None else default_dataset_path()
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError("eval case must be a JSON object")  # noqa: TRY004
            case = EvalCase.from_mapping(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid eval case at {dataset_path}:{line_number}"
            ) from exc
        if case.case_id in seen:
            raise ValueError(f"duplicate eval case id: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"eval dataset is empty: {dataset_path}")
    return cases


def load_thresholds(path: str | Path | None = None) -> EvalThresholds:
    """Load and validate aggregate quality gates."""
    thresholds_path = Path(path) if path is not None else default_thresholds_path()
    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("eval thresholds must be a JSON object")  # noqa: TRY004
    return EvalThresholds.from_mapping(payload)


def run_eval_suite(
    cases: Sequence[EvalCase],
    thresholds: EvalThresholds | None = None,
    *,
    dataset_name: str = "in-memory",
    dataset_sha256: str = "",
    thresholds_sha256: str = "",
) -> EvalReport:
    """Run every case in an isolated local RAG application."""
    if not cases:
        raise ValueError("eval suite requires at least one case")
    active_thresholds = thresholds or EvalThresholds()
    results = tuple(_evaluate_case(case) for case in cases)
    metrics = _aggregate_metrics(results)
    threshold_values = {
        key: float(value) for key, value in asdict(active_thresholds).items()
    }
    failed_gates = tuple(_failed_gates(metrics, threshold_values))
    return EvalReport(
        suite="ragx-core-rag",
        dataset=dataset_name,
        dataset_sha256=dataset_sha256 or _canonical_sha256(cases),
        thresholds_sha256=(
            thresholds_sha256 or _canonical_sha256(active_thresholds)
        ),
        evaluator_version=EVALUATOR_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        passed=not failed_gates,
        coverage=_coverage(cases),
        metrics=metrics,
        thresholds=threshold_values,
        failed_gates=failed_gates,
        cases=results,
    )


def run_eval_suite_from_paths(
    dataset_path: str | Path | None = None,
    thresholds_path: str | Path | None = None,
) -> EvalReport:
    """Load repository artifacts and run a reproducible eval suite."""
    dataset = (
        Path(dataset_path) if dataset_path is not None else default_dataset_path()
    )
    thresholds_file = (
        Path(thresholds_path)
        if thresholds_path is not None
        else default_thresholds_path()
    )
    return run_eval_suite(
        load_eval_cases(dataset),
        load_thresholds(thresholds_file),
        dataset_name=dataset.name,
        dataset_sha256=_file_sha256(dataset),
        thresholds_sha256=_file_sha256(thresholds_file),
    )


def write_eval_report(report: EvalReport, path: str | Path) -> Path:
    """Write a JSON report and create its parent directory."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _evaluate_case(case: EvalCase) -> EvalCaseResult:
    with TemporaryDirectory(prefix=f"ragx-eval-{case.case_id}-") as temp:
        root = Path(temp)
        source_dir = root / "resources"
        source_dir.mkdir()
        for document in case.documents:
            (source_dir / document.source).write_text(
                document.content,
                encoding="utf-8",
            )

        settings = Settings(
            provider="eval",
            vector_backend="sqlite",
            sqlite_path=str(root / "ragx-store.sqlite"),
            manifest_path=str(root / "ragx-manifest.json"),
            search_backend="bi_encoder",
            reranker_backend="cross_encoder",
            top_k=case.top_k,
            chunk_size=220,
            chunk_overlap=30,
        )
        app = RagApplication(
            settings,
            providers=ProviderBundle(
                embedding=DeterministicEmbeddingProvider(),
                llm=ExtractiveEvidenceProvider(),
            ),
        )
        try:
            index_started = time.perf_counter()
            index_report = app.index(str(source_dir))
            index_latency_ms = (time.perf_counter() - index_started) * 1_000

            case_started = time.perf_counter()
            result = app.ask(case.query, top_k=case.top_k)
            case_latency_ms = (time.perf_counter() - case_started) * 1_000
        finally:
            close = getattr(app._store, "close", None)
            if callable(close):
                close()

    contexts = list(result.get("contexts", []))
    answer = str(result.get("answer", ""))
    sources = [str(context.get("source") or "") for context in contexts]
    relevant_rank = _first_relevant_rank(sources, case.relevant_sources)
    answer_term_hits = sum(
        _contains(answer, term) for term in case.expected_answer_terms
    )
    forbidden_hits = sum(
        _contains(answer, term) for term in case.forbidden_answer_terms
    )
    grounded = _is_grounded(answer, contexts)
    passed = (
        relevant_rank is not None
        and relevant_rank <= 3
        and answer_term_hits == len(case.expected_answer_terms)
        and forbidden_hits == 0
        and grounded
    )
    return EvalCaseResult(
        case_id=case.case_id,
        passed=passed,
        hit_at_1=relevant_rank == 1,
        hit_at_3=relevant_rank is not None and relevant_rank <= 3,
        reciprocal_rank=(
            round(1.0 / relevant_rank, 6) if relevant_rank is not None else 0.0
        ),
        answer_term_hits=answer_term_hits,
        answer_term_total=len(case.expected_answer_terms),
        grounded_answer=grounded,
        forbidden_term_hits=forbidden_hits,
        case_latency_ms=round(case_latency_ms, 3),
        index_latency_ms=round(index_latency_ms, 3),
        diagnostics={
            "category": case.category,
            "description": case.description,
            "answer": answer,
            "ranked_sources": sources,
            "relevant_rank": relevant_rank,
            "missing_answer_terms": [
                term
                for term in case.expected_answer_terms
                if not _contains(answer, term)
            ],
            "forbidden_answer_terms": [
                term
                for term in case.forbidden_answer_terms
                if _contains(answer, term)
            ],
            "context_count": len(contexts),
            "index": {
                "added": index_report.added,
                "updated": index_report.updated,
                "reembedded_chunks": index_report.reembedded_chunks,
            },
        },
    )


class DeterministicEmbeddingProvider:
    """Stable local hashing embeddings for zero-network evaluation."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class ExtractiveEvidenceProvider:
    """Return the first ranked evidence block as a deterministic answer."""

    def chat(self, messages: Sequence[object], **kwargs: object) -> str:
        del kwargs
        prompt = next(
            (
                str(getattr(message, "content", ""))
                for message in reversed(messages)
                if getattr(message, "role", "") == "user"
            ),
            "",
        )
        match = _FIRST_CONTEXT_RE.search(prompt)
        if match is None:
            return "无法从现有资料中确认"
        return match.group(1).strip()


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.casefold()):
        if raw.isascii():
            tokens.append(raw)
            continue
        tokens.extend(raw)
        tokens.extend(
            raw[index:index + 2]
            for index in range(max(0, len(raw) - 1))
        )
    return tokens


def _first_relevant_rank(
    sources: Sequence[str],
    relevant_sources: Sequence[str],
) -> int | None:
    relevant = set(relevant_sources)
    for index, source in enumerate(sources, start=1):
        if source in relevant:
            return index
    return None


def _contains(text: str, term: str) -> int:
    return int(_normalize_text(term) in _normalize_text(text))


def _is_grounded(answer: str, contexts: Sequence[Mapping[str, Any]]) -> bool:
    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return False
    evidence = _normalize_text(
        "\n".join(str(context.get("preview") or "") for context in contexts)
    )
    return normalized_answer in evidence


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _aggregate_metrics(results: Sequence[EvalCaseResult]) -> dict[str, float]:
    answer_hits = sum(item.answer_term_hits for item in results)
    answer_total = sum(item.answer_term_total for item in results)
    metrics = {
        "case_pass_rate": _ratio(sum(item.passed for item in results), len(results)),
        "hit_rate_at_1": _ratio(sum(item.hit_at_1 for item in results), len(results)),
        "hit_rate_at_3": _ratio(sum(item.hit_at_3 for item in results), len(results)),
        "mrr": _mean(item.reciprocal_rank for item in results),
        "answer_term_recall": _ratio(answer_hits, answer_total),
        "grounded_answer_rate": _ratio(
            sum(item.grounded_answer for item in results),
            len(results),
        ),
        "forbidden_term_violation_rate": _ratio(
            sum(item.forbidden_term_hits > 0 for item in results),
            len(results),
        ),
        "p95_case_latency_ms": _percentile(
            [item.case_latency_ms for item in results],
            0.95,
        ),
    }
    return {key: round(value, 6) for key, value in metrics.items()}


def _coverage(cases: Sequence[EvalCase]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for case in cases:
        coverage[case.category] = coverage.get(case.category, 0) + 1
    return dict(sorted(coverage.items()))


def _failed_gates(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> Iterable[str]:
    for threshold_name, threshold in thresholds.items():
        if threshold_name.startswith("min_"):
            metric_name = threshold_name[4:]
            if metrics[metric_name] < threshold:
                yield threshold_name
        elif threshold_name.startswith("max_"):
            metric_name = threshold_name[4:]
            if metrics[metric_name] > threshold:
                yield threshold_name


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 1.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        payload = [asdict(item) for item in value]
    else:
        payload = asdict(value)  # type: ignore[arg-type]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
