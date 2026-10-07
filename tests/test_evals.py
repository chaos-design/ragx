"""Tests for the deterministic RAGX quality evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.evals import (
    EvalThresholds,
    load_eval_cases,
    load_thresholds,
    run_eval_suite,
    run_eval_suite_from_paths,
    write_eval_report,
)
from rag.evals.__main__ import main


def test_core_eval_dataset_passes_quality_gates() -> None:
    cases = load_eval_cases()
    report = run_eval_suite_from_paths()

    assert len(cases) == 16
    assert report.passed is True
    assert report.failed_gates == ()
    assert report.metrics["case_pass_rate"] == 1.0
    assert report.metrics["hit_rate_at_1"] == 1.0
    assert report.metrics["answer_term_recall"] == 1.0
    assert report.metrics["grounded_answer_rate"] == 1.0
    assert sum(report.coverage.values()) == len(cases)
    assert report.coverage["policy-fact"] == 2
    assert len(report.dataset_sha256) == 64
    assert len(report.thresholds_sha256) == 64


def test_core_eval_dataset_covers_required_failure_modes() -> None:
    cases = load_eval_cases()
    categories = {case.category for case in cases}

    assert {
        "configuration",
        "exact-match",
        "fusion",
        "incremental-index",
        "long-context",
        "metadata-signal",
        "mixed-language",
        "numeric-disambiguation",
        "parent-child",
        "policy-fact",
        "provenance",
        "security-boundary",
        "semantic-rewrite",
        "structure",
        "versioning",
    } <= categories
    assert all(len(case.documents) >= 2 for case in cases)
    assert all(case.forbidden_answer_terms for case in cases)
    assert all(
        set(case.relevant_sources) < {document.source for document in case.documents}
        for case in cases
    )


def test_eval_threshold_regression_is_reported() -> None:
    report = run_eval_suite(
        load_eval_cases()[:1],
        EvalThresholds(max_p95_case_latency_ms=-1.0),
    )

    assert report.passed is False
    assert report.failed_gates == ("max_p95_case_latency_ms",)


def test_eval_loader_rejects_duplicate_invalid_and_unsafe_cases(
    tmp_path: Path,
) -> None:
    valid = {
        "case_id": "same",
        "documents": [
            {
                "source": "guide.md",
                "content": "A sufficiently long evaluation document with expected evidence.",
            }
        ],
        "query": "expected evidence",
        "relevant_sources": ["guide.md"],
        "expected_answer_terms": ["evidence"],
    }
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        json.dumps(valid) + "\n" + json.dumps(valid) + "\n",
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{}\n", encoding="utf-8")
    unsafe = tmp_path / "unsafe.jsonl"
    unsafe_payload = {
        **valid,
        "case_id": "unsafe",
        "documents": [{"source": "../escape.md", "content": "unsafe"}],
        "relevant_sources": ["../escape.md"],
    }
    unsafe.write_text(json.dumps(unsafe_payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate eval case id"):
        load_eval_cases(duplicate)
    with pytest.raises(ValueError, match="invalid eval case"):
        load_eval_cases(invalid)
    with pytest.raises(ValueError, match="invalid eval case"):
        load_eval_cases(unsafe)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "Not A Slug"),
        ("category", "Bad Category"),
        ("relevant_sources", ["guide.md", "guide.md"]),
        ("expected_answer_terms", ["evidence", "evidence"]),
        ("forbidden_answer_terms", ["evidence"]),
    ],
)
def test_eval_loader_rejects_low_quality_case_contracts(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = {
        "case_id": "quality-contract",
        "category": "contract",
        "documents": [
            {
                "source": "guide.md",
                "content": "Expected evidence is present in this isolated document.",
            },
            {
                "source": "distractor.md",
                "content": "A plausible but incorrect statement is present here.",
            },
        ],
        "query": "Where is the expected evidence?",
        "relevant_sources": ["guide.md"],
        "expected_answer_terms": ["evidence"],
        "forbidden_answer_terms": ["incorrect"],
    }
    payload[field] = value
    dataset = tmp_path / "invalid-contract.jsonl"
    dataset.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid eval case"):
        load_eval_cases(dataset)


def test_eval_threshold_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text('{"min_unknown_metric": 1.0}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown eval thresholds"):
        load_thresholds(thresholds)


@pytest.mark.parametrize(
    "payload",
    [
        {"min_case_pass_rate": -0.01},
        {"min_mrr": 1.01},
        {"max_forbidden_term_violation_rate": 1.01},
        {"max_p95_case_latency_ms": -1},
    ],
)
def test_eval_threshold_loader_rejects_out_of_range_values(
    tmp_path: Path,
    payload: dict[str, float],
) -> None:
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be"):
        load_thresholds(thresholds)


def test_eval_cli_writes_report_and_honors_gate_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "report.json"

    exit_code = main(["--output", str(output_path), "--fail-on-regression"])
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["coverage"]["exact-match"] == 1
    assert written["suite"] == "ragx-core-rag"
    assert write_eval_report(
        run_eval_suite(load_eval_cases()),
        output_path,
    ) == output_path

    strict_thresholds = tmp_path / "strict-thresholds.json"
    strict_thresholds.write_text(
        '{"max_p95_case_latency_ms": 0.0}',
        encoding="utf-8",
    )
    failed_exit_code = main(
        [
            "--thresholds",
            str(strict_thresholds),
            "--fail-on-regression",
        ]
    )
    failed_payload = json.loads(capsys.readouterr().out)

    assert failed_exit_code == 1
    assert failed_payload["passed"] is False
    assert failed_payload["failed_gates"] == ["max_p95_case_latency_ms"]
