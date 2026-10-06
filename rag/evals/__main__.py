"""Command-line entry point for the deterministic RAGX eval suite."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag.evals.runner import (
    default_dataset_path,
    default_thresholds_path,
    run_eval_suite_from_paths,
    write_eval_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone eval CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run deterministic RAGX retrieval and grounding evals.",
    )
    parser.add_argument("--dataset", default=str(default_dataset_path()))
    parser.add_argument("--thresholds", default=str(default_thresholds_path()))
    parser.add_argument("--output")
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite, print JSON, and optionally enforce release gates."""
    args = build_parser().parse_args(argv)
    report = run_eval_suite_from_paths(args.dataset, args.thresholds)
    if args.output:
        write_eval_report(report, Path(args.output))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if args.fail_on_regression and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
