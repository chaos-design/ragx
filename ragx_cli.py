"""RAGX 模块化命令行入口。

提供 index、query、evaluate 和 run 四个子命令，使索引构建、查询处理、
效果评估和完整链路都可以独立调用与验证。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rag.entrypoints import (
    DEFAULT_SOURCE,
    DEFAULT_TOP_K_VALUES,
    run_evaluation,
    run_full_pipeline,
    run_index,
    run_query,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 RAGX 模块化 CLI 参数。

    Example Input:
        parse_args(["query", "RAGX 是什么？", "--top-k", "3"])

    Example Output:
        argparse.Namespace(command="query", query="RAGX 是什么？", top_k=3)
    """
    parser = argparse.ArgumentParser(
        description="RAGX 独立模块验证 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_index_command(subparsers)
    _add_query_command(subparsers)
    _add_evaluate_command(subparsers)
    _add_run_command(subparsers)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行 RAGX 模块化 CLI。

    Example Input:
        main(["index", "--source", "agent-library/ragx/data/resources"])

    Example Output:
        0
    """
    args = parse_args(argv)
    try:
        result = dispatch(args)
    except Exception as exc:  # noqa: BLE001 - CLI 边界负责转换为清晰错误。
        print(f"[ragx] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _warn_if_mock_provider()
    print_result(result, as_json=args.as_json)
    return 0


def _warn_if_mock_provider() -> None:
    """在使用了 mock provider 时输出醒目警告。

    mock 是显式 opt-in，但 opt-in 不等于「用户知道自己拿到了假答案」。
    检索链路真实工作、生成环节返回占位文本，这种组合最容易让人误判系统可用，
    因此必须在唯一的 CLI 出口无条件提示。

    Example Input:
        _warn_if_mock_provider()

    Example Output:
        stderr: [ragx] 警告：当前使用 mock provider ...
    """
    from config.settings import load_settings

    try:
        cfg = load_settings()
    except Exception:  # noqa: BLE001 - 警告失败不应影响主流程。
        return
    if cfg.provider.lower().strip() != "mock":
        return
    message = (
        "[ragx] 警告：当前使用 mock provider —— 检索链路真实执行，"
        "但 embedding 与生成均为本地确定性桩，产物不可用于生产。"
    )
    # 无论是否 --json，警告都只走 stderr，避免污染 JSON 输出契约。
    print(message, file=sys.stderr)


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    """把解析后的命令分派到独立 entrypoint。

    Example Input:
        dispatch(argparse.Namespace(command="index", source="data"))

    Example Output:
        {"module": "index", ...}
    """
    if args.command == "index":
        return run_index(
            args.source,
            sqlite_path=args.sqlite_path,
            manifest_path=args.manifest_path,
            reset=args.reset,
            acl=args.acl,
        )
    if args.command == "query":
        return run_query(
            args.query,
            sqlite_path=args.sqlite_path,
            manifest_path=args.manifest_path,
            reranker_backend=args.reranker_backend,
            metadata_filter=args.metadata_filter,
            top_k=args.top_k,
        )
    if args.command == "evaluate":
        return run_evaluation(
            query=args.query,
            dataset_path=args.dataset,
            sqlite_path=args.sqlite_path,
            manifest_path=args.manifest_path,
            reranker_backend=args.reranker_backend,
            metadata_filter=args.metadata_filter,
            relevant_sources=args.relevant_source,
            relevant_doc_ids=args.relevant_doc_id,
            relevant_chunk_ids=args.relevant_chunk_id,
            top_k_values=args.top_k_values,
        )
    if args.command == "run":
        return run_full_pipeline(
            args.source,
            args.query,
            sqlite_path=args.sqlite_path,
            manifest_path=args.manifest_path,
            reranker_backend=args.reranker_backend,
            reset=args.reset,
            acl=args.acl,
            metadata_filter=args.metadata_filter,
            top_k=args.top_k,
            evaluate=args.evaluate,
            relevant_sources=args.relevant_source,
            relevant_doc_ids=args.relevant_doc_id,
            relevant_chunk_ids=args.relevant_chunk_id,
            top_k_values=args.top_k_values,
        )
    raise ValueError(f"未知命令: {args.command}")


def print_result(result: dict[str, Any], *, as_json: bool = False) -> None:
    """打印 entrypoint 执行结果。

    Example Input:
        print_result({"module": "query", "answer": "ok"}, as_json=True)

    Example Output:
        None
    """
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    module = result.get("module")
    if module == "index":
        _print_index_result(result)
    elif module == "query":
        _print_query_result(result)
    elif module == "evaluation":
        _print_evaluation_result(result)
    elif module == "run":
        _print_run_result(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_metadata_filter(value: str | None) -> dict[str, Any] | None:
    """解析 metadata filter JSON。

    Example Input:
        parse_metadata_filter('{"acl": "public"}')

    Example Output:
        {"acl": "public"}
    """
    if value in (None, ""):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--metadata-filter 必须是 JSON 对象") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--metadata-filter 必须是 JSON 对象")
    return parsed


def parse_top_k_values(value: str) -> tuple[int, ...]:
    """解析逗号分隔的评估 K 值。

    Example Input:
        parse_top_k_values("1,3,5")

    Example Output:
        (1, 3, 5)
    """
    try:
        values = tuple(
            int(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--top-k-values 必须是整数列表") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--top-k-values 至少包含一个正整数")
    return tuple(sorted(set(values)))


def _add_index_command(subparsers: argparse._SubParsersAction) -> None:
    """注册 index 子命令。

    Example Input:
        _add_index_command(subparsers)

    Example Output:
        None
    """
    parser = subparsers.add_parser("index", help="单独运行索引构建流程")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="知识源目录或单文件路径，默认 ragx/data/resources。",
    )
    parser.add_argument("--reset", action="store_true", help="索引前清理旧库。")
    parser.add_argument(
        "--acl",
        action="append",
        default=None,
        help="写入 chunk metadata 的 ACL，可重复传入。",
    )
    _add_store_args(parser)
    _add_json_arg(parser)


def _add_query_command(subparsers: argparse._SubParsersAction) -> None:
    """注册 query 子命令。

    Example Input:
        _add_query_command(subparsers)

    Example Output:
        None
    """
    parser = subparsers.add_parser("query", help="单独运行查询处理流程")
    parser.add_argument("query", help="用户查询文本。")
    parser.add_argument("--top-k", type=int, default=None, help="返回证据片段数量。")
    _add_reranker_arg(parser)
    _add_filter_arg(parser)
    _add_store_args(parser)
    _add_json_arg(parser)


def _add_evaluate_command(subparsers: argparse._SubParsersAction) -> None:
    """注册 evaluate 子命令。

    Example Input:
        _add_evaluate_command(subparsers)

    Example Output:
        None
    """
    parser = subparsers.add_parser("evaluate", help="单独运行效果评估流程")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", default=None, help="单条待评估查询。")
    source.add_argument("--dataset", default=None, help="JSONL 评估集路径。")
    _add_evaluation_label_args(parser)
    _add_reranker_arg(parser)
    _add_filter_arg(parser)
    _add_store_args(parser)
    _add_json_arg(parser)


def _add_run_command(subparsers: argparse._SubParsersAction) -> None:
    """注册 run 子命令。

    Example Input:
        _add_run_command(subparsers)

    Example Output:
        None
    """
    parser = subparsers.add_parser("run", help="组合运行索引、查询和可选评估")
    parser.add_argument("query", help="用户查询文本。")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="知识源目录或单文件路径，默认 ragx/data/resources。",
    )
    parser.add_argument("--reset", action="store_true", help="索引前清理旧库。")
    parser.add_argument(
        "--acl",
        action="append",
        default=None,
        help="写入 chunk metadata 的 ACL，可重复传入。",
    )
    parser.add_argument("--top-k", type=int, default=None, help="返回证据片段数量。")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="查询后同步输出检索评估指标。",
    )
    _add_evaluation_label_args(parser)
    _add_reranker_arg(parser)
    _add_filter_arg(parser)
    _add_store_args(parser)
    _add_json_arg(parser)


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    """添加 SQLite 与 manifest 路径参数。

    Example Input:
        _add_store_args(parser)

    Example Output:
        None
    """
    parser.add_argument("--sqlite-path", default=None, help="SQLite 索引库路径。")
    parser.add_argument("--manifest-path", default=None, help="增量 manifest 路径。")


def _add_filter_arg(parser: argparse.ArgumentParser) -> None:
    """添加 metadata filter 参数。

    Example Input:
        _add_filter_arg(parser)

    Example Output:
        None
    """
    parser.add_argument(
        "--metadata-filter",
        type=parse_metadata_filter,
        default=None,
        help='metadata 过滤 JSON，例如 {"acl": "public"}。',
    )


def _add_reranker_arg(parser: argparse.ArgumentParser) -> None:
    """添加 reranker 后端参数。

    Example Input:
        _add_reranker_arg(parser)

    Example Output:
        None
    """
    parser.add_argument(
        "--reranker-backend",
        choices=("cross_encoder", "colbert", "none"),
        default=None,
        help="候选重排后端：cross_encoder、colbert 或 none。",
    )


def _add_evaluation_label_args(parser: argparse.ArgumentParser) -> None:
    """添加评估标注参数。

    Example Input:
        _add_evaluation_label_args(parser)

    Example Output:
        None
    """
    parser.add_argument(
        "--relevant-source",
        action="append",
        default=None,
        help="相关 source，可重复传入。",
    )
    parser.add_argument(
        "--relevant-doc-id",
        action="append",
        default=None,
        help="相关 doc_id，可重复传入。",
    )
    parser.add_argument(
        "--relevant-chunk-id",
        action="append",
        default=None,
        help="相关 chunk_id，可重复传入。",
    )
    parser.add_argument(
        "--top-k-values",
        type=parse_top_k_values,
        default=DEFAULT_TOP_K_VALUES,
        help="评估 K 值，逗号分隔，默认 1,3,5。",
    )


def _add_json_arg(parser: argparse.ArgumentParser) -> None:
    """添加 JSON 输出参数。

    Example Input:
        _add_json_arg(parser)

    Example Output:
        None
    """
    parser.add_argument("--json", dest="as_json", action="store_true", help="输出 JSON。")


def _print_index_result(result: dict[str, Any]) -> None:
    """打印索引结果摘要。

    Example Input:
        _print_index_result({"report": {"added": 1}, "chunk_count": 2})

    Example Output:
        None
    """
    report = result["report"]
    storage = result["storage"]
    print(
        "[index] "
        f"added={report['added']} updated={report['updated']} "
        f"skipped={report['skipped']} deleted={report['deleted']} "
        f"reused={report['reused_chunks']} reembedded={report['reembedded_chunks']}",
    )
    print(f"[index] source={result['source']}")
    print(f"[index] chunk_count={result['chunk_count']}")
    print(
        "[index] "
        f"sqlite={storage['sqlite_path']} manifest={storage['manifest_path']}",
    )


def _print_query_result(result: dict[str, Any]) -> None:
    """打印查询结果摘要。

    Example Input:
        _print_query_result({"answer": "ok", "contexts": []})

    Example Output:
        None
    """
    print(f"[query] {result['query']}")
    print(f"\n答> {result['answer']}")
    print(f"\n[query] context_count={result['context_count']}")
    for index, context in enumerate(result["contexts"], 1):
        print(
            f"  {index}. score={context.get('score')} "
            f"source={context.get('source')} page={context.get('page')}",
        )
        preview = str(context.get("preview") or "")
        if preview:
            print(f"     {preview}")


def _print_evaluation_result(result: dict[str, Any]) -> None:
    """打印评估结果摘要。

    Example Input:
        _print_evaluation_result({"summary": {"query_count": 1}, "reports": []})

    Example Output:
        None
    """
    summary = result["summary"]
    print(
        "[evaluate] "
        f"query_count={summary['query_count']} "
        f"labelled={summary['labelled_query_count']} "
        f"avg_mrr={summary['average_mrr']}",
    )
    print(f"[evaluate] hit_rate_at_k={summary['hit_rate_at_k']}")
    print(f"[evaluate] precision_at_k={summary['average_precision_at_k']}")
    print(f"[evaluate] recall_at_k={summary['average_recall_at_k']}")
    for report in result["reports"]:
        print(
            "  "
            f"query={report['query']!r} candidates={report['candidate_count']} "
            f"best_rank={report['best_rank']} mrr={report['mrr']}",
        )


def _print_run_result(result: dict[str, Any]) -> None:
    """打印完整链路结果摘要。

    Example Input:
        _print_run_result({"index": {...}, "query": {...}, "evaluation": None})

    Example Output:
        None
    """
    _print_index_result(result["index"])
    print()
    _print_query_result(result["query"])
    if result.get("evaluation"):
        print()
        _print_evaluation_result(result["evaluation"])


if __name__ == "__main__":
    raise SystemExit(main())
