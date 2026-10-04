"""独立模块入口层。

本模块为索引构建、查询处理、效果评估和完整链路提供可导入 API。
它只负责配置覆盖、参数校验、结果整形和资源关闭，核心能力仍由
RagApplication 组合根提供。
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from config.settings import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SQLITE_PATH,
    Settings,
    ensure_real_provider,
    load_settings,
    normalize_search_backend,
)
from rag.app import RagApplication
from rag.ingestion.sync import SyncReport
from rag.retrieval.reranker import normalize_reranker_backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "resources"
DEFAULT_TOP_K_VALUES = (1, 3, 5)


def run_index(
    source: str | Path = DEFAULT_SOURCE,
    *,
    sqlite_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    reset: bool = False,
    acl: Sequence[str] | None = None,
    cfg: Settings | None = None,
    providers: object | None = None,
    validate_provider: bool = True,
) -> dict[str, Any]:
    """单独执行索引构建流程。

    Example Input:
        run_index("agent-library/ragx/data/resources", reset=True)

    Example Output:
        {"module": "index", "report": {"added": 1}, "chunk_count": 12}
    """
    settings = prepare_settings(
        cfg,
        sqlite_path=sqlite_path,
        manifest_path=manifest_path,
    )
    if validate_provider:
        ensure_real_provider(settings)
    source_path = _resolve_existing_path(source, "数据源")
    if reset:
        reset_local_index(settings.sqlite_path, settings.manifest_path)

    app = RagApplication(settings, providers=providers)
    try:
        report = app.index(str(source_path), acl=list(acl) if acl else None)
    finally:
        close_application_store(app)

    return {
        "module": "index",
        "source": display_path(source_path),
        "retrieval_backend": "hybrid",
        "search_backend": settings.search_backend,
        "reranker_backend": settings.reranker_backend,
        "report": sync_report_to_dict(report),
        "chunk_count": count_sqlite_chunks(settings.sqlite_path),
        "storage": build_storage_report(settings),
    }


def run_query(
    query: str,
    *,
    sqlite_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    reranker_backend: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    top_k: int | None = None,
    cfg: Settings | None = None,
    providers: object | None = None,
    validate_provider: bool = True,
) -> dict[str, Any]:
    """单独执行查询理解、检索和结果生成流程。

    Example Input:
        run_query("报销材料几天内提交？", top_k=3)

    Example Output:
        {"module": "query", "answer": "...", "context_count": 3}
    """
    normalized_query = _require_text(query, "query")
    settings = prepare_settings(
        cfg,
        sqlite_path=sqlite_path,
        manifest_path=manifest_path,
        reranker_backend=reranker_backend,
        top_k=top_k,
    )
    if validate_provider:
        ensure_real_provider(settings)

    app = RagApplication(settings, providers=providers)
    try:
        result = app.ask(
            normalized_query,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )
    finally:
        close_application_store(app)

    contexts = list(result.get("contexts", []))
    return {
        "module": "query",
        "query": normalized_query,
        "answer": result.get("answer", ""),
        "contexts": contexts,
        "context_count": len(contexts),
        "retrieval_backend": "hybrid",
        "search_backend": settings.search_backend,
        "reranker_backend": settings.reranker_backend,
        "storage": build_storage_report(settings),
    }


def run_evaluation(
    *,
    query: str | None = None,
    dataset_path: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    reranker_backend: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    relevant_sources: Sequence[str] | None = None,
    relevant_doc_ids: Sequence[str] | None = None,
    relevant_chunk_ids: Sequence[str] | None = None,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    cfg: Settings | None = None,
    providers: object | None = None,
    validate_provider: bool = True,
) -> dict[str, Any]:
    """单独执行检索效果评估流程。

    Example Input:
        run_evaluation(query="报销材料", relevant_sources=["rule.md"])

    Example Output:
        {"module": "evaluation", "summary": {"query_count": 1}}
    """
    items = build_evaluation_items(
        query=query,
        dataset_path=dataset_path,
        relevant_sources=relevant_sources,
        relevant_doc_ids=relevant_doc_ids,
        relevant_chunk_ids=relevant_chunk_ids,
    )
    settings = prepare_settings(
        cfg,
        sqlite_path=sqlite_path,
        manifest_path=manifest_path,
        reranker_backend=reranker_backend,
    )
    if validate_provider:
        ensure_real_provider(settings)

    app = RagApplication(settings, providers=providers)
    try:
        reports = [
            app.evaluate_retrieval(
                item["query"],
                metadata_filter=metadata_filter,
                relevant_sources=item["relevant_sources"],
                relevant_doc_ids=item["relevant_doc_ids"],
                relevant_chunk_ids=item["relevant_chunk_ids"],
                top_k_values=tuple(normalize_top_k_values(top_k_values)),
            )
            for item in items
        ]
    finally:
        close_application_store(app)

    return {
        "module": "evaluation",
        "mode": "dataset" if dataset_path else "single",
        "reports": reports,
        "summary": summarize_evaluation_reports(reports),
        "retrieval_backend": "hybrid",
        "search_backend": settings.search_backend,
        "reranker_backend": settings.reranker_backend,
        "storage": build_storage_report(settings),
    }


def run_full_pipeline(
    source: str | Path,
    query: str,
    *,
    sqlite_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    reranker_backend: str | None = None,
    reset: bool = False,
    acl: Sequence[str] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    top_k: int | None = None,
    evaluate: bool = False,
    relevant_sources: Sequence[str] | None = None,
    relevant_doc_ids: Sequence[str] | None = None,
    relevant_chunk_ids: Sequence[str] | None = None,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    cfg: Settings | None = None,
    providers: object | None = None,
    validate_provider: bool = True,
) -> dict[str, Any]:
    """组合执行索引、查询和可选评估流程。

    Example Input:
        run_full_pipeline("data/resources", "RAGX 是什么？", evaluate=True)

    Example Output:
        {"module": "run", "index": {...}, "query": {...}, "evaluation": {...}}
    """
    normalized_query = _require_text(query, "query")
    settings = prepare_settings(
        cfg,
        sqlite_path=sqlite_path,
        manifest_path=manifest_path,
        reranker_backend=reranker_backend,
        top_k=top_k,
    )
    if validate_provider:
        ensure_real_provider(settings)
    source_path = _resolve_existing_path(source, "数据源")
    if reset:
        reset_local_index(settings.sqlite_path, settings.manifest_path)

    app = RagApplication(settings, providers=providers)
    try:
        index_report = app.index(str(source_path), acl=list(acl) if acl else None)
        query_result = app.ask(
            normalized_query,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )
        evaluation_report = None
        if evaluate:
            evaluation_report = app.evaluate_retrieval(
                normalized_query,
                metadata_filter=metadata_filter,
                relevant_sources=relevant_sources or (),
                relevant_doc_ids=relevant_doc_ids or (),
                relevant_chunk_ids=relevant_chunk_ids or (),
                top_k_values=tuple(normalize_top_k_values(top_k_values)),
            )
    finally:
        close_application_store(app)

    query_contexts = list(query_result.get("contexts", []))
    evaluation_payload = None
    if evaluation_report is not None:
        reports = [evaluation_report]
        evaluation_payload = {
            "module": "evaluation",
            "mode": "single",
            "reports": reports,
            "summary": summarize_evaluation_reports(reports),
            "retrieval_backend": "hybrid",
            "search_backend": settings.search_backend,
            "reranker_backend": settings.reranker_backend,
            "storage": build_storage_report(settings),
        }

    return {
        "module": "run",
        "index": {
            "module": "index",
            "source": display_path(source_path),
            "retrieval_backend": "hybrid",
            "search_backend": settings.search_backend,
            "reranker_backend": settings.reranker_backend,
            "report": sync_report_to_dict(index_report),
            "chunk_count": count_sqlite_chunks(settings.sqlite_path),
            "storage": build_storage_report(settings),
        },
        "query": {
            "module": "query",
            "query": normalized_query,
            "answer": query_result.get("answer", ""),
            "contexts": query_contexts,
            "context_count": len(query_contexts),
            "retrieval_backend": "hybrid",
            "search_backend": settings.search_backend,
            "reranker_backend": settings.reranker_backend,
            "storage": build_storage_report(settings),
        },
        "evaluation": evaluation_payload,
    }


def prepare_settings(
    cfg: Settings | None = None,
    *,
    sqlite_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    reranker_backend: str | None = None,
    top_k: int | None = None,
) -> Settings:
    """加载配置并应用模块入口覆盖项。

    Example Input:
        prepare_settings(sqlite_path="/tmp/ragx.sqlite", top_k=3)

    Example Output:
        Settings(vector_backend="sqlite", retrieval_backend="hybrid", top_k=3)
    """
    settings = cfg or load_settings()
    settings.vector_backend = "sqlite"
    settings.retrieval_backend = "hybrid"
    settings.search_backend = normalize_search_backend(settings.search_backend)
    settings.reranker_backend = normalize_reranker_backend(
        reranker_backend or settings.reranker_backend,
    )
    settings.sqlite_path = _resolve_output_path(
        sqlite_path,
        settings.sqlite_path,
        DEFAULT_SQLITE_PATH,
    )
    settings.manifest_path = _resolve_output_path(
        manifest_path,
        settings.manifest_path,
        DEFAULT_MANIFEST_PATH,
    )
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        settings.top_k = int(top_k)
    return settings


def build_evaluation_items(
    *,
    query: str | None = None,
    dataset_path: str | Path | None = None,
    relevant_sources: Sequence[str] | None = None,
    relevant_doc_ids: Sequence[str] | None = None,
    relevant_chunk_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """生成评估输入项，支持单 query 或 JSONL 数据集。

    Example Input:
        build_evaluation_items(query="报销", relevant_sources=["rule.md"])

    Example Output:
        [{"query": "报销", "relevant_sources": ["rule.md"]}]
    """
    if dataset_path:
        if query:
            raise ValueError("query 和 dataset_path 不能同时传入")
        return load_evaluation_dataset(dataset_path)
    if not query:
        raise ValueError("必须提供 query 或 dataset_path")
    return [
        {
            "query": _require_text(query, "query"),
            "relevant_sources": list(relevant_sources or ()),
            "relevant_doc_ids": list(relevant_doc_ids or ()),
            "relevant_chunk_ids": list(relevant_chunk_ids or ()),
        }
    ]


def load_evaluation_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 格式评估集。

    Example Input:
        load_evaluation_dataset("eval.jsonl")

    Example Output:
        [{"query": "报销", "relevant_sources": ["rule.md"]}]
    """
    path = _resolve_existing_path(dataset_path, "评估集")
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评估集第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"评估集第 {line_number} 行必须是 JSON 对象")
        raw_query = raw.get("query")
        item = {
            "query": _require_text(raw_query, f"评估集第 {line_number} 行 query"),
            "relevant_sources": _list_field(raw, "relevant_sources", line_number),
            "relevant_doc_ids": _list_field(raw, "relevant_doc_ids", line_number),
            "relevant_chunk_ids": _list_field(raw, "relevant_chunk_ids", line_number),
        }
        items.append(item)
    if not items:
        raise ValueError("评估集为空")
    return items


def summarize_evaluation_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """汇总单条或批量评估结果。

    Example Input:
        summarize_evaluation_reports([{"mrr": 1.0, "labelled": True}])

    Example Output:
        {"query_count": 1, "average_mrr": 1.0}
    """
    report_list = list(reports)
    labelled = [report for report in report_list if report.get("labelled")]
    return {
        "query_count": len(report_list),
        "labelled_query_count": len(labelled),
        "average_candidate_count": _average(
            report.get("candidate_count", 0) for report in report_list
        ),
        "average_mrr": _average(report.get("mrr", 0.0) for report in labelled),
        "hit_rate_at_k": _average_metric_map(labelled, "hit_at_k"),
        "average_precision_at_k": _average_metric_map(labelled, "precision_at_k"),
        "average_recall_at_k": _average_metric_map(labelled, "recall_at_k"),
    }


def normalize_top_k_values(values: Sequence[int]) -> tuple[int, ...]:
    """过滤非法 K 值并去重排序。

    Example Input:
        normalize_top_k_values([5, 1, 0, 5])

    Example Output:
        (1, 5)
    """
    normalized = tuple(sorted({int(value) for value in values if int(value) > 0}))
    if not normalized:
        raise ValueError("top_k_values 至少需要一个正整数")
    return normalized


def reset_local_index(sqlite_path: str, manifest_path: str) -> list[str]:
    """删除本地 SQLite 索引和 manifest 文件。

    Example Input:
        reset_local_index("store.sqlite", "manifest.json")

    Example Output:
        ["agent-library/ragx/data/stores/ragx-store.sqlite"]
    """
    deleted: list[str] = []
    for candidate in (
        Path(sqlite_path),
        Path(f"{sqlite_path}-wal"),
        Path(f"{sqlite_path}-shm"),
        Path(manifest_path) if manifest_path else None,
    ):
        if candidate is None:
            continue
        path = candidate.expanduser().resolve()
        if path.exists():
            path.unlink()
            deleted.append(display_path(path))
    return deleted


def count_sqlite_chunks(sqlite_path: str) -> int:
    """统计 SQLite chunks 表中的记录数量。

    Example Input:
        count_sqlite_chunks("agent-library/ragx/data/stores/ragx-store.sqlite")

    Example Output:
        42
    """
    path = Path(sqlite_path).expanduser().resolve()
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        try:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row[0]) if row else 0


def build_storage_report(settings: Settings) -> dict[str, Any]:
    """生成索引存储状态报告。

    Example Input:
        build_storage_report(Settings(sqlite_path="/tmp/store.sqlite"))

    Example Output:
        {"sqlite_path": "/tmp/store.sqlite", "store_exists": true}
    """
    sqlite_path = Path(settings.sqlite_path).expanduser().resolve()
    manifest_path = Path(settings.manifest_path).expanduser().resolve()
    return {
        "sqlite_path": display_path(sqlite_path),
        "manifest_path": display_path(manifest_path),
        "store_exists": sqlite_path.exists(),
        "manifest_exists": manifest_path.exists(),
    }


def sync_report_to_dict(report: SyncReport) -> dict[str, Any]:
    """把 SyncReport 转为 JSON 可序列化字典。

    Example Input:
        sync_report_to_dict(SyncReport(added=1))

    Example Output:
        {"added": 1, "updated": 0}
    """
    return {
        "added": report.added,
        "updated": report.updated,
        "deleted": report.deleted,
        "skipped": report.skipped,
        "reembedded_chunks": report.reembedded_chunks,
        "reused_chunks": report.reused_chunks,
        "details": report.details,
    }


def close_application_store(app: object) -> None:
    """关闭 RagApplication 持有的底层 store。

    Example Input:
        close_application_store(app)

    Example Output:
        None
    """
    store = getattr(app, "_store", None)
    close = getattr(store, "close", None)
    if callable(close):
        close()


def display_path(path: str | Path) -> str:
    """把项目内绝对路径显示为 agent-library 相对路径。

    Example Input:
        display_path(PROJECT_ROOT / "data")

    Example Output:
        "agent-library/ragx/data"
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(AGENTS_ROOT))
    except ValueError:
        return str(resolved)


def _resolve_output_path(
    override: str | Path | None,
    configured: str | Path | None,
    default: str | Path,
) -> str:
    value = override or configured or default
    return str(Path(value).expanduser().resolve())


def _resolve_existing_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label}不存在: {path}")
    return resolved


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空")
    return value.strip()


def _list_field(raw: dict[str, Any], key: str, line_number: int) -> list[str]:
    value = raw.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"评估集第 {line_number} 行 {key} 必须是列表")
    return [str(item) for item in value]


def _average(values: Iterable[object]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 6)


def _average_metric_map(
    reports: Sequence[dict[str, Any]],
    metric_key: str,
) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for report in reports:
        metric = report.get(metric_key, {})
        if not isinstance(metric, dict):
            continue
        for key, value in metric.items():
            buckets.setdefault(int(key), []).append(float(value))
    return {key: _average(values) for key, values in sorted(buckets.items())}
