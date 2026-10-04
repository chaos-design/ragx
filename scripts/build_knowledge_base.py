"""统一创建 RAGX 知识库。

本脚本把 data/resources 下的真实资源统一解析为 chunks，并通过 RagApplication
写入配置指定的知识库。默认写入 SQLite，必要时可额外导出 chunks JSON 供审计。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SQLITE_PATH,
    ensure_real_provider,
    load_settings,
)
from rag.app import RagApplication
from rag.ingestion.chunker import ChunkPolicy, StructureAwareChunker
from rag.ingestion.layout import (
    LayoutAwareChunker,
    MarkdownLayoutAnalyzer,
)
from rag.ingestion.loader import read_source_documents
from rag.ingestion.sync import SyncReport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "resources"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    输入 (Input):
        argv: 命令行参数列表；None 表示读取 sys.argv。

    输出 (Output):
        argparse.Namespace，包含 source、reset、export_chunks 等参数。

    示例 (Example):
        parse_args(["--source", "./data/resources"])
    """
    parser = argparse.ArgumentParser(
        description="把 ragx/data/resources 的真实资源统一分块并创建 RAG 知识库。",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="原始知识源目录或单文件路径，默认 ragx/data/resources。",
    )
    parser.add_argument(
        "--sqlite-path",
        default="",
        help="SQLite 知识库路径；不填则使用 RAG_SQLITE_PATH 或默认 data/stores/ragx-store.sqlite。",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="增量同步 manifest 路径；不填则使用 RAG_MANIFEST_PATH 或默认 data/stores/ragx-manifest.json。",
    )
    parser.add_argument(
        "--export-chunks",
        default="",
        help="可选：导出真实 chunks JSON 到指定文件。",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="索引前删除 SQLite 库、WAL/SHM 与 manifest，强制全量重建。",
    )
    parser.add_argument(
        "--acl",
        action="append",
        default=None,
        help="写入 chunk metadata 的 ACL，可重复传入；默认 public。",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(args: argparse.Namespace):
    """加载配置并应用 CLI 覆盖项。

    输入 (Input):
        args: parse_args 返回的命名空间。

    输出 (Output):
        Settings 实例，已应用 SQLite 和 manifest 路径覆盖。

    示例 (Example):
        cfg = apply_cli_overrides(parse_args(["--sqlite-path", "/tmp/kb.sqlite"]))
    """
    cfg = load_settings()
    cfg.vector_backend = "sqlite"
    cfg.retrieval_backend = "hybrid"
    if args.sqlite_path:
        cfg.sqlite_path = str(Path(args.sqlite_path).expanduser().resolve())
    elif not cfg.sqlite_path:
        cfg.sqlite_path = str(DEFAULT_SQLITE_PATH)
    if args.manifest_path:
        cfg.manifest_path = str(Path(args.manifest_path).expanduser().resolve())
    elif not cfg.manifest_path:
        cfg.manifest_path = str(DEFAULT_MANIFEST_PATH)
    return cfg


def reset_knowledge_base(sqlite_path: str, manifest_path: str) -> list[str]:
    """删除本地 SQLite 知识库和增量 manifest。

    输入 (Input):
        sqlite_path: SQLite 数据库路径。
        manifest_path: manifest JSON 路径。

    输出 (Output):
        已删除文件路径列表。

    示例 (Example):
        reset_knowledge_base("data/stores/ragx-store.sqlite", "data/stores/ragx-manifest.json")
    """
    deleted: list[str] = []
    candidates = [
        Path(sqlite_path),
        Path(f"{sqlite_path}-wal"),
        Path(f"{sqlite_path}-shm"),
        Path(manifest_path) if manifest_path else None,
    ]
    for path in candidates:
        if path is None:
            continue
        expanded = path.expanduser().resolve()
        if expanded.exists():
            expanded.unlink()
            deleted.append(str(expanded))
    return deleted


def display_path(path: str | Path) -> str:
    """格式化项目内路径，避免报告中出现本机 agents 目录绝对前缀。

    输入 (Input):
        path: 任意文件或目录路径。

    输出 (Output):
        若路径位于 agents 根目录下，返回 `agent-library/...` 形式；否则返回绝对路径。

    示例 (Example):
        display_path(PROJECT_ROOT / "data")
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(AGENTS_ROOT))
    except ValueError:
        return str(resolved)


def build_export_payload(source: str, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    """生成与建库策略一致的真实 chunks 导出数据。

    输入 (Input):
        source: 知识源目录或单文件路径。
        chunk_size: 正文 chunk 目标大小。
        chunk_overlap: 正文 chunk overlap。

    输出 (Output):
        包含 summary 和 documents 的 JSON 可序列化字典。

    示例 (Example):
        build_export_payload("./data/resources", 300, 50)
    """
    sources = read_source_documents(source)
    chunker = LayoutAwareChunker(
        StructureAwareChunker(
            ChunkPolicy(text_size=chunk_size or 512, text_overlap=chunk_overlap or 80),
        ),
        MarkdownLayoutAnalyzer(),
    )
    documents: list[dict[str, Any]] = []
    total_chunks = 0
    for doc_id, raw_text in sources.items():
        chunks = chunker.chunk(raw_text, _build_doc_meta(doc_id))
        total_chunks += len(chunks)
        documents.append(
            {
                "doc_id": doc_id,
                "raw_text_length": len(raw_text),
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    }
                    for chunk in chunks
                ],
            },
        )
    return {
        "summary": {
            "source": display_path(source),
            "document_count": len(documents),
            "total_chunk_count": total_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "documents": documents,
    }


def _build_doc_meta(doc_id: str) -> dict[str, Any]:
    """生成导出 chunks 时使用的文档元数据。

    输入 (Input):
        doc_id: 文档 ID。

    输出 (Output):
        可传入 StructureAwareChunker.chunk 的 metadata。

    示例 (Example):
        _build_doc_meta("rag-sample.md")
    """
    return {
        "doc_id": doc_id,
        "source": doc_id,
        "version": 1,
        "is_latest": True,
        "acl": ["public"],
        "updated_at": 0,
    }


def write_chunks_export(payload: dict[str, Any], output_path: str) -> Path:
    """写出 chunks JSON。

    输入 (Input):
        payload: build_export_payload 返回的数据。
        output_path: 输出文件路径。

    输出 (Output):
        已写入的绝对路径。

    示例 (Example):
        write_chunks_export({"summary": {}}, "/tmp/chunks.json")
    """
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def count_sqlite_chunks(sqlite_path: str) -> int:
    """统计 SQLite 知识库中的 chunk 数。

    输入 (Input):
        sqlite_path: SQLite 数据库路径。

    输出 (Output):
        chunks 表中的记录数；数据库不存在时返回 0。

    示例 (Example):
        count_sqlite_chunks("data/stores/ragx-store.sqlite")
    """
    db_path = Path(sqlite_path).expanduser().resolve()
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        try:
            cur = conn.execute("SELECT COUNT(*) FROM chunks")
        except sqlite3.OperationalError:
            return 0
        return int(cur.fetchone()[0])


def build_knowledge_base(args: argparse.Namespace) -> dict[str, Any]:
    """执行真实资源分块、知识库创建和可选 chunks 导出。

    输入 (Input):
        args: parse_args 返回的命名空间。

    输出 (Output):
        包含 source、report、chunk_count、export_path 的执行报告。

    示例 (Example):
        build_knowledge_base(parse_args([]))
    """
    cfg = apply_cli_overrides(args)
    ensure_real_provider(cfg)

    if args.reset:
        reset_knowledge_base(cfg.sqlite_path, cfg.manifest_path)

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"数据源不存在: {args.source}")
    source = str(source_path)
    app = RagApplication(cfg)
    report = app.index(source, acl=args.acl)
    export_path = ""
    if args.export_chunks:
        payload = build_export_payload(source, cfg.chunk_size, cfg.chunk_overlap)
        export_path = str(write_chunks_export(payload, args.export_chunks))
    close = getattr(app._store, "close", None)
    if callable(close):
        close()

    retrieval_backend = getattr(app, "_retrieval_backend", cfg.retrieval_backend)
    chunk_count = count_sqlite_chunks(cfg.sqlite_path)
    return {
        "source": display_path(source),
        "retrieval_backend": retrieval_backend,
        "sqlite_path": display_path(cfg.sqlite_path),
        "manifest_path": display_path(cfg.manifest_path),
        "report": _report_to_dict(report),
        "chunk_count": chunk_count,
        "export_path": display_path(export_path) if export_path else "",
    }


def _report_to_dict(report: SyncReport) -> dict[str, Any]:
    """把 SyncReport 转换为可打印字典。

    输入 (Input):
        report: IncrementalSyncer.sync 返回的报告。

    输出 (Output):
        可 JSON 序列化的报告字典。

    示例 (Example):
        _report_to_dict(SyncReport(added=1))
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


def print_summary(result: dict[str, Any]) -> None:
    """打印知识库创建结果。

    输入 (Input):
        result: build_knowledge_base 返回的执行报告。

    输出 (Output):
        None。

    示例 (Example):
        print_summary({"report": {"added": 1}, "chunk_count": 3})
    """
    report = result["report"]
    print(
        "[知识库] "
        f"新增 {report['added']} / 更新 {report['updated']} / 跳过 {report['skipped']} / "
        f"删除 chunk {report['deleted']} / 复用块 {report['reused_chunks']} / "
        f"重嵌入块 {report['reembedded_chunks']}",
    )
    print(f"[知识库] backend={result['retrieval_backend']}")
    print(f"[知识库] source={result['source']}")
    print(f"[知识库] sqlite={result['sqlite_path']}")
    print(f"[知识库] manifest={result['manifest_path']}")
    print(f"[知识库] chunk_count={result['chunk_count']}")
    if result["export_path"]:
        print(f"[知识库] chunks_json={result['export_path']}")


def build_failure_hint(error: Exception) -> str:
    """生成建库失败的可执行排查提示。

    输入 (Input):
        error: 建库过程中抛出的异常。

    输出 (Output):
        面向 CLI 用户的排查建议。

    示例 (Example):
        build_failure_hint(RuntimeError("404 page not found"))
    """
    message = str(error).lower()
    error_type = f"{type(error).__module__}.{type(error).__name__}".lower()
    if "openai" in error_type or "provider" in message or "404" in message:
        return (
            "请检查 provider 配置：OPENAI_API_KEY、OPENAI_ENDPOINT 或 "
            "RAG_OPENAI_ENDPOINT、OPENAI_MODEL、embedding endpoint/model "
            "是否与当前服务匹配。"
        )
    if "数据源" in str(error) or "source" in message:
        return "请检查 --source 是否存在且包含可读取的 Markdown、TXT、PDF 或 DOCX 文件。"
    return "请根据错误信息检查配置、源文件和本地存储路径。"


def main(argv: list[str] | None = None) -> int:
    """执行统一知识库创建命令。

    输入 (Input):
        argv: 命令行参数列表；None 表示读取 sys.argv。

    输出 (Output):
        进程退出码，0 表示成功。

    示例 (Example):
        main(["--export-chunks", "/tmp/chunks.json"])
    """
    try:
        result = build_knowledge_base(parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - CLI 边界负责清晰报告失败原因。
        print(f"[知识库] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[知识库] hint: {build_failure_hint(exc)}", file=sys.stderr)
        return 1
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
