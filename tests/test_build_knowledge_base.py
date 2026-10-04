"""统一知识库创建脚本测试。"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from rag import app as rag_app
from scripts import build_knowledge_base as builder

DOC = (
    "# RAGX 知识库\n\n"
    "RAGX 会把 data/resources 中的真实 Markdown、PDF 和 DOCX 资源解析为结构化 chunk，"
    "再写入 SQLite 知识库用于检索增强生成流程。\n\n"
    "## 建库流程\n"
    "读取资源、结构感知切分、生成 embedding、写入 chunks 表，并保存 manifest 支持增量更新。\n"
)


class _FakeEmbeddingProvider:
    """测试用 embedding provider。

    输入 (Input):
        texts: 待编码文本列表。

    输出 (Output):
        固定维度向量列表，避免测试依赖外部模型服务。

    示例 (Example):
        _FakeEmbeddingProvider().embed(["hello"])
    """

    def embed(self, texts):
        """把文本编码为确定性向量。

        输入 (Input):
            texts: 文本列表。

        输出 (Output):
            每个文本对应一个 4 维向量。

        示例 (Example):
            _FakeEmbeddingProvider().embed(["rag"])
        """
        vectors = []
        for text in texts:
            base = float(len(text) or 1)
            vectors.append([base, base % 7, base % 11, 1.0])
        return vectors


class _FakeLLMProvider:
    """测试用 LLM provider。

    输入 (Input):
        messages: 对话消息列表。

    输出 (Output):
        固定回答文本，建库测试不会依赖生成质量。

    示例 (Example):
        _FakeLLMProvider().chat([])
    """

    def chat(self, messages, **kwargs):
        """返回固定测试回答。

        输入 (Input):
            messages: 对话消息列表。
            kwargs: 生成参数。

        输出 (Output):
            固定字符串。

        示例 (Example):
            _FakeLLMProvider().chat([])
        """
        return "ok"


def _patch_providers(monkeypatch):
    """注入本地 fake provider 构造函数。

    输入 (Input):
        monkeypatch: pytest monkeypatch fixture。

    输出 (Output):
        None。

    示例 (Example):
        _patch_providers(monkeypatch)
    """
    monkeypatch.setattr(rag_app, "build_embedding_provider", lambda cfg: _FakeEmbeddingProvider())
    monkeypatch.setattr(rag_app, "build_llm_provider", lambda cfg: _FakeLLMProvider())


def _settings(tmp_path):
    """生成测试用 Settings。

    输入 (Input):
        tmp_path: pytest 临时目录。

    输出 (Output):
        使用 openai 配置和 SQLite 后端的 Settings；provider 构造在测试中注入 fake。

    示例 (Example):
        _settings(tmp_path)
    """
    return Settings(
        provider="openai",
        api_key="sk-test",
        vector_backend="sqlite",
        sqlite_path=str(tmp_path / "store" / "ragx-store.sqlite"),
        manifest_path=str(tmp_path / "store" / "ragx-manifest.json"),
        chunk_size=80,
        chunk_overlap=10,
    )


def _write_resource(tmp_path, name="guide.md", text=DOC):
    """写入测试资源文件。

    输入 (Input):
        tmp_path: pytest 临时目录。
        name: 文件名。
        text: 文件内容。

    输出 (Output):
        资源目录路径。

    示例 (Example):
        _write_resource(tmp_path, "guide.md", DOC)
    """
    source = tmp_path / "resources"
    source.mkdir()
    (source / name).write_text(text, encoding="utf-8")
    return source


def _count_rows(sqlite_path):
    """统计 SQLite chunks 表记录数。

    输入 (Input):
        sqlite_path: SQLite 数据库路径。

    输出 (Output):
        chunks 表中的记录数量。

    示例 (Example):
        _count_rows("/tmp/ragx-store.sqlite")
    """
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM chunks")
        return int(cur.fetchone()[0])


def test_build_knowledge_base_writes_sqlite_and_exports_chunks(monkeypatch, tmp_path):
    """验证真实资源可统一切分、写入 SQLite 并导出 JSON。"""
    source = _write_resource(tmp_path)
    cfg = _settings(tmp_path)
    export_path = tmp_path / "chunks.json"
    _patch_providers(monkeypatch)
    monkeypatch.setattr(builder, "load_settings", lambda: cfg)

    args = builder.parse_args(
        [
            "--source",
            str(source),
            "--export-chunks",
            str(export_path),
        ],
    )
    result = builder.build_knowledge_base(args)

    assert result["report"]["added"] == 1
    assert result["chunk_count"] > 0
    assert _count_rows(cfg.sqlite_path) == result["chunk_count"]
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["summary"]["document_count"] == 1
    assert payload["summary"]["total_chunk_count"] == result["chunk_count"]
    assert payload["documents"][0]["chunks"][0]["content"]


def test_build_knowledge_base_uses_hybrid_for_responses_endpoint(monkeypatch, tmp_path):
    """验证 responses endpoint 不触发 RAGX 检索后端降级。"""
    source = _write_resource(tmp_path)
    cfg = _settings(tmp_path)
    cfg.provider = "openai"
    cfg.endpoint = "https://example.test/v1/responses"
    cfg.retrieval_backend = "hybrid"
    _patch_providers(monkeypatch)
    monkeypatch.setattr(builder, "load_settings", lambda: cfg)

    args = builder.parse_args(["--source", str(source)])
    result = builder.build_knowledge_base(args)

    assert cfg.retrieval_backend == "hybrid"
    assert result["retrieval_backend"] == "hybrid"
    assert result["report"]["added"] == 1
    assert result["chunk_count"] > 0
    assert _count_rows(cfg.sqlite_path) == result["chunk_count"]
    assert os.path.exists(cfg.manifest_path)


def test_build_knowledge_base_skips_unchanged_documents(monkeypatch, tmp_path):
    """验证重复建库会复用 manifest 并跳过未变化资源。"""
    source = _write_resource(tmp_path)
    cfg = _settings(tmp_path)
    _patch_providers(monkeypatch)
    monkeypatch.setattr(builder, "load_settings", lambda: cfg)
    args = builder.parse_args(["--source", str(source)])

    first = builder.build_knowledge_base(args)
    second = builder.build_knowledge_base(args)

    assert first["report"]["added"] == 1
    assert second["report"]["skipped"] == 1
    assert second["report"]["reembedded_chunks"] == 0
    assert second["chunk_count"] == first["chunk_count"]


def test_build_knowledge_base_reset_rebuilds_store(monkeypatch, tmp_path):
    """验证 reset 会清理旧库并触发全量重建。"""
    source = _write_resource(tmp_path)
    cfg = _settings(tmp_path)
    _patch_providers(monkeypatch)
    monkeypatch.setattr(builder, "load_settings", lambda: cfg)
    args = builder.parse_args(["--source", str(source)])
    reset_args = builder.parse_args(["--source", str(source), "--reset"])

    first = builder.build_knowledge_base(args)
    rebuilt = builder.build_knowledge_base(reset_args)

    assert first["chunk_count"] > 0
    assert rebuilt["report"]["added"] == 1
    assert rebuilt["report"]["skipped"] == 0
    assert rebuilt["chunk_count"] == first["chunk_count"]


def test_build_knowledge_base_rejects_test_provider_by_default(monkeypatch, tmp_path):
    """验证生产默认不允许测试 provider。"""
    source = _write_resource(tmp_path)
    cfg = _settings(tmp_path)
    cfg.provider = "mock"
    monkeypatch.setattr(builder, "load_settings", lambda: cfg)
    args = builder.parse_args(["--source", str(source)])

    with pytest.raises(ValueError, match="生产运行禁止使用测试 provider"):
        builder.build_knowledge_base(args)


def test_build_knowledge_base_missing_source_raises(monkeypatch, tmp_path):
    """验证不存在的 source 会返回清晰错误。"""
    monkeypatch.setattr(builder, "load_settings", lambda: _settings(tmp_path))
    args = builder.parse_args(["--source", str(tmp_path / "missing")])

    with pytest.raises(FileNotFoundError, match="数据源不存在"):
        builder.build_knowledge_base(args)


def test_apply_cli_overrides_fills_default_paths(monkeypatch):
    """验证未显式传路径且配置为空时会回填默认 SQLite 与 manifest 路径。"""
    cfg = Settings(sqlite_path="", manifest_path="", vector_backend="pgvector")
    monkeypatch.setattr(builder, "load_settings", lambda: cfg)

    result = builder.apply_cli_overrides(builder.parse_args([]))

    assert result is cfg
    assert result.vector_backend == "sqlite"
    assert result.retrieval_backend == "hybrid"
    assert result.sqlite_path == str(builder.DEFAULT_SQLITE_PATH)
    assert result.manifest_path == str(builder.DEFAULT_MANIFEST_PATH)


def test_count_sqlite_chunks_handles_missing_db_and_missing_table(tmp_path):
    """验证 chunk 统计对不存在数据库和无 chunks 表数据库安全返回 0。"""
    missing = tmp_path / "missing.sqlite"
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()

    assert builder.count_sqlite_chunks(str(missing)) == 0
    assert builder.count_sqlite_chunks(str(empty)) == 0


def test_print_summary_includes_optional_export_path(capsys):
    """验证建库摘要打印完整指标，并在存在导出文件时输出 chunks_json。"""
    builder.print_summary(
        {
            "report": {
                "added": 1,
                "updated": 2,
                "skipped": 3,
                "deleted": 4,
                "reused_chunks": 5,
                "reembedded_chunks": 6,
            },
            "retrieval_backend": "hybrid",
            "source": "agent-library/ragx/data/resources",
            "sqlite_path": "agent-library/ragx/data/stores/ragx-store.sqlite",
            "manifest_path": "agent-library/ragx/data/stores/ragx-manifest.json",
            "chunk_count": 7,
            "export_path": "agent-library/ragx/south-chunks.json",
        }
    )

    output = capsys.readouterr().out
    assert "新增 1 / 更新 2 / 跳过 3" in output
    assert "backend=hybrid" in output
    assert "chunk_count=7" in output
    assert "chunks_json=agent-library/ragx/south-chunks.json" in output


def test_main_invokes_build_and_prints_summary(monkeypatch, capsys):
    """验证脚本 main 编排 parse_args、build_knowledge_base 与 print_summary。"""

    def fake_build(args):
        assert args.source == "source-dir"
        return {
            "report": {
                "added": 0,
                "updated": 0,
                "skipped": 1,
                "deleted": 0,
                "reused_chunks": 2,
                "reembedded_chunks": 0,
            },
            "retrieval_backend": "hybrid",
            "source": "source-dir",
            "sqlite_path": "store.sqlite",
            "manifest_path": "manifest.json",
            "chunk_count": 2,
            "export_path": "",
        }

    monkeypatch.setattr(builder, "build_knowledge_base", fake_build)

    exit_code = builder.main(["--source", "source-dir"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "跳过 1" in output
    assert "chunk_count=2" in output


def test_main_reports_provider_error_without_traceback(monkeypatch, capsys):
    """验证建库 CLI 对 provider 连通性错误输出清晰提示。

    输入 (Input):
        main(["--source", "data/resources"])

    输出 (Output):
        返回 1，stderr 包含 provider 配置排查提示。
    """

    def fail_build_knowledge_base(args):
        """模拟真实 provider 返回 404。

        输入 (Input):
            args: parse_args 返回的命名空间。

        输出 (Output):
            抛出 RuntimeError。
        """
        raise RuntimeError("404 page not found")

    monkeypatch.setattr(builder, "build_knowledge_base", fail_build_knowledge_base)

    exit_code = builder.main(["--source", "data/resources"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "[知识库] failed: RuntimeError: 404 page not found" in captured.err
    assert "OPENAI_ENDPOINT" in captured.err
