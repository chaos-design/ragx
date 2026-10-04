"""RAGX 模块化入口测试。"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ragx_cli  # noqa: E402
from config.settings import Settings  # noqa: E402
from rag import entrypoints  # noqa: E402


DOC = (
    "# RAGX 模块化入口\n\n"
    "RAGX 支持索引构建、查询处理和效果评估三个独立模块。"
    "索引模块负责创建 SQLite 存储并保存 manifest。"
    "查询模块负责检索证据并生成回答。"
    "评估模块负责计算 MRR、Hit@K、Precision@K 和 Recall@K。\n\n"
    "## 验证方法\n"
    "每个模块都可以单独调用，也可以通过 run 命令组合执行完整流程。\n"
)


def _settings(tmp_path):
    """生成测试用 Settings。

    Example Input:
        _settings(tmp_path)

    Example Output:
        Settings(provider="openai", sqlite_path="/tmp/.../store.sqlite")
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


def _source(tmp_path, text=DOC):
    """写入测试知识源。

    Example Input:
        _source(tmp_path)

    Example Output:
        Path("/tmp/.../resources")
    """
    source = tmp_path / "resources"
    source.mkdir()
    (source / "guide.md").write_text(text, encoding="utf-8")
    return source


def test_run_index_creates_store_and_reports_loadable_chunks(tmp_path):
    """验证索引入口可独立创建、存储并加载 chunk 统计。"""
    cfg = _settings(tmp_path)
    result = entrypoints.run_index(_source(tmp_path), cfg=cfg, reset=True)

    assert result["module"] == "index"
    assert result["report"]["added"] == 1
    assert result["chunk_count"] > 0
    assert result["storage"]["store_exists"] is True
    assert result["storage"]["manifest_exists"] is True
    assert "agent-library" not in result["storage"]["sqlite_path"] or (
        result["storage"]["sqlite_path"].startswith("agent-library/")
    )
    assert entrypoints.count_sqlite_chunks(cfg.sqlite_path) == result["chunk_count"]


def test_run_query_loads_existing_store_and_returns_contexts(tmp_path):
    """验证查询入口可独立加载已有索引并生成回答。"""
    cfg = _settings(tmp_path)
    source = _source(tmp_path)
    entrypoints.run_index(source, cfg=cfg, reset=True)

    result = entrypoints.run_query(
        "RAGX 如何验证索引和查询模块？",
        cfg=cfg,
        reranker_backend="colbert",
        top_k=2,
    )

    assert result["module"] == "query"
    assert result["query"] == "RAGX 如何验证索引和查询模块？"
    assert result["search_backend"] == "bi_encoder"
    assert result["reranker_backend"] == "colbert"
    assert result["answer"].startswith("(test)")
    assert result["context_count"] <= 2
    assert result["contexts"]
    assert result["contexts"][0]["source"] == "guide.md"


def test_run_evaluation_single_query_returns_labelled_metrics(tmp_path):
    """验证评估入口可单独计算带标注的检索指标。"""
    cfg = _settings(tmp_path)
    entrypoints.run_index(_source(tmp_path), cfg=cfg, reset=True)

    result = entrypoints.run_evaluation(
        query="RAGX 评估模块计算哪些指标？",
        relevant_sources=["guide.md"],
        top_k_values=(1, 3),
        cfg=cfg,
    )

    report = result["reports"][0]
    assert result["module"] == "evaluation"
    assert result["mode"] == "single"
    assert result["summary"]["query_count"] == 1
    assert result["summary"]["labelled_query_count"] == 1
    assert report["labelled"] is True
    assert report["mrr"] > 0
    assert set(report["hit_at_k"]) == {1, 3}


def test_run_evaluation_dataset_and_invalid_dataset_edges(tmp_path):
    """验证 JSONL 评估集正例、空集和字段类型反例。"""
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        "\n# comment\n"
        + json.dumps(
            {
                "query": "索引模块负责什么？",
                "relevant_sources": ["guide.md"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    invalid_dataset = tmp_path / "invalid.jsonl"
    invalid_dataset.write_text(
        json.dumps({"query": "bad", "relevant_sources": "guide.md"}),
        encoding="utf-8",
    )
    empty_dataset = tmp_path / "empty.jsonl"
    empty_dataset.write_text("# only comment\n", encoding="utf-8")

    items = entrypoints.load_evaluation_dataset(dataset)

    assert items == [
        {
            "query": "索引模块负责什么？",
            "relevant_sources": ["guide.md"],
            "relevant_doc_ids": [],
            "relevant_chunk_ids": [],
        }
    ]
    with pytest.raises(ValueError, match="relevant_sources 必须是列表"):
        entrypoints.load_evaluation_dataset(invalid_dataset)
    with pytest.raises(ValueError, match="评估集为空"):
        entrypoints.load_evaluation_dataset(empty_dataset)


def test_run_full_pipeline_combines_index_query_and_evaluation(tmp_path):
    """验证组合入口在同一应用实例中串联三个模块。"""
    cfg = _settings(tmp_path)
    result = entrypoints.run_full_pipeline(
        _source(tmp_path),
        "完整流程如何运行？",
        cfg=cfg,
        reset=True,
        evaluate=True,
        relevant_sources=["guide.md"],
        top_k=1,
    )

    assert result["module"] == "run"
    assert result["index"]["report"]["added"] == 1
    assert result["query"]["context_count"] == 1
    assert result["query"]["answer"].startswith("(test)")
    assert result["evaluation"]["summary"]["query_count"] == 1


def test_entrypoint_validation_boundaries(tmp_path):
    """验证入口层参数边界和互斥约束。"""
    cfg = _settings(tmp_path)

    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        entrypoints.prepare_settings(cfg, top_k=0)
    with pytest.raises(ValueError, match="Unsupported reranker backend"):
        entrypoints.prepare_settings(cfg, reranker_backend="bad")
    cfg.search_backend = "colbert"
    with pytest.raises(ValueError, match="Unsupported search backend"):
        entrypoints.prepare_settings(cfg)
    cfg.search_backend = "bi_encoder"
    with pytest.raises(ValueError, match="必须提供 query 或 dataset_path"):
        entrypoints.build_evaluation_items()
    with pytest.raises(ValueError, match="不能同时传入"):
        entrypoints.build_evaluation_items(query="q", dataset_path=tmp_path)
    with pytest.raises(FileNotFoundError, match="数据源不存在"):
        entrypoints.run_index(tmp_path / "missing", cfg=cfg)
    with pytest.raises(ValueError, match="不能为空"):
        entrypoints.run_query("   ", cfg=cfg)
    with pytest.raises(ValueError, match="至少需要一个正整数"):
        entrypoints.normalize_top_k_values([0, -1])


def test_cli_dispatch_query_passes_arguments(monkeypatch):
    """验证 CLI query 子命令参数能正确传递到 API 层。"""
    captured = {}

    def fake_run_query(query, **kwargs):
        """记录 CLI 分派参数。

        Example Input:
            fake_run_query("hello", top_k=2)

        Example Output:
            {"module": "query"}
        """
        captured["query"] = query
        captured.update(kwargs)
        return {"module": "query", "query": query, "answer": "ok", "contexts": []}

    monkeypatch.setattr(ragx_cli, "run_query", fake_run_query)
    args = ragx_cli.parse_args(
        [
            "query",
            "hello",
            "--metadata-filter",
            '{"source": "guide.md"}',
            "--top-k",
            "2",
            "--reranker-backend",
            "colbert",
            "--sqlite-path",
            "store.sqlite",
        ],
    )

    result = ragx_cli.dispatch(args)

    assert result["module"] == "query"
    assert captured["query"] == "hello"
    assert captured["metadata_filter"] == {"source": "guide.md"}
    assert captured["top_k"] == 2
    assert captured["reranker_backend"] == "colbert"
    assert captured["sqlite_path"] == "store.sqlite"


def test_cli_main_json_output_and_error_path(monkeypatch, capsys):
    """验证 CLI JSON 输出与异常转换。"""

    def fake_run_index(*args, **kwargs):
        """返回稳定索引报告。

        Example Input:
            fake_run_index("source")

        Example Output:
            {"module": "index", "chunk_count": 1}
        """
        return {
            "module": "index",
            "source": "source",
            "report": {
                "added": 1,
                "updated": 0,
                "skipped": 0,
                "deleted": 0,
                "reused_chunks": 0,
                "reembedded_chunks": 1,
            },
            "chunk_count": 1,
            "storage": {
                "sqlite_path": "store.sqlite",
                "manifest_path": "manifest.json",
            },
        }

    monkeypatch.setattr(ragx_cli, "run_index", fake_run_index)

    assert ragx_cli.main(["index", "--source", "source", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["module"] == "index"
    assert output["chunk_count"] == 1

    def fail_run_index(*args, **kwargs):
        """模拟入口异常。

        Example Input:
            fail_run_index("source")

        Example Output:
            raises RuntimeError
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(ragx_cli, "run_index", fail_run_index)
    assert ragx_cli.main(["index", "--source", "source"]) == 1
    assert "RuntimeError: boom" in capsys.readouterr().err


def test_cli_evaluate_and_run_dispatch(monkeypatch):
    """验证 evaluate 与 run 子命令分派。"""
    calls = []

    def fake_run_evaluation(**kwargs):
        """记录 evaluate 参数。

        Example Input:
            fake_run_evaluation(query="q")

        Example Output:
            {"module": "evaluation"}
        """
        calls.append(("evaluation", kwargs))
        return {"module": "evaluation", "summary": {}, "reports": []}

    def fake_run_full_pipeline(source, query, **kwargs):
        """记录 run 参数。

        Example Input:
            fake_run_full_pipeline("source", "q")

        Example Output:
            {"module": "run"}
        """
        calls.append(("run", {"source": source, "query": query, **kwargs}))
        return {"module": "run", "index": {}, "query": {}, "evaluation": None}

    monkeypatch.setattr(ragx_cli, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(ragx_cli, "run_full_pipeline", fake_run_full_pipeline)

    eval_args = ragx_cli.parse_args(
        [
            "evaluate",
            "--query",
            "q",
            "--relevant-source",
            "guide.md",
            "--top-k-values",
            "3,1,3",
        ],
    )
    run_args = ragx_cli.parse_args(
        [
            "run",
            "q",
            "--source",
            "source",
            "--evaluate",
            "--metadata-filter",
            '{"is_latest": true}',
        ],
    )

    assert ragx_cli.dispatch(eval_args)["module"] == "evaluation"
    assert ragx_cli.dispatch(run_args)["module"] == "run"
    assert calls[0][1]["top_k_values"] == (1, 3)
    assert calls[0][1]["relevant_sources"] == ["guide.md"]
    assert calls[1][1]["source"] == "source"
    assert calls[1][1]["metadata_filter"] == {"is_latest": True}


def test_cli_print_result_human_modes(capsys):
    """验证 CLI 人类可读输出覆盖四类模块。"""
    index = {
        "module": "index",
        "source": "source",
        "report": {
            "added": 1,
            "updated": 2,
            "skipped": 3,
            "deleted": 4,
            "reused_chunks": 5,
            "reembedded_chunks": 6,
        },
        "chunk_count": 7,
        "storage": {
            "sqlite_path": "store.sqlite",
            "manifest_path": "manifest.json",
        },
    }
    query = {
        "module": "query",
        "query": "q",
        "answer": "answer",
        "context_count": 1,
        "contexts": [
            {"score": 0.9, "source": "guide.md", "page": 1, "preview": "chunk"}
        ],
    }
    evaluation = {
        "module": "evaluation",
        "summary": {
            "query_count": 1,
            "labelled_query_count": 1,
            "average_mrr": 1.0,
            "hit_rate_at_k": {1: 1.0},
            "average_precision_at_k": {1: 1.0},
            "average_recall_at_k": {1: 1.0},
        },
        "reports": [
            {
                "query": "q",
                "candidate_count": 1,
                "best_rank": 1,
                "mrr": 1.0,
            }
        ],
    }

    ragx_cli.print_result(index)
    ragx_cli.print_result(query)
    ragx_cli.print_result(evaluation)
    ragx_cli.print_result(
        {"module": "run", "index": index, "query": query, "evaluation": evaluation}
    )
    ragx_cli.print_result({"module": "custom", "ok": True})

    out = capsys.readouterr().out
    assert "[index] added=1" in out
    assert "答> answer" in out
    assert "[evaluate] query_count=1" in out
    assert '"ok": true' in out


def test_cli_parser_validation_edges():
    """验证 CLI 参数解析边界。"""
    assert ragx_cli.parse_metadata_filter(None) is None
    assert ragx_cli.parse_top_k_values("5,1,5") == (1, 5)
    with pytest.raises(SystemExit):
        ragx_cli.parse_args(["evaluate"])
    with pytest.raises(Exception, match="JSON 对象"):
        ragx_cli.parse_metadata_filter("[1]")
    with pytest.raises(Exception, match="正整数"):
        ragx_cli.parse_top_k_values("0,-1")
