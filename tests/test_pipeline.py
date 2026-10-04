"""端到端冒烟测试 —— 显式注入离线 Provider 跑通全链路。

运行：python -m pytest tests/  或  python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from rag.app import build_pipeline

DOC = """# RAG

## Provider 抽象层
RAG 的核心流程包括数据接入、结构化分块、embedding、向量检索、prompt 增强和 LLM 生成。
Provider 抽象层通过统一接口切换不同真实厂商。
"""


def _build_test_pipeline(tmp_path):
    data_dir = tmp_path / "data"
    store_dir = data_dir / "stores"
    store_dir.mkdir(parents=True)
    path = data_dir / "rag.md"
    path.write_text(DOC, encoding="utf-8")
    return build_pipeline(
        source=str(data_dir),
        cfg=Settings(
            provider="mock",
            vector_backend="sqlite",
            sqlite_path=str(store_dir / "vectors.db"),
            manifest_path="",
        ),
    )


def test_end_to_end(tmp_path):
    pipeline = _build_test_pipeline(tmp_path)
    result = pipeline.ask("RAG 的核心流程有哪些？")
    assert result["answer"], "应返回非空答案"
    assert result["contexts"], "应检索到上下文片段"
    # 相关片段应命中知识库
    assert any("RAG" in c["preview"] or "Provider" in c["preview"] for c in result["contexts"])


def test_multi_turn_memory(tmp_path):
    pipeline = _build_test_pipeline(tmp_path)
    pipeline.ask("什么是 Provider 抽象层？")
    result = pipeline.ask("它如何切换厂商？")
    assert result["answer"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as data_dir:
        test_end_to_end(Path(data_dir) / "case1")
        test_multi_turn_memory(Path(data_dir) / "case2")
    print("✓ all tests passed")
