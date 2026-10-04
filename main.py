"""命令行入口 —— 演示完整生产 RAG 链路。

链路：版面分析 → 结构感知分块 → 增量同步建库 → metadata 过滤检索 → 生成。

用法：
    OPENAI_API_KEY=sk-xxx python main.py
    python main.py "你的问题"  # 单次提问
依赖：rag.app（组合根 RagApplication）。
"""
from __future__ import annotations

import os
import sys

from config.settings import ensure_real_provider, load_settings
from rag.app import RagApplication

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main() -> None:
    cfg = load_settings()
    ensure_real_provider(cfg)
    app = RagApplication(cfg)
    report = app.index(DATA_DIR)  # 可重复调用：未变跳过、变更 delete-then-insert
    print(f"[索引] 新增 {report.added} 文档 / 更新 {report.updated} / "
          f"跳过 {report.skipped} / 删除 chunk {report.deleted} / "
          f"复用块 {report.reused_chunks} / 重嵌入块 {report.reembedded_chunks}")

    if len(sys.argv) > 1:
        _print(app.ask(" ".join(sys.argv[1:])))
        return

    print("RAG 交互问答（输入 exit 退出）")
    while True:
        try:
            query = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            _print(app.ask(query))


def _print(result: dict) -> None:
    print("\n答> " + result["answer"])
    print("\n— 命中片段（含溯源）—")
    for c in result["contexts"]:
        print(f"  · score={c['score']} v{c['version']} "
              f"src={c['source']} page={c['page']} path={c['heading_path']!r}")
        print(f"    {c['preview']}")


if __name__ == "__main__":
    main()
