"""结构感知分块边界验证：长正文、代码和表格。"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.ingestion.chunker import (
    BlockType,
    ChunkPolicy,
    StructureAwareChunker,
    parse_markdown_blocks,
)


def _meta() -> dict:
    return {"doc_id": "boundary.md", "source": "boundary.md", "version": 1}


def test_text_chunks_split_by_sentence_with_overlap():
    raw = "# Guide\n\n## Section\n第一句很长需要切分。第二句继续补充上下文。第三句用于覆盖滑窗。"
    chunker = StructureAwareChunker(
        ChunkPolicy(text_size=8, text_overlap=4, min_chunk=1)
    )
    chunks = chunker.chunk(raw, _meta())

    assert len(chunks) >= 2
    assert all(chunk.metadata["heading_path"] == "Guide > Section" for chunk in chunks)
    assert all(chunk.metadata["block_type"] == "text" for chunk in chunks)
    print("✓ chunker: long text split with heading context")


def test_code_chunks_split_on_function_boundaries():
    code = "\n".join(  # noqa: FLY002 - 多行字面量列表 join 比巨型 f-string 更易读
        [
            "def first():",
            "    return 'alpha'",
            "def second():",
            "    return 'beta'",
            "def third():",
            "    return 'gamma'",
        ]
    )
    raw = f"# Dev\n\n```python\n{code}\n```"
    chunker = StructureAwareChunker(
        ChunkPolicy(code_size=6, hard_max=5, min_chunk=1)
    )
    chunks = chunker.chunk(raw, _meta())

    assert len(chunks) >= 2
    assert all(chunk.metadata["block_type"] == "code" for chunk in chunks)
    assert any("def first" in chunk.content for chunk in chunks)
    print("✓ chunker: code split on boundaries")


def test_large_table_chunks_repeat_header():
    rows = [
        "| 字段 | 说明 |",
        "| --- | --- |",
        "| provider | agent_provider |",
        "| vector | sqlite |",
        "| data | real documents |",
    ]
    raw = "# Data\n\n" + "\n".join(rows)
    chunker = StructureAwareChunker(
        ChunkPolicy(table_size=10, hard_max=30, min_chunk=1)
    )
    chunks = chunker.chunk(raw, _meta())

    assert len(chunks) >= 2
    assert all("| 字段 | 说明 |" in chunk.content for chunk in chunks)
    assert all(chunk.metadata["block_type"] == "table" for chunk in chunks)
    print("✓ chunker: large table split repeats header")


def test_markdown_image_becomes_searchable_image_chunk(caplog):
    raw = '# Report\n\n![季度 GMV 趋势图](charts/gmv.png "Q4 GMV")\n\n正文说明。'

    caplog.set_level(logging.INFO, logger="rag.ingestion.chunker")
    blocks = parse_markdown_blocks(raw)
    chunks = StructureAwareChunker(ChunkPolicy(min_chunk=1)).chunk(raw, _meta())

    assert any(block.type is BlockType.IMAGE for block in blocks)
    image_chunks = [
        chunk for chunk in chunks if chunk.metadata["block_type"] == "image"
    ]
    assert len(image_chunks) == 1
    assert image_chunks[0].metadata["heading_path"] == "Report"
    assert "季度 GMV 趋势图" in image_chunks[0].content
    assert "Q4 GMV" in image_chunks[0].content
    assert "charts/gmv.png" in image_chunks[0].content
    assert "image chunk generated" in caplog.text
    assert "季度 GMV 趋势图" in caplog.text
    assert "Q4 GMV" in caplog.text
    assert "charts/gmv.png" in caplog.text
    print("✓ chunker: markdown image alt/title/path becomes searchable")


if __name__ == "__main__":
    test_text_chunks_split_by_sentence_with_overlap()
    test_code_chunks_split_on_function_boundaries()
    test_large_table_chunks_repeat_header()
    print("\n✓✓✓ chunker boundary tests passed")
