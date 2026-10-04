"""版面分析层验证：LayoutAnalyzer → Block → chunk_blocks 全链路对接。"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.ingestion.chunker import Block, BlockType, StructureAwareChunker
from rag.ingestion.layout import (
    LayoutAwareChunker,
    MarkdownLayoutAnalyzer,
    ModelLayoutAnalyzer,
    PdfTextLayoutAnalyzer,
    _detect_columns,
    _page_width,
    _PdfTextBlock,
    _sort_pdf_blocks_by_reading_order,
    _valid_bbox,
    analyze_pdf_text_page,
    select_analyzer,
)

DOC = """# 制度

## 报销
出差报销需在 7 天内提交。

| 类型 | 上限 |
| --- | --- |
| 交通 | 2000 |
"""


def _pdf_block(text: str, bbox: tuple[float, float, float, float]) -> _PdfTextBlock:
    return _PdfTextBlock(
        block_type=BlockType.TEXT,
        text=text,
        level=0,
        page=1,
        bbox=bbox,
        max_size=10,
    )


def _pdf_heading(text: str, bbox: tuple[float, float, float, float]) -> _PdfTextBlock:
    return _PdfTextBlock(
        block_type=BlockType.HEADING,
        text=text,
        level=1,
        page=1,
        bbox=bbox,
        max_size=18,
    )


def _raw_pdf_block(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    size: float | None = 10,
    block_type: int = 0,
) -> dict:
    span = {"text": text}
    if size is not None:
        span["size"] = size
    return {
        "type": block_type,
        "bbox": bbox,
        "lines": [{"spans": [span]}],
    }


def test_layout_to_chunk():
    analyzer = MarkdownLayoutAnalyzer()
    blocks = analyzer.analyze(DOC)
    assert any(b.type is BlockType.TABLE for b in blocks)

    chunker = StructureAwareChunker()
    docs = chunker.chunk_blocks(blocks, {"doc_id": "rule.md", "source": "rule.md", "version": 1})
    assert docs, "应产出 chunk"
    # 溯源字段 page/bbox/layout_confidence 已透传进 metadata
    for d in docs:
        assert {"page", "bbox", "layout_confidence", "heading_path"} <= d.metadata.keys()
    print(f"✓ layout→chunk: {len(blocks)} blocks → {len(docs)} chunks")


def test_layout_aware_chunker_invokes_analyzer_before_chunking():
    """验证应用级适配器会显式执行版面分析再分块。"""

    class RecordingAnalyzer(MarkdownLayoutAnalyzer):
        def __init__(self) -> None:
            self.called = False

        def analyze(self, source: str):
            self.called = True
            return super().analyze(source)

    analyzer = RecordingAnalyzer()
    chunker = LayoutAwareChunker(StructureAwareChunker(), analyzer)

    docs = chunker.chunk(
        DOC,
        {"doc_id": "rule.md", "source": "rule.md", "version": 1},
    )

    assert analyzer.called is True
    assert docs
    assert docs[0].metadata["heading_path"] == "制度 > 报销"
    assert docs[0].metadata["layout_confidence"] == 1.0


def test_block_with_coordinates():
    """模拟 PDF 版面检测产出带 page/bbox 的 Block，验证溯源透传。"""
    blocks = [
        Block(BlockType.HEADING, "财报", level=1, page=1, bbox=(10, 10, 200, 40)),
        Block(BlockType.TEXT, "本季度公司营收同比增长百分之二十，主要由海外市场与新产品线共同驱动，"
              "毛利率保持稳定，经营性现金流为正，整体经营质量持续改善。", page=1,
              bbox=(10, 50, 400, 90), confidence=0.96),
    ]
    docs = StructureAwareChunker().chunk_blocks(
        blocks, {"doc_id": "fin.pdf", "source": "fin.pdf", "version": 1})
    d = docs[0]
    assert d.metadata["page"] == 1 and d.metadata["bbox"] == (10, 50, 400, 90)
    assert d.metadata["layout_confidence"] == 0.96
    print(f"✓ coords traceability: page={d.metadata['page']} bbox={d.metadata['bbox']}")


def test_pdf_double_column_reading_order():
    """验证双栏 PDF 会先读完左栏，再进入右栏，避免左右逐行交错。"""
    blocks = [
        _pdf_heading("报告标题", (40, 20, 560, 50)),
        _pdf_block("左栏第一段", (40, 100, 260, 130)),
        _pdf_block("右栏第一段", (330, 100, 550, 130)),
        _pdf_block("左栏第二段", (40, 160, 260, 190)),
        _pdf_block("右栏第二段", (330, 160, 550, 190)),
    ]

    ordered = _sort_pdf_blocks_by_reading_order(blocks, page_width=600)

    assert [block.text for block in ordered] == [
        "报告标题",
        "左栏第一段",
        "左栏第二段",
        "右栏第一段",
        "右栏第二段",
    ]


def test_pdf_spanning_block_splits_column_bands():
    """验证跨栏小标题会切开上下两个分栏阅读区间。"""
    blocks = [
        _pdf_block("上左", (40, 80, 260, 110)),
        _pdf_block("上右", (330, 80, 550, 110)),
        _pdf_heading("跨栏小标题", (40, 160, 560, 190)),
        _pdf_block("下左", (40, 220, 260, 250)),
        _pdf_block("下右", (330, 220, 550, 250)),
    ]

    ordered = _sort_pdf_blocks_by_reading_order(blocks, page_width=600)

    assert [block.text for block in ordered] == [
        "上左",
        "上右",
        "跨栏小标题",
        "下左",
        "下右",
    ]


def test_pdf_sparse_sidebar_does_not_force_columns():
    """验证稀疏侧边栏不足以触发双栏，避免把批注误认为正文列。"""
    blocks = [
        _pdf_block("正文第一段内容较长", (40, 80, 420, 110)),
        _pdf_block("侧注", (470, 90, 540, 110)),
        _pdf_block("正文第二段内容较长", (40, 140, 420, 170)),
        _pdf_block("正文第三段内容较长", (40, 200, 420, 230)),
    ]

    ordered = _sort_pdf_blocks_by_reading_order(blocks, page_width=600)

    assert [block.text for block in ordered] == [
        "正文第一段内容较长",
        "侧注",
        "正文第二段内容较长",
        "正文第三段内容较长",
    ]


def test_pdf_layout_analyzer_reads_fake_fitz(monkeypatch):
    """验证 PdfTextLayoutAnalyzer 会通过 PyMuPDF 页面分析路径产出 Block。"""
    raw = {
        "blocks": [
            _raw_pdf_block("标题", (40, 20, 560, 50), size=18),
            _raw_pdf_block("左一", (40, 100, 260, 130)),
            _raw_pdf_block("右一", (330, 100, 550, 130)),
            _raw_pdf_block("左二", (40, 160, 260, 190)),
            _raw_pdf_block("右二", (330, 160, 550, 190)),
        ]
    }

    class FakePage:
        rect = SimpleNamespace(width=600)

        def get_text(self, mode):
            assert mode == "dict"
            return raw

    class FakeDoc:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([FakePage()])

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(open=lambda source: FakeDoc()),
    )

    blocks = PdfTextLayoutAnalyzer().analyze("double-column.pdf")

    assert [block.text for block in blocks] == ["标题", "左一", "左二", "右一", "右二"]
    assert blocks[0].type is BlockType.HEADING


def test_pdf_page_analysis_filters_invalid_blocks_and_uses_width_fallback():
    """验证页面级分析会过滤图片块、空文本、非法 bbox，并可从 bbox 推导宽度。"""
    raw = {
        "blocks": [
            _raw_pdf_block("图片", (0, 0, 20, 20), block_type=1),
            _raw_pdf_block("非法", (10, 10, 10, 20)),
            _raw_pdf_block("", (20, 20, 40, 40)),
            _raw_pdf_block("正文", (30, 30, 180, 60), size=None),
        ]
    }

    class FakePage:
        def get_text(self, mode):
            assert mode == "dict"
            return raw

    blocks = analyze_pdf_text_page(FakePage(), 3)

    assert _page_width(FakePage(), raw) == 180
    assert [(block.text, block.page, block.bbox) for block in blocks] == [
        ("正文", 3, (30.0, 30.0, 180.0, 60.0)),
    ]


def test_pdf_low_level_layout_boundaries():
    """覆盖 bbox、列检测和 analyzer 选择的低层边界分支。"""
    assert _valid_bbox("bad") is None
    assert _valid_bbox((0, 0, "x", 1)) is None
    assert _valid_bbox((0, 0, float("inf"), 1)) is None
    assert _valid_bbox((0, 0, 0, 1)) is None

    one = _pdf_block("一段", (10, 50, 100, 80))
    assert _sort_pdf_blocks_by_reading_order([one], page_width=600) == [one]

    no_span_blocks = [
        _pdf_block("左一", (40, 80, 260, 110)),
        _pdf_block("右一", (330, 80, 550, 110)),
        _pdf_block("左二", (40, 130, 260, 160)),
        _pdf_block("右二", (330, 130, 550, 160)),
    ]
    assert [block.text for block in _sort_pdf_blocks_by_reading_order(no_span_blocks, 600)] == [
        "左一",
        "左二",
        "右一",
        "右二",
    ]

    too_many_columns = [
        _pdf_block(f"列{i}-{j}", (40 + i * 120, 80 + j * 40, 110 + i * 120, 100 + j * 40))
        for i in range(4)
        for j in range(2)
    ]
    assert _detect_columns(too_many_columns, page_width=600) == ()

    weak_sidebar = [
        _pdf_block("主栏第一段内容较长", (40, 80, 260, 110)),
        _pdf_block("主栏第二段内容较长", (40, 130, 260, 160)),
        _pdf_block("主栏第三段内容较长", (40, 180, 260, 210)),
        _pdf_block("注", (330, 80, 350, 100)),
    ]
    assert _detect_columns(weak_sidebar, page_width=600) == ()
    assert isinstance(select_analyzer("a.pdf", is_path=True), PdfTextLayoutAnalyzer)
    assert isinstance(select_analyzer("# title"), MarkdownLayoutAnalyzer)


def test_model_analyzer_stub():
    try:
        ModelLayoutAnalyzer().analyze("x.pdf")
        assert False, "骨架应提示需接入后端"
    except NotImplementedError:
        print("✓ model analyzer stub raises as expected")


if __name__ == "__main__":
    test_layout_to_chunk()
    test_layout_aware_chunker_invokes_analyzer_before_chunking()
    test_block_with_coordinates()
    test_pdf_double_column_reading_order()
    test_pdf_spanning_block_splits_column_bands()
    test_pdf_sparse_sidebar_does_not_force_columns()
    test_model_analyzer_stub()
    print("\n✓✓✓ layout analysis layer passed")
