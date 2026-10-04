"""版面分析层 (Layout Analysis) —— 统一「页面 → Block 列表」契约。

设计目标：把「版面分析」与「分块」彻底解耦。无论上游是
  - 规则解析(Markdown / 文本型 PDF 坐标聚类)，
  - 视觉检测模型(PP-Structure / LayoutParser)，
  - 多模态文档大模型，
最终都实现同一个 LayoutAnalyzer 接口，归一化产出 Block 列表，
下游 StructureAwareChunker.chunk_blocks() 完全不感知来源。

本文件提供两类实现：
  1. MarkdownLayoutAnalyzer —— 规则版，零依赖，复用 chunker 的 parse_markdown_blocks。
  2. PdfTextLayoutAnalyzer  —— 文本型 PDF 的坐标聚类版(需 pymupdf，缺失时降级)。
  3. ModelLayoutAnalyzer    —— 视觉检测模型适配骨架(标注生产接入点，默认不可用)。

生产适配注意项：
- 选型分流：先判断 PDF 是文本型还是图片型(有无文本层)，分别走规则版/模型版。
- 坐标透传：版面阶段产出的 page/bbox/confidence 必须带进 Block，
  才能在 chunk metadata 里支撑「点击跳转原文」溯源与质检健康度信号。
- 阅读顺序：多栏文档必须先分栏再排序，否则左右栏交错语序全乱。
依赖：rag.ingestion.chunker(Block/BlockType)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

from rag.ingestion.chunker import (
    Block,
    BlockType,
    StructureAwareChunker,
    parse_markdown_blocks,
)
from rag.interfaces import Document

DEFAULT_PDF_HEADING_SIZE_RATIO = 1.2
_COLUMN_GAP_RATIO = 0.035
_COLUMN_MIN_GAP = 18.0
_COLUMN_SPAN_WIDTH_RATIO = 0.62
_COLUMN_MAX_COUNT = 3
_COLUMN_MIN_SUPPORT_RATIO = 0.18
_COLUMN_MIN_TEXT_CHARS = 40


class LayoutAnalyzer(ABC):
    """版面分析统一接口：把一个数据源解析为有序的 Block 列表。"""

    @abstractmethod
    def analyze(self, source: str) -> list[Block]:
        """source 可为文本内容或文件路径，由具体实现约定。"""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 路线 A：规则版(Markdown / 纯文本)                                            #
# --------------------------------------------------------------------------- #
class MarkdownLayoutAnalyzer(LayoutAnalyzer):
    """零依赖规则版：直接复用 Markdown 结构(标题/代码/表格/正文)。"""

    def analyze(self, source: str) -> list[Block]:
        return parse_markdown_blocks(source)


class BlockChunker(Protocol):
    """可消费版面 Block 的分块器协议。"""

    def chunk_blocks(self, blocks: list[Block], doc_meta: dict) -> list[Document]:
        """把版面 Block 列表转换为可索引 Document 列表。"""
        ...


class LayoutAwareChunker:
    """显式串联 LayoutAnalyzer 与结构感知分块器的适配器。

    Example Input:
        chunker = LayoutAwareChunker()
        chunker.chunk("# Title\\nbody", {"doc_id": "a.md", "source": "a.md"})

    Example Output:
        [Document(id="a.md::0", content="...", metadata={...})]
    """

    def __init__(
        self,
        chunker: BlockChunker | None = None,
        analyzer: LayoutAnalyzer | None = None,
    ) -> None:
        """初始化版面感知分块适配器。

        Example Input:
            LayoutAwareChunker(StructureAwareChunker(), MarkdownLayoutAnalyzer())

        Example Output:
            一个先做版面分析、再调用 chunk_blocks 的分块器。
        """
        self._chunker = chunker or StructureAwareChunker()
        self._analyzer = analyzer or MarkdownLayoutAnalyzer()

    def chunk(self, raw_text: str, doc_meta: dict) -> list[Document]:
        """执行版面分析后再做结构感知分块。

        Example Input:
            chunker.chunk("# Guide\\ncontent", {"doc_id": "guide.md"})

        Example Output:
            带 heading_path、page、bbox、layout_confidence metadata 的 chunks。
        """
        return self.chunk_blocks(self._analyzer.analyze(raw_text), doc_meta)

    def chunk_blocks(self, blocks: list[Block], doc_meta: dict) -> list[Document]:
        """直接消费上游已经完成版面分析的 Block 列表。

        Example Input:
            chunker.chunk_blocks([Block(BlockType.TEXT, "content")], doc_meta)

        Example Output:
            [Document(...)]
        """
        return self._chunker.chunk_blocks(blocks, doc_meta)


# --------------------------------------------------------------------------- #
# 路线 A':文本型 PDF —— 字符坐标聚类 + 投影分栏(XY-Cut 思路)                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PdfTextBlock:
    """PDF 文本层中间块，保留字号和 bbox 供阅读顺序排序使用。"""

    block_type: BlockType
    text: str
    level: int
    page: int
    bbox: tuple[float, float, float, float]
    max_size: float

    def to_block(self) -> Block:
        """转换为下游统一消费的 Block。

        Example Input:
            _PdfTextBlock(BlockType.TEXT, "正文", 0, 1, (0, 0, 10, 10), 12)

        Example Output:
            Block(type=BlockType.TEXT, text="正文", page=1, bbox=(0, 0, 10, 10))
        """
        return Block(
            type=self.block_type,
            text=self.text,
            level=self.level,
            page=self.page,
            bbox=self.bbox,
        )


@dataclass(frozen=True)
class _ColumnRegion:
    """页面列区域，基于文本块横向投影聚类得到。"""

    x0: float
    x1: float
    block_count: int
    text_chars: int


class PdfTextLayoutAnalyzer(LayoutAnalyzer):
    """适用于有文本层的 PDF：按 bbox 聚块、检测多栏、字号判标题。

    需要 pymupdf(fitz)。未安装时 analyze 抛 RuntimeError，由上游决定降级。
    双栏处理遵循“先识别列，再按列内 y,x 排序”，跨栏标题或摘要会作为
    分段边界保留，避免左右栏逐行交错。
    """

    def __init__(
        self,
        heading_size_ratio: float = DEFAULT_PDF_HEADING_SIZE_RATIO,
    ) -> None:
        self.heading_size_ratio = heading_size_ratio  # 字号 > 正文中位数*ratio 视为标题

    def analyze(self, source: str) -> list[Block]:
        """读取 PDF 路径并返回稳定阅读顺序的 Block 列表。

        Example Input:
            PdfTextLayoutAnalyzer().analyze("agent-library/ragx/data/a.pdf")

        Example Output:
            [Block(type=BlockType.TEXT, page=1, bbox=(...))]
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:  # pragma: no cover - 取决于运行环境
            raise RuntimeError("PdfTextLayoutAnalyzer 需要 pymupdf，请 pip install pymupdf") from e

        blocks: list[Block] = []
        with fitz.open(source) as doc:
            for page_no, page in enumerate(doc, start=1):
                blocks.extend(
                    analyze_pdf_text_page(
                        page,
                        page_no,
                        heading_size_ratio=self.heading_size_ratio,
                    )
                )
        return blocks


def analyze_pdf_text_page(
    page: Any,
    page_no: int,
    *,
    heading_size_ratio: float = DEFAULT_PDF_HEADING_SIZE_RATIO,
) -> list[Block]:
    """分析单个 PyMuPDF 页面，返回已按阅读顺序排序的 Block。

    Example Input:
        analyze_pdf_text_page(page, 1, heading_size_ratio=1.2)

    Example Output:
        [Block(type=BlockType.HEADING, page=1), Block(type=BlockType.TEXT, page=1)]
    """
    raw = page.get_text("dict")
    page_width = _page_width(page, raw)
    median_size = _median_font_size(raw)
    pdf_blocks = _extract_pdf_text_blocks(
        raw,
        page_no=page_no,
        median_size=median_size,
        heading_size_ratio=heading_size_ratio,
    )
    ordered = _sort_pdf_blocks_by_reading_order(pdf_blocks, page_width)
    return [block.to_block() for block in ordered]


def _median_font_size(raw: dict[str, Any]) -> float:
    """返回页面文本 span 的中位字号。

    Example Input:
        {"blocks": [{"lines": [{"spans": [{"size": 10}, {"size": 12}]}]}]}

    Example Output:
        12.0
    """
    sizes = [
        float(span["size"])
        for block in raw.get("blocks", [])
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if "size" in span
    ]
    if not sizes:
        return 0.0
    ordered = sorted(sizes)
    return ordered[len(ordered) // 2]


def _extract_pdf_text_blocks(
    raw: dict[str, Any],
    *,
    page_no: int,
    median_size: float,
    heading_size_ratio: float,
) -> list[_PdfTextBlock]:
    """从 PyMuPDF dict 输出中抽取有效文本块。

    Example Input:
        _extract_pdf_text_blocks(raw, page_no=1, median_size=10, heading_size_ratio=1.2)

    Example Output:
        [_PdfTextBlock(block_type=BlockType.TEXT, text="正文", ...)]
    """
    blocks: list[_PdfTextBlock] = []
    for raw_block in raw.get("blocks", []):
        if raw_block.get("type", 0) != 0:
            continue
        bbox = _valid_bbox(raw_block.get("bbox"))
        if bbox is None:
            continue
        text = _block_text(raw_block)
        if not text:
            continue
        max_size = _max_span_size(raw_block, default=median_size)
        block_type = (
            BlockType.HEADING
            if median_size and max_size >= median_size * heading_size_ratio
            else BlockType.TEXT
        )
        blocks.append(
            _PdfTextBlock(
                block_type=block_type,
                text=text,
                level=1 if block_type is BlockType.HEADING else 0,
                page=page_no,
                bbox=bbox,
                max_size=max_size,
            )
        )
    return blocks


def _block_text(raw_block: dict[str, Any]) -> str:
    """保留行级顺序抽取 PDF 文本块内容。

    Example Input:
        {"lines": [{"spans": [{"text": "A"}, {"text": "B"}]}]}

    Example Output:
        "A B"
    """
    lines: list[str] = []
    for line in raw_block.get("lines", []):
        text = " ".join(
            span.get("text", "").strip()
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ).strip()
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _max_span_size(raw_block: dict[str, Any], *, default: float) -> float:
    """返回文本块内最大字号。

    Example Input:
        _max_span_size(raw_block, default=10)

    Example Output:
        14.0
    """
    sizes = [
        float(span["size"])
        for line in raw_block.get("lines", [])
        for span in line.get("spans", [])
        if "size" in span
    ]
    return max(sizes, default=default)


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """校验并归一化 PDF bbox。

    Example Input:
        _valid_bbox((0, 1, 20, 30))

    Example Output:
        (0.0, 1.0, 20.0, 30.0)
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _page_width(page: Any, raw: dict[str, Any]) -> float:
    """读取页面宽度，缺失时从 bbox 推导。

    Example Input:
        _page_width(page, raw)

    Example Output:
        595.0
    """
    rect = getattr(page, "rect", None)
    width = float(getattr(rect, "width", 0.0) or 0.0)
    if width > 0:
        return width
    x1_values = [
        bbox[2]
        for block in raw.get("blocks", [])
        if (bbox := _valid_bbox(block.get("bbox"))) is not None
    ]
    return max(x1_values, default=1.0)


def _sort_pdf_blocks_by_reading_order(
    blocks: list[_PdfTextBlock],
    page_width: float,
) -> list[_PdfTextBlock]:
    """按页面阅读顺序排序 PDF 文本块。

    Example Input:
        _sort_pdf_blocks_by_reading_order([left, right], page_width=600)

    Example Output:
        [left, right]
    """
    if len(blocks) < 4 or page_width <= 0:
        return sorted(blocks, key=_y_x_key)
    columns = _detect_columns(blocks, page_width)
    if len(columns) < 2:
        return sorted(blocks, key=_y_x_key)
    return _sort_columnar_blocks(blocks, columns, page_width)


def _detect_columns(
    blocks: list[_PdfTextBlock],
    page_width: float,
) -> tuple[_ColumnRegion, ...]:
    """基于横向投影检测页面列。

    Example Input:
        _detect_columns(blocks, page_width=600)

    Example Output:
        (_ColumnRegion(x0=40, x1=260, ...), _ColumnRegion(x0=320, x1=540, ...))
    """
    candidates = [
        block
        for block in blocks
        if block.block_type is not BlockType.HEADING
        and len(block.text) >= 2
        and _bbox_width(block.bbox) <= page_width * _COLUMN_SPAN_WIDTH_RATIO
    ]
    if len(candidates) < 4:
        return ()

    gap_threshold = max(page_width * _COLUMN_GAP_RATIO, _COLUMN_MIN_GAP)
    clusters: list[_ColumnRegion] = []
    for block in sorted(candidates, key=lambda item: item.bbox[0]):
        x0, _, x1, _ = block.bbox
        if not clusters or x0 - clusters[-1].x1 > gap_threshold:
            clusters.append(
                _ColumnRegion(
                    x0=x0,
                    x1=x1,
                    block_count=1,
                    text_chars=len(block.text),
                )
            )
            continue
        last = clusters[-1]
        clusters[-1] = _ColumnRegion(
            x0=min(last.x0, x0),
            x1=max(last.x1, x1),
            block_count=last.block_count + 1,
            text_chars=last.text_chars + len(block.text),
        )

    if not 2 <= len(clusters) <= _COLUMN_MAX_COUNT:
        return ()

    max_chars = max(cluster.text_chars for cluster in clusters)
    min_chars = max(_COLUMN_MIN_TEXT_CHARS, int(max_chars * _COLUMN_MIN_SUPPORT_RATIO))
    if any(cluster.block_count < 2 and cluster.text_chars < min_chars for cluster in clusters):
        return ()
    return tuple(clusters)


def _sort_columnar_blocks(
    blocks: list[_PdfTextBlock],
    columns: tuple[_ColumnRegion, ...],
    page_width: float,
) -> list[_PdfTextBlock]:
    """在多栏页面中保持跨栏块边界，并按列阅读。

    Example Input:
        _sort_columnar_blocks([title, left, right], columns, 600)

    Example Output:
        [title, left, right]
    """
    spans: list[_PdfTextBlock] = []
    column_blocks: list[tuple[int, _PdfTextBlock]] = []
    for block in blocks:
        column_index = _column_index(block, columns, page_width)
        if column_index < 0:
            spans.append(block)
        else:
            column_blocks.append((column_index, block))

    if not spans:
        return _sort_band_by_columns(column_blocks)

    ordered: list[_PdfTextBlock] = []
    remaining = list(column_blocks)
    for span in sorted(spans, key=_y_x_key):
        above = [(index, block) for index, block in remaining if block.bbox[1] < span.bbox[1]]
        if above:
            ordered.extend(_sort_band_by_columns(above))
            above_ids = {id(block) for _, block in above}
            remaining = [
                (index, block)
                for index, block in remaining
                if id(block) not in above_ids
            ]
        ordered.append(span)
    if remaining:
        ordered.extend(_sort_band_by_columns(remaining))
    return ordered


def _sort_band_by_columns(
    blocks: list[tuple[int, _PdfTextBlock]],
) -> list[_PdfTextBlock]:
    """对一个跨栏边界之间的文本块按列排序。

    Example Input:
        _sort_band_by_columns([(0, left1), (1, right1), (0, left2)])

    Example Output:
        [left1, left2, right1]
    """
    ordered: list[_PdfTextBlock] = []
    for column_index in sorted({index for index, _ in blocks}):
        same_column = [block for index, block in blocks if index == column_index]
        ordered.extend(sorted(same_column, key=_y_x_key))
    return ordered


def _column_index(
    block: _PdfTextBlock,
    columns: tuple[_ColumnRegion, ...],
    page_width: float,
) -> int:
    """返回文本块所属列；跨栏块返回 -1。

    Example Input:
        _column_index(block, columns, page_width=600)

    Example Output:
        0
    """
    if _is_spanning_block(block, columns, page_width):
        return -1
    center = (block.bbox[0] + block.bbox[2]) / 2
    distances = [
        0.0 if column.x0 <= center <= column.x1 else min(
            abs(center - column.x0),
            abs(center - column.x1),
        )
        for column in columns
    ]
    return min(range(len(columns)), key=lambda index: distances[index])


def _is_spanning_block(
    block: _PdfTextBlock,
    columns: tuple[_ColumnRegion, ...],
    page_width: float,
) -> bool:
    """判断文本块是否跨越多个页面列。

    Example Input:
        _is_spanning_block(title_block, columns, 600)

    Example Output:
        True
    """
    width = _bbox_width(block.bbox)
    if width >= page_width * _COLUMN_SPAN_WIDTH_RATIO:
        return True
    overlaps = 0
    for column in columns:
        overlap = max(0.0, min(block.bbox[2], column.x1) - max(block.bbox[0], column.x0))
        if overlap >= (column.x1 - column.x0) * 0.25:
            overlaps += 1
    return overlaps > 1


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    """返回 bbox 宽度。

    Example Input:
        _bbox_width((0, 0, 12, 8))

    Example Output:
        12.0
    """
    return bbox[2] - bbox[0]


def _y_x_key(block: _PdfTextBlock) -> tuple[float, float]:
    """返回稳定的 y,x 排序键。

    Example Input:
        _y_x_key(block)

    Example Output:
        (100.0, 40.0)
    """
    return round(block.bbox[1], 2), round(block.bbox[0], 2)


# --------------------------------------------------------------------------- #
# 路线 B：视觉检测模型适配骨架(PP-Structure / LayoutParser / DocLayout)        #
# --------------------------------------------------------------------------- #
class ModelLayoutAnalyzer(LayoutAnalyzer):
    """视觉版面检测模型适配骨架。生产接入步骤(在 analyze 内实现)：

    1. 把 PDF 页渲染成图(200~300 DPI)。
    2. 调用检测模型 → 每区域 bbox + 类别 + 置信度。
    3. 类别映射到 BlockType；表格区送表格结构识别(TSR)输出 Markdown 表。
    4. 阅读顺序：先分栏再 y 排序，或调排序模型(LayoutReader)。
    5. 区域内文本：文本型裁文本层、扫描件送 OCR(置信度写入 Block.confidence)。

    这里仅定义契约，默认抛错以提示需接入具体后端。
    """

    def __init__(self, backend: str = "pp-structure") -> None:
        self.backend = backend

    def analyze(self, source: str) -> list[Block]:  # pragma: no cover - 需模型后端
        raise NotImplementedError(
            f"ModelLayoutAnalyzer({self.backend}) 需接入视觉检测后端，"
            "参考 docstring 步骤实现后返回归一化的 Block 列表"
        )


# --------------------------------------------------------------------------- #
# 选型分流：按数据源自动选择分析器                                              #
# --------------------------------------------------------------------------- #
def select_analyzer(path_or_text: str, *, is_path: bool = False) -> LayoutAnalyzer:
    """简化版选型：
    - .md/.txt 或纯文本 → 规则版
    - .pdf → 文本型走坐标聚类，图片型(无文本层)应走 ModelLayoutAnalyzer
    生产应真正探测 PDF 是否含文本层来决定，而非仅看扩展名。
    """
    if is_path and path_or_text.lower().endswith(".pdf"):
        return PdfTextLayoutAnalyzer()
    return MarkdownLayoutAnalyzer()
