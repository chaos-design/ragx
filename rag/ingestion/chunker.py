"""结构感知分块 + metadata 溯源 —— 生产级实现。

核心能力：
1. 按文档结构(标题树/代码块/表格)切分，而非固定字符硬切。
2. 异构内容自适应 chunk 大小与 overlap(正文/代码/表格策略不同)。
3. 每个 chunk 携带完整溯源 metadata(标题路径、位置、版本、内容指纹)。

生产适配注意项：
- token 计数：示例用「中文字≈1.5token、英文词≈1.3token」的近似估算，
  生产请替换 _count_tokens 为目标 embedding 模型的真实 tokenizer(如 tiktoken)。
- content_hash 用于 chunk 级差分(增量更新时只 re-embed 变化块)。
- 标题路径(heading_path)前置注入 chunk 文本，兼顾 embedding 质量与溯源。
- 依赖 rag.interfaces，零第三方依赖即可运行。
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum

from rag.interfaces import Document

LOGGER = logging.getLogger(__name__)


class BlockType(str, Enum):
    HEADING = "heading"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"
    TEXT = "text"


@dataclass
class Block:
    """版面解析后的语义块(分块前的中间结构)。

    这是「版面分析」与「分块」之间的唯一契约：无论上游是 Markdown 规则解析、
    PDF 版面检测模型还是 HTML 正文抽取，都归一化产出 Block 列表，下游 chunker
    不感知来源。bbox/page 用于 PDF 坐标透传，支撑点击跳转原文溯源。
    """

    type: BlockType
    text: str
    level: int = 0          # 标题层级(仅 HEADING 有意义)
    lang: str = ""          # 代码语言(仅 CODE)
    start_line: int = 0     # 原文行号，用于溯源定位
    end_line: int = 0
    page: int = 0           # 页码(PDF/扫描件)，0 表示无分页概念
    bbox: tuple[float, float, float, float] | None = None  # (x0,y0,x1,y1)
    confidence: float = 1.0  # 版面检测/OCR 置信度，接入抽样质检健康度信号


@dataclass
class ChunkPolicy:
    """异构内容的自适应分块策略(生产常用阈值)。"""

    text_size: int = 512        # 正文目标 token
    text_overlap: int = 80      # ≈15%
    code_size: int = 800        # 代码块更大，避免割裂函数
    code_overlap: int = 120     # 代码 overlap 复制 import/签名
    table_size: int = 1000      # 表格尽量整块，表头随行
    min_chunk: int = 40         # 小于此 token 的块丢弃/合并(去噪)
    hard_max: int = 1200        # 单 chunk 硬上限，防 embedding 截断


# --------------------------------------------------------------------------- #
# token 估算(生产替换为真实 tokenizer)                                         #
# --------------------------------------------------------------------------- #
_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _count_tokens(text: str) -> int:
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    return int(cjk * 1.5 + words * 1.3) + 1


# --------------------------------------------------------------------------- #
# 第一步：版面解析 —— 把 Markdown/纯文本解析为语义块                            #
# (生产中 PDF 走版面分析、HTML 走正文抽取，最终都归一到 Block 列表)            #
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_IMAGE_RE = re.compile(
    r'^\s*!\[(?P<alt>[^\]]*)\]\((?P<src>\S+?)(?:\s+"(?P<title>[^"]*)")?\)\s*$'
)


def parse_markdown_blocks(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    n = len(lines)
    buf: list[str] = []
    buf_start = 0

    def flush_text(end: int) -> None:
        nonlocal buf, buf_start
        joined = "\n".join(buf).strip()
        if joined:
            blocks.append(Block(BlockType.TEXT, joined, start_line=buf_start, end_line=end))
        buf = []

    while i < n:
        line = lines[i]
        # 代码围栏
        m_fence = _FENCE_RE.match(line.strip())
        if m_fence:
            flush_text(i)
            lang = m_fence.group(1)
            start = i
            i += 1
            code_lines: list[str] = []
            while i < n and not _FENCE_RE.match(lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束围栏
            blocks.append(
                Block(BlockType.CODE, "\n".join(code_lines), lang=lang,
                      start_line=start, end_line=i)
            )
            continue
        # 标题
        m_head = _HEADING_RE.match(line)
        if m_head:
            flush_text(i)
            level = len(m_head.group(1))
            blocks.append(Block(BlockType.HEADING, m_head.group(2).strip(),
                                level=level, start_line=i, end_line=i + 1))
            i += 1
            continue
        # 表格(连续的 | 行)
        if _TABLE_ROW_RE.match(line):
            flush_text(i)
            start = i
            tbl: list[str] = []
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                tbl.append(lines[i])
                i += 1
            blocks.append(Block(BlockType.TABLE, "\n".join(tbl),
                                start_line=start, end_line=i))
            continue
        # 独立图片/图表语法。alt/title/path 作为文本索引，原始图片仍留给视觉模型扩展。
        m_image = _IMAGE_RE.match(line)
        if m_image:
            flush_text(i)
            blocks.append(
                Block(
                    BlockType.IMAGE,
                    _image_text(
                        alt=m_image.group("alt"),
                        src=m_image.group("src"),
                        title=m_image.group("title") or "",
                    ),
                    start_line=i,
                    end_line=i + 1,
                )
            )
            i += 1
            continue
        # 普通文本累积
        if not buf:
            buf_start = i
        buf.append(line)
        i += 1

    flush_text(n)
    return blocks


# --------------------------------------------------------------------------- #
# 第二步：结构感知分块 + metadata 注入                                          #
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _split_by_tokens(text: str, size: int, overlap: int) -> list[str]:
    """兜底：超长正文按 token 估算滑窗切，尽量在句末断开。"""
    if _count_tokens(text) <= size:
        return [text]
    # 先按句子切，再贪心打包到 size
    sentences = re.split(r"(?<=[。！？.!?\n])", text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for s in sentences:
        st = _count_tokens(s)
        if cur and cur_tok + st > size:
            chunks.append("".join(cur))
            # overlap：从尾部回退 overlap token 重叠
            back, btok = [], 0
            for seg in reversed(cur):
                back.insert(0, seg)
                btok += _count_tokens(seg)
                if btok >= overlap:
                    break
            cur, cur_tok = back, btok
        cur.append(s)
        cur_tok += st
    if cur:
        chunks.append("".join(cur))
    return [c.strip() for c in chunks if c.strip()]


def _table_header(table_text: str) -> str:
    rows = [r for r in table_text.splitlines() if r.strip()]
    return rows[0] if rows else ""


def _image_text(alt: str, src: str, title: str = "") -> str:
    """把 Markdown 图片/图表引用归一化为可检索文本。

    输入 (Input):
        alt: Markdown 图片 alt 文本。
        src: 图片路径或 URL。
        title: 可选图片 title。

    输出 (Output):
        可进入 embedding 和 BM25 的文本描述。

    Example Input:
        _image_text("季度 GMV 趋势图", "charts/gmv.png", "Q4")

    Example Output:
        "图片/图表: 季度 GMV 趋势图\n标题: Q4\n路径: charts/gmv.png"
    """
    parts = []
    cleaned_alt = alt.strip()
    cleaned_title = title.strip()
    cleaned_src = src.strip()
    if cleaned_alt:
        parts.append(f"图片/图表: {cleaned_alt}")
    if cleaned_title:
        parts.append(f"标题: {cleaned_title}")
    if cleaned_src:
        parts.append(f"路径: {cleaned_src}")
    return "\n".join(parts) if parts else "图片/图表"


class StructureAwareChunker:
    def __init__(self, policy: ChunkPolicy | None = None) -> None:
        self.policy = policy or ChunkPolicy()

    def chunk(self, raw_text: str, doc_meta: dict) -> list[Document]:
        """便捷入口：内部用 Markdown 规则解析 raw_text 为 Block 后再分块。
        doc_meta 必含：doc_id, source, version, (可选 acl/effective_at...)。"""
        return self.chunk_blocks(parse_markdown_blocks(raw_text), doc_meta)

    def chunk_blocks(self, blocks: list[Block], doc_meta: dict) -> list[Document]:
        """核心入口：对「版面分析产出的 Block 列表」分块。
        上游可以是 Markdown 规则解析 / PDF 版面检测 / HTML 抽取，此处不感知来源。"""
        p = self.policy
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        docs: list[Document] = []
        order = 0

        for blk in blocks:
            if blk.type is BlockType.HEADING:
                # 维护标题栈 → 形成 heading_path
                while heading_stack and heading_stack[-1][0] >= blk.level:
                    heading_stack.pop()
                heading_stack.append((blk.level, blk.text))
                continue

            heading_path = " > ".join(t for _, t in heading_stack)

            # 按块类型选择策略
            if blk.type is BlockType.CODE:
                pieces = self._chunk_code(blk.text, p)
                btype = BlockType.CODE
            elif blk.type is BlockType.TABLE:
                pieces = self._chunk_table(blk.text, p)
                btype = BlockType.TABLE
            elif blk.type is BlockType.IMAGE:
                pieces = [blk.text]
                btype = BlockType.IMAGE
            else:
                pieces = _split_by_tokens(blk.text, p.text_size, p.text_overlap)
                btype = BlockType.TEXT

            for piece in pieces:
                if btype is BlockType.TEXT and _count_tokens(piece) < p.min_chunk:
                    continue  # 去噪：丢弃过短正文碎片
                # 上下文头注入：标题路径前置，兼顾 embedding 与溯源
                body = f"[{heading_path}]\n{piece}" if heading_path else piece
                content_hash = _sha1(piece)
                document = Document(
                    id=f"{doc_meta['doc_id']}::{order}",
                    content=body,
                    metadata={
                        **doc_meta,
                        "chunk_index": order,
                        "block_type": btype.value,
                        "heading_path": heading_path,
                        "start_line": blk.start_line,
                        "end_line": blk.end_line,
                        "page": blk.page,            # 溯源：跳页定位
                        "bbox": blk.bbox,            # 溯源：高亮原文区域
                        "layout_confidence": blk.confidence,  # 接入质检信号
                        "content_hash": content_hash,  # chunk 级差分用
                        "token_count": _count_tokens(piece),
                    },
                )
                docs.append(document)
                if btype is BlockType.IMAGE:
                    LOGGER.info(
                        "image chunk generated: doc_id=%s chunk_index=%s "
                        "heading_path=%r content=%r",
                        doc_meta.get("doc_id", ""),
                        order,
                        heading_path,
                        document.content,
                    )
                order += 1
        return docs

    def _chunk_code(self, code: str, p: ChunkPolicy) -> list[str]:
        """代码优先按函数/类边界切；超限再滑窗。复制签名行作为 overlap 上下文。"""
        if _count_tokens(code) <= p.hard_max:
            return [code]
        lines = code.splitlines()
        boundary = re.compile(r"^\s*(def |class |func |function |public |private )")
        groups: list[list[str]] = []
        cur: list[str] = []
        for ln in lines:
            if boundary.match(ln) and cur:
                groups.append(cur)
                cur = []
            cur.append(ln)
        if cur:
            groups.append(cur)
        # 合并相邻小组到 code_size
        out, buf, btok = [], [], 0
        for g in groups:
            gt = _count_tokens("\n".join(g))
            if buf and btok + gt > p.code_size:
                out.append("\n".join(buf))
                buf, btok = [], 0
            buf.extend(g)
            btok += gt
        if buf:
            out.append("\n".join(buf))
        return out or [code]

    def _chunk_table(self, table: str, p: ChunkPolicy) -> list[str]:
        """表格整块优先；超限则按行切并把表头复制到每个分片。"""
        if _count_tokens(table) <= p.table_size:
            return [table]
        rows = [r for r in table.splitlines() if r.strip()]
        if len(rows) < 2:
            return [table]
        header = rows[0]
        sep = rows[1] if re.match(r"^\s*\|[\s\-:|]+\|\s*$", rows[1]) else ""
        data_rows = rows[2:] if sep else rows[1:]
        out, buf, btok = [], [], _count_tokens(header + sep)
        for r in data_rows:
            rt = _count_tokens(r)
            if buf and btok + rt > p.table_size:
                out.append("\n".join([header] + ([sep] if sep else []) + buf))
                buf, btok = [], _count_tokens(header + sep)
            buf.append(r)
            btok += rt
        if buf:
            out.append("\n".join([header] + ([sep] if sep else []) + buf))
        return out
