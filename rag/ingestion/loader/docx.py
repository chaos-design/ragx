"""DOCX 文档读取。"""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx(path: Path) -> str:
    """读取 DOCX 正文和表格文本。

    输入 (Input):
        path: DOCX 文件路径。

    输出 (Output):
        由段落和 Markdown 风格表格组成的文本。

    Example Input:
        read_docx(Path("agent-library/ragx/data/resources/ai-agent-engineer.docx"))

    Example Output:
        "# ai-agent-engineer.docx\\n\\nAI Agent Engineer"
    """
    try:
        with zipfile.ZipFile(path) as docx:
            xml = docx.read("word/document.xml")
    except KeyError as exc:
        raise ValueError(f"DOCX 缺少 word/document.xml: {path}") from exc

    root = ElementTree.fromstring(xml)
    body = root.find(f"{WORD_NS}body")
    if body is None:
        return ""

    parts: list[str] = [f"# {path.name}"]
    for child in body:
        if child.tag == f"{WORD_NS}p":
            text = paragraph_text(child)
            if text:
                parts.append(text)
        elif child.tag == f"{WORD_NS}tbl":
            table = table_text(child)
            if table:
                parts.append(table)
    return "\n\n".join(parts)


def paragraph_text(paragraph: ElementTree.Element) -> str:
    """读取 DOCX 段落文本。

    输入 (Input):
        paragraph: w:p XML 节点。

    输出 (Output):
        段落纯文本。

    Example Input:
        paragraph_text(paragraph)

    Example Output:
        "AI Agent Engineer"
    """
    return "".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t")).strip()


def table_text(table: ElementTree.Element) -> str:
    """读取 DOCX 表格为 Markdown 风格文本。

    输入 (Input):
        table: w:tbl XML 节点。

    输出 (Output):
        Markdown 风格表格文本。

    Example Input:
        table_text(table)

    Example Output:
        "| 字段 | 说明 |\\n| provider | agent_provider |"
    """
    rows: list[str] = []
    for row in table.iter(f"{WORD_NS}tr"):
        cells = [paragraph_text(cell) for cell in row.iter(f"{WORD_NS}tc")]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
