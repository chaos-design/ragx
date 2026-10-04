"""源文档读取入口。"""
from __future__ import annotations

from pathlib import Path

from .docx import read_docx
from .pdf import read_pdf
from .utils import iter_source_paths, make_doc_id, read_text_file


def read_source_documents(source: str) -> dict[str, str]:
    """读取目录或单文件中的真实文档文本。

    输入 (Input):
        source: 数据源目录或单文件路径。目录会递归读取所有文件。

    输出 (Output):
        {doc_id: text} 字典。目录输入时 doc_id 为相对路径，单文件输入时为文件名。

    Example Input:
        read_source_documents("agent-library/ragx/data/resources")

    Example Output:
        {"rag-sample.md": "# RAG\\n..."}
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"数据源不存在: {source}")

    documents: dict[str, str] = {}
    for path in iter_source_paths(source_path):
        text = read_source_file(path).strip()
        if not text:
            continue
        documents[make_doc_id(path, source_path)] = text

    if not documents:
        raise ValueError(f"数据源未包含可读取的真实文档: {source}")
    return documents


def read_source_file(path: Path) -> str:
    """按文件格式读取源文档文本。

    输入 (Input):
        path: 源文件路径。

    输出 (Output):
        可用于分块和 embedding 的文本。

    Example Input:
        read_source_file(Path("agent-library/ragx/data/resources/rag-sample.md"))

    Example Output:
        "# RAG\\n..."
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return read_text_file(path)
