"""数据接入 loader 包。

对外保持 `rag.ingestion.loader` 的稳定导入路径，内部按职责拆分到 source、
pdf、docx、legacy 和 utils 模块。
"""
from __future__ import annotations

from .legacy import TextFileLoader
from .source import read_source_documents

__all__ = ["TextFileLoader", "read_source_documents"]
