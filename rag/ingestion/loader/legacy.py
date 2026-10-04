"""兼容旧接口的文本文件 Loader。"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from rag.interfaces import Document, DocumentLoader

from .utils import read_text_file


class TextFileLoader(DocumentLoader):
    """加载 .txt/.md 文件并按字符窗口切分（带 overlap）。"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        """初始化文本切分参数。

        输入 (Input):
            chunk_size: 每个 chunk 的最大字符数。
            chunk_overlap: 相邻 chunk 的重叠字符数。

        输出 (Output):
            None。

        Example Input:
            TextFileLoader(chunk_size=300, chunk_overlap=50)

        Example Output:
            TextFileLoader 实例
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk(self, text: str) -> list[str]:
        """按字符窗口切分文本。

        输入 (Input):
            text: 原始文本。

        输出 (Output):
            非空 chunk 列表。

        Example Input:
            self._chunk("abcdef")

        Example Output:
            ["abc", "cde", "ef"]
        """
        text = text.strip()
        if not text:
            return []
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), step)
            if text[i : i + self.chunk_size].strip()
        ]

    def load(self, source: str) -> list[Document]:
        """读取旧版文本源并返回 Document chunk。

        输入 (Input):
            source: .txt/.md 文件路径或目录路径。

        输出 (Output):
            Document 列表，每个 Document 带 source 与 chunk 元数据。

        Example Input:
            TextFileLoader(chunk_size=3, chunk_overlap=1).load("agent-library/ragx/data")

        Example Output:
            [Document(id="...", content="abc", metadata={"source": "...", "chunk": 0})]
        """
        paths = []
        if os.path.isdir(source):
            for root, _, files in os.walk(source):
                paths += [
                    os.path.join(root, filename)
                    for filename in files
                    if filename.endswith((".txt", ".md"))
                ]
        else:
            paths = [source]

        docs: list[Document] = []
        for path in paths:
            content = read_text_file(Path(path))
            for idx, chunk in enumerate(self._chunk(content)):
                docs.append(
                    Document(
                        id=str(uuid.uuid4()),
                        content=chunk,
                        metadata={"source": path, "chunk": idx},
                    )
                )
        return docs
