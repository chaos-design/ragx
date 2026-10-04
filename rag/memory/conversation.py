"""对话上下文节点 —— 维护多轮历史。

职责边界：纯状态存储，与检索/生成完全分离。
依赖：rag.interfaces。
"""
from __future__ import annotations

from rag.interfaces import ChatMessage, ConversationMemory


class WindowBufferMemory(ConversationMemory):
    """滑动窗口记忆：仅保留最近 N 轮，防止 prompt 过长。"""

    def __init__(self, max_messages: int = 10) -> None:
        self._max = max_messages
        self._buffer: list[ChatMessage] = []

    def add(self, message: ChatMessage) -> None:
        self._buffer.append(message)
        if len(self._buffer) > self._max:
            self._buffer = self._buffer[-self._max :]

    def history(self) -> list[ChatMessage]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
