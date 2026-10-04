"""生成节点 —— 调用 LLMProvider 产出最终回答。

职责边界：只负责「messages→answer」
依赖：rag.interfaces。
"""
from __future__ import annotations

from typing import Sequence

from rag.interfaces import ChatMessage, LLMProvider


class Generator:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate(self, messages: Sequence[ChatMessage], **kwargs) -> str:
        return self._provider.chat(messages, **kwargs)
