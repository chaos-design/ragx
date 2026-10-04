"""模块④RAG 调用逻辑 —— ChatService 抽象 + 适配器。

解耦设计：定义 ChatService 协议，CLI 只依赖它，不知道背后是真实 RAG 还是 mock。
  - RagChatService：适配现有 RagApplication(注入)，把 ask() 结果归一化。
  - EchoChatService：零依赖假实现，无需建库即可联调 CLI 交互/loading。

对外接口：
  ChatService.answer(query: str) -> ChatAnswer(text, contexts)

依赖：rag.app.RagApplication(仅 RagChatService 用，且通过注入)。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ChatAnswer:
    text: str
    contexts: list[dict] = field(default_factory=list)


@runtime_checkable
class ChatService(Protocol):
    def answer(self, query: str) -> ChatAnswer: ...


class RagChatService:
    """适配 RagApplication：注入实例，调用 ask() 并归一化输出。"""

    def __init__(self, app) -> None:
        self._app = app

    def answer(self, query: str) -> ChatAnswer:
        res = self._app.ask(query)
        return ChatAnswer(text=res["answer"], contexts=res.get("contexts", []))


class EchoChatService:
    """假实现：用于无知识库时联调 CLI(可模拟延迟以观察 loading)。"""

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    def answer(self, query: str) -> ChatAnswer:
        if self._delay:
            time.sleep(self._delay)
        return ChatAnswer(text=f"(echo) 你说的是：{query}", contexts=[])
