"""模块⑤会话状态管理 —— 多轮对话历史与轮次。

解耦设计：Session 只管"状态"(纯内存数据结构)，不感知输入/渲染/RAG。
  - 记录每轮 (user, assistant, contexts)；
  - 提供轮次计数、清空、导出为 ChatMessage 历史(供 Provider 多轮拼接)；
  - 零 IO、可单测。

对外接口：
  Session.add_turn(user, assistant, contexts=None)
  Session.turns -> list[Turn]
  Session.count -> int
  Session.clear()
  Session.history(max_turns=None) -> list[ChatMessage]

依赖：rag.interfaces.ChatMessage(仅 history() 用，做多轮上下文拼装)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.interfaces import ChatMessage


@dataclass
class Turn:
    user: str
    assistant: str
    contexts: list[dict] = field(default_factory=list)


class Session:
    """多轮会话状态:纯内存、可替换、可单测。"""

    def __init__(self) -> None:
        self._turns: list[Turn] = []

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def count(self) -> int:
        return len(self._turns)

    def add_turn(self, user: str, assistant: str,
                 contexts: list[dict] | None = None) -> Turn:
        turn = Turn(user=user, assistant=assistant, contexts=contexts or [])
        self._turns.append(turn)
        return turn

    def clear(self) -> None:
        self._turns.clear()

    def history(self, max_turns: int | None = None) -> list[ChatMessage]:
        """导出为 ChatMessage 序列，供多轮上下文拼接。

        max_turns 限制只取最近 N 轮(滑动窗口，防上下文超长)。
        """
        turns = self._turns if max_turns is None else self._turns[-max_turns:]
        msgs: list[ChatMessage] = []
        for t in turns:
            msgs.append(ChatMessage(role="user", content=t.user))
            msgs.append(ChatMessage(role="assistant", content=t.assistant))
        return msgs
