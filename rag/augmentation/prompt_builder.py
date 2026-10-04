"""增强节点 —— 把 检索上下文 + 对话历史 + 用户问题 组装为最终 messages。

职责边界：只做 prompt 工程，不调用任何模型。
依赖：rag.interfaces。
"""
from __future__ import annotations

from typing import Iterable, Sequence

from rag.interfaces import ChatMessage, PromptBuilder, ScoredDocument

_SYSTEM = (
    "你是面向企业知识库的专业问答助手。必须严格依据【上下文】作答，"
    "优先给出可验证、结构清晰、措辞审慎的结论；"
    "当上下文缺少充分证据或无法支持结论时，应明确说明"
    "“无法从现有资料中确认”，不得补充、推测或编造未被资料支持的信息。"
)


class RagPromptBuilder(PromptBuilder):
    def build(
        self,
        query: str,
        contexts: Sequence[ScoredDocument],
        history: Iterable[ChatMessage],
    ) -> list[ChatMessage]:
        ctx_text = "\n\n".join(
            f"[片段{i + 1} | 来源:{c.document.metadata.get('source', 'NA')} "
            f"| 相关度:{c.score:.3f}]\n{c.document.content}"
            for i, c in enumerate(contexts)
        ) or "（无检索结果）"

        messages: list[ChatMessage] = [ChatMessage(role="system", content=_SYSTEM)]
        messages.extend(history)
        messages.append(
            ChatMessage(
                role="user",
                content=f"【上下文】\n{ctx_text}\n\n【问题】\n{query}",
            )
        )
        return messages
