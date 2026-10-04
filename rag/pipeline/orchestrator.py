"""流程编排节点 —— 串联 检索 → 增强 → 生成 → 记忆 全链路。

职责边界：只做「编排」，不实现任何具体能力；所有依赖通过构造函数注入。
这是唯一知道各节点如何协作的地方，其余节点彼此互不感知。
依赖：各节点抽象 + DTO。
"""
from __future__ import annotations

from rag.augmentation.prompt_builder import RagPromptBuilder
from rag.generation.generator import Generator
from rag.interfaces import ChatMessage, ConversationMemory, Retriever


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: RagPromptBuilder,
        generator: Generator,
        memory: ConversationMemory,
        top_k: int = 4,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generator = generator
        self._memory = memory
        self._top_k = top_k

    def ask(self, query: str) -> dict:
        # 1) 检索
        contexts = self._retriever.retrieve(query, top_k=self._top_k)
        # 2) 增强（注入历史）
        messages = self._prompt_builder.build(
            query=query, contexts=contexts, history=self._memory.history()
        )
        # 3) 生成
        answer = self._generator.generate(messages)
        # 4) 写回对话记忆
        self._memory.add(ChatMessage(role="user", content=query))
        self._memory.add(ChatMessage(role="assistant", content=answer))
        return {
            "answer": answer,
            "contexts": [
                {
                    "source": c.document.metadata.get("source"),
                    "score": round(c.score, 4),
                    "preview": c.document.content,
                }
                for c in contexts
            ],
        }
