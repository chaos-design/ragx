"""统一接口契约层 (Ports)。

本模块只定义抽象基类(ABC)与数据结构(DTO)，不包含任何具体实现。
所有上层模块仅依赖这里的抽象，从而实现「依赖倒置 / 高内聚低耦合」。

依赖关系：Provider 抽象来自共享 `agent_provider` 包，其余流程接口保留在本文件。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Sequence

try:  # pragma: no cover - 取决于运行环境是否安装了共享包
    from agent_provider import (
        ChatMessage as ChatMessage,
        EmbeddingProvider as EmbeddingProvider,
        LLMProvider as LLMProvider,
    )
except ImportError:  # pragma: no cover - 独立运行时的正常路径
    from rag.providers._fallback import (
        ChatMessage as ChatMessage,
        EmbeddingProvider as EmbeddingProvider,
        LLMProvider as LLMProvider,
    )


# --------------------------------------------------------------------------- #
# 数据传输对象 (DTO)：模块间唯一的数据交换格式                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    """切分后的最小知识单元。"""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredDocument:
    """检索命中的文档 + 相关性得分。"""

    document: Document
    score: float


# --------------------------------------------------------------------------- #
# 各流程节点抽象接口                                                            #
# --------------------------------------------------------------------------- #
class DocumentLoader(ABC):
    """数据接入：把原始数据源加载并切分为 Document 列表。"""

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        raise NotImplementedError


class VectorStore(ABC):
    """向量存储 + 相似度检索。"""

    @abstractmethod
    def add(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[ScoredDocument]:
        raise NotImplementedError

    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> int:
        """按 doc_id 删除其名下全部 chunk，返回删除条数(规避孤儿向量)。"""
        raise NotImplementedError

    @abstractmethod
    def list_hashes(self, doc_id: str) -> dict[str, str]:
        """返回该 doc_id 下 {chunk_id: content_hash}，供 chunk 级差分。"""
        raise NotImplementedError


class Retriever(ABC):
    """检索器：给定 query 文本，返回相关文档。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4, metadata_filter: dict | None = None) -> list[ScoredDocument]:
        raise NotImplementedError


class PromptBuilder(ABC):
    """增强：把检索结果 + 历史 + 问题拼装为最终 messages。"""

    @abstractmethod
    def build(
        self,
        query: str,
        contexts: Sequence[ScoredDocument],
        history: Iterable[ChatMessage],
    ) -> list[ChatMessage]:
        raise NotImplementedError


class ConversationMemory(ABC):
    """对话上下文存储。"""

    @abstractmethod
    def add(self, message: ChatMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    def history(self) -> list[ChatMessage]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
