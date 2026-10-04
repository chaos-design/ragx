"""父子分块 (small-to-big / parent-child) —— 解决「小块精度高但上下文不足」。

思路：
  - 子块(small)：粒度小，语义聚焦，用于 embedding 与检索 → 召回精度高。
  - 父块(big)：粒度大(整段/整节)，命中子块后返回其父块给生成层 → 上下文完整。

实现：
  1. ParentChildChunker 先用 StructureAwareChunker 在「父粒度」切出父块，
     再在每个父块内部切出更小的子块，子块 metadata 带 parent_id。
  2. 只把【子块】向量化入向量库；父块文本存入 ParentStore(可换 DB)。
  3. ParentExpandingRetriever 包装基础检索器：检索子块 → 按 parent_id 去重
     → 用父块文本替换，喂给生成层。

生产适配注意项：
  - 父子比例：父 512~1024 token、子 128~256 token 较常用；子太碎会丢指代。
  - 去重：多个子块可能命中同一父块，需按 parent_id 去重并保留最高分。
  - ParentStore 生产应与向量库同源(同一 doc_id 可级联删除)，避免父块成孤儿。
依赖：rag.ingestion.chunker + rag.interfaces。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.ingestion.chunker import (
    Block,
    BlockType,
    ChunkPolicy,
    StructureAwareChunker,
    parse_markdown_blocks,
)
from rag.interfaces import Document, Retriever, ScoredDocument


# --------------------------------------------------------------------------- #
# 父块存储(可换 SQLite/Redis；这里用内存 + 按 doc_id 级联删除)                  #
# --------------------------------------------------------------------------- #
class ParentStore:
    def __init__(self) -> None:
        self._parents: dict[str, Document] = {}

    def put(self, parent: Document) -> None:
        self._parents[parent.id] = parent

    def get(self, parent_id: str) -> Document | None:
        return self._parents.get(parent_id)

    def delete_by_doc(self, doc_id: str) -> int:
        targets = [pid for pid, p in self._parents.items()
                   if p.metadata.get("doc_id") == doc_id]
        for pid in targets:
            self._parents.pop(pid, None)
        return len(targets)


@dataclass
class ParentChildPolicy:
    parent: ChunkPolicy = field(
        default_factory=lambda: ChunkPolicy(text_size=900, text_overlap=0, min_chunk=20))
    child: ChunkPolicy = field(
        default_factory=lambda: ChunkPolicy(text_size=200, text_overlap=40, min_chunk=10))


class ParentChildChunker:
    """产出 (子块 Documents 用于检索, 父块写入 parent_store)。"""

    def __init__(self, policy: ParentChildPolicy | None = None,
                 parent_store: ParentStore | None = None) -> None:
        self.policy = policy or ParentChildPolicy()
        self.parent_store = parent_store or ParentStore()
        self._parent_chunker = StructureAwareChunker(self.policy.parent)
        self._child_chunker = StructureAwareChunker(self.policy.child)

    def chunk(self, raw_text: str, doc_meta: dict) -> list[Document]:
        """把原始文本解析为 Block 后产出用于检索的子块。

        Example Input:
            chunker.chunk("# Guide\\nbody", {"doc_id": "guide.md"})

        Example Output:
            [Document(id="guide.md::0::c0", metadata={"parent_id": ...})]
        """
        return self.chunk_blocks(parse_markdown_blocks(raw_text), doc_meta)

    def chunk_blocks(self, blocks: list[Block], doc_meta: dict) -> list[Document]:
        """消费版面分析后的 Block，产出子块并保存父块。

        Example Input:
            chunker.chunk_blocks([Block(BlockType.TEXT, "body")], doc_meta)

        Example Output:
            子块入向量库，父块写入 ParentStore。
        """
        # 1) 父粒度切块
        parents = self._parent_chunker.chunk_blocks(blocks, doc_meta)
        child_docs: list[Document] = []
        for parent in parents:
            parent.metadata["is_parent"] = True
            self.parent_store.put(parent)
            # 2) 在父块文本内部再切子块(去掉注入的标题头，按原 piece 切)
            inner_blocks = [Block(BlockType.TEXT, parent.content)]
            children = self._child_chunker.chunk_blocks(
                inner_blocks,
                {**doc_meta, "parent_id": parent.id},
            )
            for c_idx, child in enumerate(children):
                child.id = f"{parent.id}::c{c_idx}"
                child.metadata["parent_id"] = parent.id
                child.metadata["is_parent"] = False
                child_docs.append(child)
        return child_docs  # 仅子块入向量库


# --------------------------------------------------------------------------- #
# 检索扩展：子块命中 → 替换为父块                                              #
# --------------------------------------------------------------------------- #
class ParentExpandingRetriever(Retriever):
    def __init__(self, base: Retriever, parent_store: ParentStore,
                 child_top_k: int = 8) -> None:
        self._base = base
        self._parents = parent_store
        self._child_top_k = child_top_k

    def retrieve(self, query: str, top_k: int = 4,
                 metadata_filter: dict | None = None) -> list[ScoredDocument]:
        # 先多召子块(child_top_k > top_k)，再按 parent_id 去重升维到父块
        children = self._base.retrieve(
            query, top_k=max(self._child_top_k, top_k), metadata_filter=metadata_filter)
        seen: dict[str, ScoredDocument] = {}
        for sc in children:
            pid = sc.document.metadata.get("parent_id")
            if pid is None:
                # 无父块(降级)：直接保留子块
                seen.setdefault(sc.document.id, sc)
                continue
            # 同父块只保留最高分子块的得分
            if pid not in seen or sc.score > seen[pid].score:
                parent = self._parents.get(pid)
                doc = parent if parent else sc.document
                seen[pid] = ScoredDocument(document=doc, score=sc.score)
        out = sorted(seen.values(), key=lambda s: s.score, reverse=True)
        return out[:top_k]
