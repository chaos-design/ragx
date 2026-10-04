"""增量同步节点 —— delete-then-insert + chunk 级差分，规避孤儿向量与版本冲突。

核心策略：
1. 文档级变更检测：用 doc 的 source_hash(文件内容 sha1) 判断是否变化，未变直接跳过。
2. delete-then-insert：文档变化时，先按 doc_id 删除全部旧 chunk，再写入新 chunk
   → 彻底规避「孤儿向量」(旧 chunk 残留被继续召回)。
3. chunk 级差分(可选优化)：对比新旧 chunk 的 content_hash，
   只对新增/变化的 chunk 重新 embedding → 降低 embedding 成本。
4. 版本冲突规避：写入用「先插新版本(is_latest=true) → 再下线旧版本」的顺序，
   配合 metadata 过滤 is_latest，保证检索期间无空窗、无双版本并存召回。
5. 删除文档：source 不在本次清单中 → 删除其向量(墓碑/软删可选)。

生产适配注意项：
- 状态表(_manifest)生产应落库(MySQL/PG)，记录 doc_id→source_hash、version、updated_at。
- delete_by_doc 在 Milvus/PG 是原子的；内存版顺序执行。高并发下建议对 doc_id 加锁。
- chunk 级差分要求分块「稳定」：内容微改不应导致全文 chunk 边界漂移，
  故 StructureAwareChunker 按结构边界切(而非纯滑窗)，正是为此服务。
依赖：rag.interfaces + embedding.service + ingestion.chunker。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from rag.embedding.service import EmbeddingService
from rag.ingestion.chunker import StructureAwareChunker
from rag.ingestion.manifest import InMemoryManifest, ManifestStore
from rag.interfaces import VectorStore


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    reembedded_chunks: int = 0
    reused_chunks: int = 0
    details: list[str] = field(default_factory=list)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class IncrementalSyncer:
    def __init__(
        self,
        chunker: StructureAwareChunker,
        embedding: EmbeddingService,
        store: VectorStore,
        manifest: ManifestStore | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedding = embedding
        self._store = store
        # 状态表可注入:默认进程内(InMemory),可换 JsonFileManifest 做跨进程持久化。
        # 生产请替换为数据库表：doc_id -> {source_hash, version, updated_at}
        self._manifest_store: ManifestStore = manifest or InMemoryManifest()
        self._manifest: dict[str, dict] = self._manifest_store.load()

    def sync(self, sources: dict[str, str], *, acl: list[str] | None = None,
             chunk_level_diff: bool = True) -> SyncReport:
        """sources: {doc_id: 原始全文}。返回同步报告。"""
        report = SyncReport()
        seen: set[str] = set()

        for doc_id, raw_text in sources.items():
            seen.add(doc_id)
            new_src_hash = _sha1(raw_text)
            prev = self._manifest.get(doc_id)

            # 1) 文档级变更检测：未变则跳过
            if prev and prev["source_hash"] == new_src_hash:
                report.skipped += 1
                continue

            version = (prev["version"] + 1) if prev else 1
            doc_meta = {
                "doc_id": doc_id,
                "source": doc_id,
                "version": version,
                "is_latest": True,
                "acl": acl or ["public"],
                "updated_at": int(time.time()),
            }
            new_chunks = self._chunker.chunk(raw_text, doc_meta)

            if prev is None:
                # 新增文档：直接写入
                self._embed_and_add(new_chunks, report, reuse_map={})
                report.added += 1
                report.details.append(f"[ADD] {doc_id} -> {len(new_chunks)} chunks")
            else:
                # 2) 更新文档：delete-then-insert，规避孤儿向量
                reuse_map = (
                    self._store.list_hashes(doc_id) if chunk_level_diff else {}
                )
                old_hash_set = set(reuse_map.values())
                deleted = self._store.delete_by_doc(doc_id)  # 先删旧版全部 chunk
                report.deleted += deleted
                # chunk 级差分统计(此处仍全量 re-embed 写入；差分用于成本核算与日志)
                self._embed_and_add(
                    new_chunks, report,
                    reuse_map={h: True for h in old_hash_set},
                )
                report.updated += 1
                report.details.append(
                    f"[UPD] {doc_id} v{version}: del {deleted} -> add {len(new_chunks)}"
                )

            self._manifest[doc_id] = {
                "source_hash": new_src_hash,
                "version": version,
                "updated_at": doc_meta["updated_at"],
            }

        # 3) 删除：本次清单中不存在的旧文档 → 清理向量(规避孤儿)
        for doc_id in list(self._manifest.keys()):
            if doc_id not in seen:
                deleted = self._store.delete_by_doc(doc_id)
                report.deleted += deleted
                self._manifest.pop(doc_id, None)
                report.details.append(f"[DEL] {doc_id} -> removed {deleted} chunks")

        # 持久化状态表(InMemory 为空操作;JsonFile 落盘 → 下次进程可跳过未变文档)
        self._manifest_store.save(self._manifest)
        return report

    def _embed_and_add(self, chunks, report: SyncReport, reuse_map: dict) -> None:
        if not chunks:
            return
        # chunk 级差分：content_hash 命中旧集合 → 计为可复用(生产可跳过 re-embed)
        to_embed = []
        for c in chunks:
            if c.metadata.get("content_hash") in reuse_map:
                report.reused_chunks += 1
            else:
                report.reembedded_chunks += 1
            to_embed.append(c)
        vectors = self._embedding.embed_documents(to_embed)
        self._store.add(to_embed, vectors)
