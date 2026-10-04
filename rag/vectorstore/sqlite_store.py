"""SQLite 持久化向量库 —— 可直接运行的真实数据库后端实现。

价值：证明 VectorStore 抽象可无缝跑在真实 DB 引擎上(非内存)，并演示
delete-then-insert、metadata 过滤、chunk 级差分在 SQL 语义下的落地。
SQLite 零依赖、随 Python 自带，适合本地/小规模生产与集成测试。

存储模型：
  chunks(chunk_id PK, doc_id, content, vector(JSON), metadata(JSON), content_hash)
  - doc_id 建索引 → delete_by_doc / list_hashes 走索引，O(log n) 定位。
  - 向量以 JSON 存(SQLite 无原生向量类型)；相似度在 Python 侧算。
    大规模请用 pgvector/Milvus(见 pgvector_store.py)，此处重在跑通真实持久化。

生产适配注意项：
  - SQLite 单写并发弱：高并发写请加应用层锁或换 PG。
  - 全表算余弦不适合百万级；本实现用于中小库或测试。
  - WAL 模式提升读写并发；已默认开启。
依赖：标准库 sqlite3 + rag.interfaces。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from typing import Sequence

from rag.interfaces import Document, ScoredDocument, VectorStore


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _match(meta: dict, flt: dict) -> bool:
    for k, v in flt.items():
        mv = meta.get(k)
        if isinstance(v, (list, tuple, set)):
            if mv not in v:
                return False
        elif mv != v:
            return False
    return True


class SQLiteVectorStore(VectorStore):
    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        # check_same_thread=False 便于多线程读；生产写仍建议串行
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                doc_id       TEXT NOT NULL,
                content      TEXT NOT NULL,
                vector       TEXT NOT NULL,   -- JSON array
                metadata     TEXT NOT NULL,   -- JSON object
                content_hash TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        self._conn.commit()

    def add(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        rows = [
            (
                doc.id,
                doc.metadata.get("doc_id", ""),
                doc.content,
                json.dumps(list(vec)),
                json.dumps(doc.metadata, ensure_ascii=False, default=str),
                doc.metadata.get("content_hash", ""),
            )
            for doc, vec in zip(documents, vectors)
        ]
        # upsert：chunk_id 冲突则覆盖，天然幂等
        self._conn.executemany(
            """INSERT INTO chunks(chunk_id, doc_id, content, vector, metadata, content_hash)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(chunk_id) DO UPDATE SET
                 content=excluded.content, vector=excluded.vector,
                 metadata=excluded.metadata, content_hash=excluded.content_hash""",
            rows,
        )
        self._conn.commit()

    def search(self, query_vector, top_k, metadata_filter=None):
        cur = self._conn.execute("SELECT content, vector, metadata FROM chunks")
        scored: list[ScoredDocument] = []
        for content, vec_json, meta_json in cur.fetchall():
            meta = json.loads(meta_json)
            if metadata_filter and not _match(meta, metadata_filter):
                continue
            vec = json.loads(vec_json)
            if not vec:
                continue
            doc = Document(id=meta.get("__chunk_id__", ""), content=content, metadata=meta)
            scored.append(ScoredDocument(document=doc, score=_cosine(query_vector, vec)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def delete_by_doc(self, doc_id: str) -> int:
        # 走 doc_id 索引，原子删除该文档全部 chunk → 规避孤儿向量
        cur = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return cur.rowcount

    def list_hashes(self, doc_id: str) -> dict[str, str]:
        cur = self._conn.execute(
            "SELECT chunk_id, content_hash FROM chunks WHERE doc_id = ?", (doc_id,)
        )
        return {cid: h for cid, h in cur.fetchall()}

    def close(self) -> None:
        self._conn.close()
