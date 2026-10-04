"""pgvector(PostgreSQL) 向量库 —— 生产级实现。

适用大规模生产：原生向量类型 + HNSW/IVFFlat 近似索引，DB 侧做相似度排序。
delete-then-insert、metadata 过滤、chunk 级差分均下推到 SQL，原子且高效。

运行依赖 psycopg(v3) + 已装 pgvector 扩展的 PostgreSQL。未安装时构造抛
RuntimeError，由上游降级到 SQLite/内存(本仓库测试即走降级，不强依赖 DB)。

== 生产建表(一次性) ==
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE chunks (
      chunk_id     TEXT PRIMARY KEY,
      doc_id       TEXT NOT NULL,
      content      TEXT NOT NULL,
      embedding    vector(1536) NOT NULL,   -- 维度对齐 embedding 模型
      metadata     JSONB NOT NULL,
      content_hash TEXT
  );
  CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);  -- 近似检索
  CREATE INDEX idx_chunks_doc ON chunks(doc_id);                    -- 删除/差分走索引
  CREATE INDEX idx_chunks_meta ON chunks USING gin (metadata);      -- metadata 过滤

生产适配注意项：
  - 维度必须与 embedding 模型一致；改模型需重建表(向量维度不可变)。
  - delete-then-insert 用同一事务包裹，避免检索期间出现空窗/双版本。
  - metadata 过滤用 JSONB 操作符下推(@>, ->>)，配合 GIN 索引。
  - HNSW 建索引耗内存；大表建议先灌数据再建索引。
  - 连接池(psycopg_pool)管理连接；本骨架用单连接演示。
依赖：(运行时) psycopg + rag.interfaces。
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from rag.interfaces import Document, ScoredDocument, VectorStore


class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, dim: int = 1536, table: str = "chunks") -> None:
        try:
            import psycopg
        except ImportError as e:  # pragma: no cover - 取决于运行环境
            raise RuntimeError("PgVectorStore 需要 psycopg，请 pip install 'psycopg[binary]'") from e
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn
        self._dim = dim
        self._table = table
        self._conn = psycopg.connect(dsn, autocommit=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:  # pragma: no cover - 需真实 DB
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._table} (
                        chunk_id     TEXT PRIMARY KEY,
                        doc_id       TEXT NOT NULL,
                        content      TEXT NOT NULL,
                        embedding    vector({self._dim}) NOT NULL,
                        metadata     JSONB NOT NULL,
                        content_hash TEXT
                    );"""
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_doc "
                f"ON {self._table}(doc_id);"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_meta "
                f"ON {self._table} USING gin (metadata);"
            )
        self._conn.commit()

    @staticmethod
    def _vec_literal(vec: Sequence[float]) -> str:
        return "[" + ",".join(str(float(x)) for x in vec) + "]"

    def add(self, documents, vectors):  # pragma: no cover - 需真实 DB
        with self._conn.cursor() as cur:
            for doc, vec in zip(documents, vectors):
                cur.execute(
                    f"""INSERT INTO {self._table}
                          (chunk_id, doc_id, content, embedding, metadata, content_hash)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                          content=EXCLUDED.content, embedding=EXCLUDED.embedding,
                          metadata=EXCLUDED.metadata, content_hash=EXCLUDED.content_hash""",
                    (
                        doc.id,
                        doc.metadata.get("doc_id", ""),
                        doc.content,
                        self._vec_literal(vec),
                        json.dumps(doc.metadata, ensure_ascii=False, default=str),
                        doc.metadata.get("content_hash", ""),
                    ),
                )
        self._conn.commit()

    def search(self, query_vector, top_k, metadata_filter=None):  # pragma: no cover
        qvec = self._vec_literal(query_vector)
        where, params = "", [qvec]  # 第一个占位符用于 SELECT 里的 score 计算
        if metadata_filter:
            # JSONB 包含匹配：metadata @> '{"is_latest": true}'，配合 GIN 索引
            where = "WHERE metadata @> %s::jsonb"
            params.append(json.dumps(metadata_filter, default=str))
        params += [qvec, top_k]  # ORDER BY 距离 + LIMIT
        sql = (
            f"SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {self._table} {where} "
            f"ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            out = []
            for content, meta, score in cur.fetchall():
                doc = Document(id="", content=content, metadata=meta)
                out.append(ScoredDocument(document=doc, score=float(score)))
            return out

    def delete_by_doc(self, doc_id: str) -> int:  # pragma: no cover - 需真实 DB
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE doc_id = %s", (doc_id,))
            n = cur.rowcount
        self._conn.commit()
        return n

    def list_hashes(self, doc_id: str) -> dict[str, str]:  # pragma: no cover
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT chunk_id, content_hash FROM {self._table} WHERE doc_id = %s",
                (doc_id,),
            )
            return {cid: h for cid, h in cur.fetchall()}
