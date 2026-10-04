"""向量库工厂 —— 生产装配点。

默认使用 SQLite 持久化；只有显式配置 pgvector 时才切换到 PostgreSQL。
依赖：rag.interfaces + 各 store 实现 + config。
"""
from __future__ import annotations

from rag.interfaces import VectorStore


def build_vector_store(cfg) -> VectorStore:
    backend = cfg.vector_backend.lower().strip()
    if backend == "pgvector":
        from rag.vectorstore.pgvector_store import PgVectorStore
        return PgVectorStore(cfg.pg_dsn, dim=cfg.embedding_dim)

    from rag.vectorstore.sqlite_store import SQLiteVectorStore
    return SQLiteVectorStore(cfg.sqlite_path)
