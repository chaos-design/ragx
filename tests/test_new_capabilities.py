"""新增能力验证：SQLite 后端 / 向量库工厂 / 父子分块 / Provider 健康检查装配。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings  # noqa: E402
from rag.app import RagApplication  # noqa: E402
from rag.embedding.service import EmbeddingService  # noqa: E402
from rag.ingestion.parent_child import (ParentChildChunker,  # noqa: E402
                                        ParentExpandingRetriever, ParentStore)
from rag.ingestion.sync import IncrementalSyncer  # noqa: E402
from rag.retrieval.retriever import VectorRetriever  # noqa: E402
from rag.vectorstore.factory import build_vector_store  # noqa: E402
from rag.vectorstore.memory_store import InMemoryVectorStore  # noqa: E402
from rag.vectorstore.sqlite_store import SQLiteVectorStore  # noqa: E402

DOC = ("# 制度\n\n## 报销\n出差报销需在七天内提交相关凭证材料，逾期未提交的费用"
       "将无法报销，请所有同事务必在规定时间窗口内完成线上提交与审批流程，避免影响个人权益。\n")


class _FakeEmbeddingProvider:
    def embed(self, texts):
        vectors = []
        for text in texts:
            cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
            ascii_words = sum(1 for char in text if char.isascii() and char.isalnum())
            checksum = sum(ord(char) for char in text) % 997
            vectors.append([float(len(text) or 1), float(cjk), float(ascii_words), float(checksum)])
        return vectors


def test_sqlite_backend_crud():
    store = SQLiteVectorStore(":memory:")
    emb = EmbeddingService(_FakeEmbeddingProvider())
    syncer = IncrementalSyncer(__import__("rag.ingestion.chunker", fromlist=["StructureAwareChunker"]).StructureAwareChunker(), emb, store)
    r1 = syncer.sync({"rule.md": DOC})
    assert r1.added == 1
    # 检索命中
    ret = VectorRetriever(emb, store)
    hits = ret.retrieve("报销几天内提交", top_k=3)
    assert hits and any("报销" in h.document.content for h in hits)
    # delete_by_doc 走 SQL，规避孤儿
    n = store.delete_by_doc("rule.md")
    assert n > 0 and store.list_hashes("rule.md") == {}
    print(f"✓ sqlite backend: add+search+delete ok (deleted {n} chunks)")


def test_vector_store_factory_uses_sqlite_by_default(tmp_path):
    store = build_vector_store(
        Settings(sqlite_path=str(tmp_path / "vectors.db"))
    )
    assert type(store).__name__ == "SQLiteVectorStore"
    store.close()
    print("✓ vector store factory: default production backend is sqlite")


def test_parent_child():
    store = InMemoryVectorStore()
    emb = EmbeddingService(_FakeEmbeddingProvider())
    pstore = ParentStore()
    chunker = ParentChildChunker(parent_store=pstore)
    children = chunker.chunk(DOC * 3, {"doc_id": "d.md", "source": "d.md", "version": 1})
    assert children, "应产出子块"
    assert all(c.metadata.get("parent_id") for c in children), "子块须带 parent_id"
    store.add(children, emb.embed_documents(children))

    base = VectorRetriever(emb, store)
    expanding = ParentExpandingRetriever(base, pstore, child_top_k=6)
    hits = expanding.retrieve("报销提交时间", top_k=2)
    assert hits
    # 命中后返回的应是父块(is_parent=True)，上下文更完整
    assert any(h.document.metadata.get("is_parent") for h in hits), "应升维到父块"
    print(f"✓ parent-child: {len(children)} children -> hit parent block")


def test_app_with_parent_child_and_sqlite(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "rule.md"), "w", encoding="utf-8") as fh:
            fh.write(DOC * 2)
        store_dir = os.path.join(d, "stores")
        os.makedirs(store_dir)
        cfg = Settings(
            provider="mock",
            vector_backend="sqlite",
            sqlite_path=os.path.join(store_dir, "vectors.db"),
            manifest_path=os.path.join(store_dir, "manifest.json"),
            parent_child=True,
        )
        app = RagApplication(cfg)
        rep = app.index(d)
        assert rep.added == 1
        res = app.ask("报销要几天内提交？")
        assert res["answer"] and res["contexts"]
        print(f"✓ app(sqlite+parent_child): indexed, answer has {len(res['contexts'])} contexts")


def test_persistent_manifest_cross_run():
    """落盘 manifest + sqlite:模拟 CLI 重启,验证增量更新跨进程生效。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        doc = os.path.join(d, "rule.md")
        with open(doc, "w", encoding="utf-8") as fh:
            fh.write(DOC)
        store_dir = os.path.join(d, "stores")
        os.makedirs(store_dir)
        mkcfg = lambda: Settings(  # noqa: E731
            provider="mock",
            vector_backend="sqlite",
            sqlite_path=os.path.join(store_dir, "v.db"),
            manifest_path=os.path.join(store_dir, "m.json"),
        )
        # run1:首次建库 -> 新增
        r1 = RagApplication(mkcfg()).index(d)
        assert r1.added == 1 and r1.skipped == 0
        # run2:新进程、内容未变 -> 跳过(manifest 落盘生效)
        r2 = RagApplication(mkcfg()).index(d)
        assert r2.added == 0 and r2.skipped == 1
        # run3:改动文档 -> 更新(delete-then-insert)
        with open(doc, "a", encoding="utf-8") as fh:
            fh.write("\n## 附则\n本制度自发布之日起执行，相关解释权归财务部门所有，"
                     "如遇政策调整以最新通知为准，请各部门遵照执行并及时反馈意见。\n")
        r3 = RagApplication(mkcfg()).index(d)
        assert r3.updated == 1 and r3.deleted > 0
        print(f"✓ persistent manifest: run1 add / run2 skip / run3 update(del {r3.deleted}) ok")


if __name__ == "__main__":
    import pathlib
    import tempfile
    test_sqlite_backend_crud()
    with tempfile.TemporaryDirectory() as tmp:
        test_vector_store_factory_uses_sqlite_by_default(pathlib.Path(tmp))
    test_parent_child()
    test_app_with_parent_child_and_sqlite()
    test_persistent_manifest_cross_run()
    print("\n✓✓✓ all new capability tests passed")
