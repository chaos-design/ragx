"""三大生产模块端到端验证：结构感知分块 / 增量同步 / 抽样质检。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.embedding.service import EmbeddingService          # noqa: E402
from rag.ingestion.chunker import ChunkPolicy, StructureAwareChunker  # noqa: E402
from rag.ingestion.sync import IncrementalSyncer            # noqa: E402
from rag.quality.sampling_qa import (                       # noqa: E402
    SamplingQA, SampleResult, compute_signal)
from rag.vectorstore.memory_store import InMemoryVectorStore  # noqa: E402

DOC_V1 = """# 员工手册

## 第3章 考勤

### 3.2 请假
员工请假需提前一天提交申请。病假需提供证明。

| 假别 | 上限 | 审批人 |
| --- | --- | --- |
| 年假 | 15天 | 直属上级 |
| 病假 | 30天 | HRBP |

```python
def apply_leave(days):
    return days <= 15
```
"""

DOC_V2 = DOC_V1.replace("提前一天", "提前三天")  # 仅正文微改


class _FakeEmbeddingProvider:
    def embed(self, texts):
        vectors = []
        for text in texts:
            cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
            ascii_words = sum(1 for char in text if char.isascii() and char.isalnum())
            checksum = sum(ord(char) for char in text) % 997
            vectors.append([float(len(text) or 1), float(cjk), float(ascii_words), float(checksum)])
        return vectors


def test_structure_aware_chunking():
    chunker = StructureAwareChunker(ChunkPolicy(text_size=128, min_chunk=5))
    chunks = chunker.chunk(DOC_V1, {"doc_id": "handbook.md", "source": "handbook.md", "version": 1})
    types = {c.metadata["block_type"] for c in chunks}
    assert "table" in types and "code" in types and "text" in types, types
    # 标题路径注入溯源
    assert any("第3章 考勤 > 3.2 请假" in c.metadata["heading_path"] for c in chunks)
    # 每个 chunk 带溯源 metadata
    for c in chunks:
        assert {"doc_id", "chunk_index", "content_hash", "start_line"} <= c.metadata.keys()
    print(f"✓ chunking: {len(chunks)} chunks, types={types}")


def test_incremental_sync():
    store = InMemoryVectorStore()
    emb = EmbeddingService(_FakeEmbeddingProvider())
    syncer = IncrementalSyncer(StructureAwareChunker(), emb, store)

    r1 = syncer.sync({"handbook.md": DOC_V1})
    assert r1.added == 1 and r1.deleted == 0
    n_after_add = len(store._docs)

    # 未变 → skip，无重复写入(规避孤儿与膨胀)
    r2 = syncer.sync({"handbook.md": DOC_V1})
    assert r2.skipped == 1
    assert len(store._docs) == n_after_add

    # 内容变更 → delete-then-insert，无孤儿残留
    r3 = syncer.sync({"handbook.md": DOC_V2})
    assert r3.updated == 1 and r3.deleted > 0
    # 旧版本 chunk 不应残留：所有 chunk 的 version 均为 2
    assert all(d.metadata["version"] == 2 for d in store._docs.values())

    # 文档下线 → 清理向量
    r4 = syncer.sync({})
    assert r4.deleted > 0 and len(store._docs) == 0
    print(f"✓ sync: add={r1.added} skip={r2.skipped} upd={r3.updated} "
          f"reused={r3.reused_chunks} reembed={r3.reembedded_chunks} del={r4.deleted}")


def test_sampling_qa():
    qa = SamplingQA(min_per_stratum=2, sample_ratio=0.5)
    signals = [compute_signal(f"a{i}.pdf", i, "正常中文内容 normal text " * 5,
                              ocr_confidence=0.97) for i in range(5)]
    signals += [compute_signal("bad.pdf", 99, "���乱码@@@###", ocr_confidence=0.4)]
    plan = qa.plan_samples(signals)
    assert any(s.doc_id == "bad.pdf" for s in plan), "低健康度页应被优先抽中"

    results = [SampleResult("a0.pdf", 0, char_accuracy=0.995),
               SampleResult("bad.pdf", 99, char_accuracy=0.80, note="OCR差")]
    rep = qa.evaluate(results)
    assert rep.failed == 1 and rep.release_ok is False
    print(f"✓ sampling-qa: sampled={len(plan)} pass_rate={rep.pass_rate} "
          f"release_ok={rep.release_ok}")


if __name__ == "__main__":
    test_structure_aware_chunking()
    test_incremental_sync()
    test_sampling_qa()
    print("\n✓✓✓ all production modules passed")
