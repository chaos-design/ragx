"""词法检索后端 —— hybrid 检索中的 BM25 辅助召回。

职责边界：只负责把结构化 chunk 放入内存索引，并用 BM25 做检索排序。
它不调用大模型、不生成 mock 数据；回答仍由注入的真实 LLMProvider 完成。
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence

from rag.ingestion.chunker import StructureAwareChunker
from rag.ingestion.manifest import InMemoryManifest, ManifestStore
from rag.ingestion.sync import SyncReport
from rag.interfaces import Document, Retriever, ScoredDocument

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]+")
_ASCII_RE = re.compile(r"^[A-Za-z0-9_]+$")
BM25_K1 = 1.5
BM25_B = 0.75
CJK_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "或",
    "及",
    "要",
    "几",
    "什么",
    "怎么",
    "如何",
    "是否",
}


def _sha1(text: str) -> str:
    """计算文本 SHA1。

    输入 (Input):
        text: 原始文本。

    输出 (Output):
        SHA1 十六进制字符串。

    示例 (Example):
        _sha1("hello")
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> list[str]:
    """把中英文混合文本切成检索词项。

    输入 (Input):
        text: 待检索文本。

    输出 (Output):
        小写词项列表；英文按词切分，中文按 2/3-gram 切分。

    示例 (Example):
        _tokenize("RAG 报销")
    """
    terms: list[str] = []
    for raw_token in _TOKEN_RE.findall(text):
        token = raw_token.lower()
        if _ASCII_RE.match(raw_token):
            if token not in CJK_STOPWORDS:
                terms.append(token)
            continue
        if len(token) == 1:
            if token not in CJK_STOPWORDS:
                terms.append(token)
            continue
        for size in (2, 3):
            terms.extend(_cjk_ngrams(token, size))
        if 2 <= len(token) <= 6 and token not in CJK_STOPWORDS:
            terms.append(token)
    return [term for term in terms if term and term not in CJK_STOPWORDS]


def _cjk_ngrams(text: str, size: int) -> list[str]:
    """生成中文连续片段的 n-gram。

    输入 (Input):
        text: 中文连续文本。
        size: n-gram 长度。

    输出 (Output):
        n-gram 列表。

    示例 (Example):
        _cjk_ngrams("体验治理", 2)
    """
    if size <= 0 or len(text) < size:
        return []
    return [text[index:index + size] for index in range(len(text) - size + 1)]


def _query_phrases(query: str) -> list[str]:
    """提取用于短语加权的查询片段。

    输入 (Input):
        query: 用户问题。

    输出 (Output):
        去重后的英文词、中文 3/4-gram 与短中文片段。

    示例 (Example):
        _query_phrases("CPO 体验治理的目标是什么？")
    """
    phrases: list[str] = []
    for raw_token in _TOKEN_RE.findall(query):
        token = raw_token.lower()
        if _ASCII_RE.match(raw_token):
            if len(token) >= 2:
                phrases.append(token)
            continue
        if 2 <= len(token) <= 8 and token not in CJK_STOPWORDS:
            phrases.append(token)
        for size in (3, 4):
            phrases.extend(_cjk_ngrams(token, size))
    return _unique(phrase for phrase in phrases if phrase not in CJK_STOPWORDS)


def _unique(values: Iterable[str]) -> list[str]:
    """按原始顺序去重。

    输入 (Input):
        values: 字符串序列。

    输出 (Output):
        去重后的字符串列表。

    示例 (Example):
        _unique(["a", "a", "b"])
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _bm25_score(
    query_terms: Counter[str],
    doc_terms: Counter[str],
    *,
    doc_count: int,
    doc_freq: Counter[str],
    doc_length: int,
    avg_doc_length: float,
) -> float:
    """计算 BM25 相关性分数。

    输入 (Input):
        query_terms: 查询词频。
        doc_terms: 文档词频。
        doc_count: 语料文档数量。
        doc_freq: 语料 document frequency。
        doc_length: 当前文档长度。
        avg_doc_length: 平均文档长度。

    输出 (Output):
        BM25 分数，越大越相关。

    示例 (Example):
        _bm25_score(Counter({"报销": 1}), Counter({"报销": 2}), doc_count=1,
                    doc_freq=Counter({"报销": 1}), doc_length=2,
                    avg_doc_length=2.0)
    """
    if not query_terms or not doc_terms or doc_count <= 0:
        return 0.0

    score = 0.0
    normalized_length = avg_doc_length or 1.0
    for term, query_count in query_terms.items():
        term_frequency = doc_terms.get(term, 0)
        if term_frequency <= 0:
            continue
        frequency = doc_freq.get(term, 0)
        idf = math.log(1.0 + (doc_count - frequency + 0.5) / (frequency + 0.5))
        denominator = term_frequency + BM25_K1 * (
            1.0 - BM25_B + BM25_B * doc_length / normalized_length
        )
        score += (
            query_count
            * idf
            * (term_frequency * (BM25_K1 + 1.0))
            / (denominator or 1.0)
        )
    return score


def _phrase_boost(query_phrases: Sequence[str], document: Document) -> float:
    """计算短语、标题和来源命中的轻量补偿分。

    输入 (Input):
        query_phrases: 查询中的重要短语。
        document: 候选文档分块。

    输出 (Output):
        轻量补偿分，用于解决精确实体被长文泛词压过的问题。

    示例 (Example):
        _phrase_boost(["cpo"], Document("1", "CPO 体验治理", {}))
    """
    if not query_phrases:
        return 0.0

    content = document.content.lower()
    heading = str(document.metadata.get("heading_path") or "").lower()
    source = str(document.metadata.get("source") or "").lower()

    boost = 0.0
    for phrase in query_phrases:
        weight = 0.5 + min(len(phrase), 8) * 0.35
        if phrase in content:
            boost += weight
        if phrase in heading:
            boost += weight * 0.75
        if phrase in source:
            boost += min(weight * 0.4, 1.0)
    return boost


def _metadata_boost(query_terms: Counter[str], document: Document) -> float:
    """计算标题和来源字段的词项补偿分。

    输入 (Input):
        query_terms: 查询词频。
        document: 候选文档分块。

    输出 (Output):
        标题/来源命中的补偿分。

    示例 (Example):
        _metadata_boost(Counter({"检索": 1}), Document("1", "", {}))
    """
    heading_terms = Counter(_tokenize(str(document.metadata.get("heading_path") or "")))
    source_terms = Counter(_tokenize(str(document.metadata.get("source") or "")))
    boost = 0.0
    for term, query_count in query_terms.items():
        if term in heading_terms:
            boost += 4.0 * query_count
        if term in source_terms:
            boost += 0.8 * query_count
    return boost


def _match(meta: dict, flt: dict) -> bool:
    """检查 metadata 是否满足过滤条件。

    输入 (Input):
        meta: 文档 metadata。
        flt: 等值或 in-list 过滤条件。

    输出 (Output):
        True 表示命中过滤条件。

    示例 (Example):
        _match({"is_latest": True}, {"is_latest": True})
    """
    for key, expected in flt.items():
        actual = meta.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class LexicalDocumentStore:
    """内存词法索引，保存真实数据分块与词项统计。"""

    def __init__(self) -> None:
        """初始化词法索引。

        输入 (Input):
            None。

        输出 (Output):
            None。

        示例 (Example):
            store = LexicalDocumentStore()
        """
        self._docs: dict[str, Document] = {}
        self._terms: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_freq: Counter[str] = Counter()
        self._postings: dict[str, set[str]] = {}
        self._avg_doc_length: float = 0.0

    def add(self, documents: Sequence[Document]) -> None:
        """写入文档分块。

        输入 (Input):
            documents: 结构化分块后的 Document 列表。

        输出 (Output):
            None。

        示例 (Example):
            store.add(chunks)
        """
        for document in documents:
            self._remove_terms(document.id)
            terms = Counter(_tokenize(document.content))
            self._docs[document.id] = document
            self._terms[document.id] = terms
            self._doc_lengths[document.id] = sum(terms.values())
            self._doc_freq.update(terms.keys())
            for term in terms:
                self._postings.setdefault(term, set()).add(document.id)
        self._refresh_avg_doc_length()

    def delete_by_doc(self, doc_id: str) -> int:
        """按 doc_id 删除分块。

        输入 (Input):
            doc_id: 文档 ID。

        输出 (Output):
            删除的 chunk 数量。

        示例 (Example):
            store.delete_by_doc("a.md")
        """
        targets = [
            chunk_id
            for chunk_id, document in self._docs.items()
            if document.metadata.get("doc_id") == doc_id
        ]
        for chunk_id in targets:
            self._docs.pop(chunk_id, None)
            self._remove_terms(chunk_id)
        self._refresh_avg_doc_length()
        return len(targets)

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[ScoredDocument]:
        """按倒排候选与 BM25 排序检索文档。

        输入 (Input):
            query: 用户问题。
            top_k: 返回数量。
            metadata_filter: metadata 过滤条件。

        输出 (Output):
            按相关性排序的 ScoredDocument 列表。

        示例 (Example):
            store.search("报销时间", top_k=4)
        """
        if top_k <= 0:
            return []

        query_terms = Counter(_tokenize(query))
        query_phrases = _query_phrases(query)
        if not query_terms and not query_phrases:
            return []

        candidate_ids = self._candidate_ids(query_terms)
        if not candidate_ids and query_phrases:
            candidate_ids = set(self._docs)

        scored: list[ScoredDocument] = []
        for chunk_id in candidate_ids:
            document = self._docs[chunk_id]
            if metadata_filter and not _match(document.metadata, metadata_filter):
                continue
            doc_terms = self._terms[chunk_id]
            score = _bm25_score(
                query_terms,
                doc_terms,
                doc_count=len(self._docs),
                doc_freq=self._doc_freq,
                doc_length=self._doc_lengths.get(chunk_id, 0),
                avg_doc_length=self._avg_doc_length,
            )
            score += _phrase_boost(query_phrases, document)
            score += _metadata_boost(query_terms, document)
            if score > 0:
                scored.append(ScoredDocument(document=document, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _candidate_ids(self, query_terms: Counter[str]) -> set[str]:
        """按倒排索引返回候选 chunk。

        输入 (Input):
            query_terms: 查询词频。

        输出 (Output):
            至少包含一个查询词项的 chunk_id 集合。

        示例 (Example):
            store._candidate_ids(Counter({"报销": 1}))
        """
        candidates: set[str] = set()
        for term in query_terms:
            candidates.update(self._postings.get(term, set()))
        return candidates

    def _remove_terms(self, chunk_id: str) -> None:
        """从语料统计中移除指定 chunk 的词项。

        输入 (Input):
            chunk_id: chunk 主键。

        输出 (Output):
            None。

        示例 (Example):
            store._remove_terms("a.md::0")
        """
        terms = self._terms.pop(chunk_id, None)
        self._doc_lengths.pop(chunk_id, None)
        if not terms:
            return
        for term in terms:
            self._doc_freq[term] -= 1
            if self._doc_freq[term] <= 0:
                del self._doc_freq[term]
            posting = self._postings.get(term)
            if not posting:
                continue
            posting.discard(chunk_id)
            if not posting:
                del self._postings[term]

    def _refresh_avg_doc_length(self) -> None:
        """刷新 BM25 平均文档长度。

        输入 (Input):
            None。

        输出 (Output):
            None。

        示例 (Example):
            store._refresh_avg_doc_length()
        """
        if not self._doc_lengths:
            self._avg_doc_length = 0.0
            return
        self._avg_doc_length = sum(self._doc_lengths.values()) / len(
            self._doc_lengths
        )


class LexicalRetriever(Retriever):
    """Retriever 适配器，面向 RagPipeline 暴露统一检索接口。"""

    def __init__(self, store: LexicalDocumentStore) -> None:
        """初始化词法 Retriever。

        输入 (Input):
            store: 词法索引。

        输出 (Output):
            None。

        示例 (Example):
            LexicalRetriever(store)
        """
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict | None = None,
    ) -> list[ScoredDocument]:
        """检索相关文档。

        输入 (Input):
            query: 用户问题。
            top_k: 返回数量。
            metadata_filter: metadata 过滤条件。

        输出 (Output):
            ScoredDocument 列表。

        示例 (Example):
            retriever.retrieve("报销时间")
        """
        return self._store.search(query, top_k=top_k, metadata_filter=metadata_filter)


class LexicalSyncer:
    """增量构建词法索引，不依赖 embedding provider。"""

    def __init__(
        self,
        chunker: StructureAwareChunker,
        store: LexicalDocumentStore,
        manifest: ManifestStore | None = None,
    ) -> None:
        """初始化词法同步器。

        输入 (Input):
            chunker: 结构感知分块器。
            store: 词法索引。
            manifest: 可选 manifest，默认进程内。

        输出 (Output):
            None。

        示例 (Example):
            LexicalSyncer(chunker, store)
        """
        self._chunker = chunker
        self._store = store
        self._manifest_store: ManifestStore = manifest or InMemoryManifest()
        self._manifest: dict[str, dict] = self._manifest_store.load()

    def sync(self, sources: dict[str, str], *, acl: list[str] | None = None) -> SyncReport:
        """同步真实数据到词法索引。

        输入 (Input):
            sources: {doc_id: 原始全文}。
            acl: 可选访问控制标签。

        输出 (Output):
            SyncReport。

        示例 (Example):
            syncer.sync({"a.md": "# A"})
        """
        report = SyncReport()
        seen: set[str] = set()

        for doc_id, raw_text in sources.items():
            seen.add(doc_id)
            source_hash = _sha1(raw_text)
            prev = self._manifest.get(doc_id)
            if prev and prev["source_hash"] == source_hash:
                report.skipped += 1
                continue

            version = (prev["version"] + 1) if prev else 1
            deleted = self._store.delete_by_doc(doc_id) if prev else 0
            report.deleted += deleted

            metadata = {
                "doc_id": doc_id,
                "source": doc_id,
                "version": version,
                "is_latest": True,
                "acl": acl or ["public"],
                "updated_at": int(time.time()),
            }
            chunks = self._chunker.chunk(raw_text, metadata)
            self._store.add(chunks)
            report.reembedded_chunks += len(chunks)

            if prev:
                report.updated += 1
                report.details.append(
                    f"[UPD-LEXICAL] {doc_id} v{version}: del {deleted} -> add {len(chunks)}"
                )
            else:
                report.added += 1
                report.details.append(f"[ADD-LEXICAL] {doc_id} -> {len(chunks)} chunks")

            self._manifest[doc_id] = {
                "source_hash": source_hash,
                "version": version,
                "updated_at": metadata["updated_at"],
            }

        for doc_id in list(self._manifest.keys()):
            if doc_id not in seen:
                deleted = self._store.delete_by_doc(doc_id)
                report.deleted += deleted
                self._manifest.pop(doc_id, None)
                report.details.append(f"[DEL-LEXICAL] {doc_id} -> removed {deleted} chunks")

        self._manifest_store.save(self._manifest)
        return report
