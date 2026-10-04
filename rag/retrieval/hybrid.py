"""混合检索 —— 向量召回 + BM25 词法召回 + RRF 排名融合。

本模块只组合现有 Retriever 与 Syncer，不关心 embedding 或存储细节。混合
检索是生产默认策略：用语义召回覆盖表达差异，同时保留词法召回对专有名词、
精确编号和标题短语的优势。
"""
from __future__ import annotations

from rag.ingestion.sync import IncrementalSyncer, SyncReport
from rag.interfaces import Retriever, ScoredDocument
from rag.retrieval.fusion import FusionConfig, reciprocal_rank_fusion
from rag.retrieval.lexical import LexicalSyncer


class HybridRetriever(Retriever):
    """混合 Retriever：并行召回、RRF 融合、去重与多样性控制。"""

    def __init__(
        self,
        vector_retriever: Retriever,
        lexical_retriever: Retriever,
        config: FusionConfig | None = None,
    ) -> None:
        """初始化混合检索器。

        输入 (Input):
            vector_retriever: 向量检索器。
            lexical_retriever: 词法检索器。
            config: 可选融合参数。

        输出 (Output):
            None。

        示例 (Example):
            HybridRetriever(vector_retriever, lexical_retriever)
        """
        self._vector = vector_retriever
        self._lexical = lexical_retriever
        self._config = config or FusionConfig()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict | None = None,
    ) -> list[ScoredDocument]:
        """执行混合检索。

        输入 (Input):
            query: 用户查询。
            top_k: 返回数量。
            metadata_filter: metadata 过滤条件。

        输出 (Output):
            融合排序后的 ScoredDocument 列表。

        示例 (Example):
            retriever.retrieve("混合检索怎么做？", top_k=4)
        """
        candidate_k = max(self._config.candidate_k, top_k)
        vector_hits = self._vector.retrieve(
            query,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )
        lexical_hits = self._lexical.retrieve(
            query,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )
        return reciprocal_rank_fusion(
            [vector_hits, lexical_hits],
            weights=[self._config.vector_weight, self._config.lexical_weight],
            top_k=top_k,
            rrf_k=self._config.rrf_k,
            raw_score_weight=self._config.raw_score_weight,
            source_diversity_penalty=self._config.source_diversity_penalty,
        )


class HybridSyncer:
    """混合同步器：向量库增量同步 + 进程内词法索引重建。"""

    def __init__(self, vector_syncer: IncrementalSyncer, lexical_syncer: LexicalSyncer) -> None:
        """初始化混合同步器。

        输入 (Input):
            vector_syncer: 负责 embedding 与向量库同步的 Syncer。
            lexical_syncer: 负责词法索引同步的 Syncer。

        输出 (Output):
            None。

        示例 (Example):
            HybridSyncer(vector_syncer, lexical_syncer)
        """
        self._vector_syncer = vector_syncer
        self._lexical_syncer = lexical_syncer

    def sync(self, sources: dict[str, str], *, acl: list[str] | None = None) -> SyncReport:
        """同步混合检索索引。

        输入 (Input):
            sources: {doc_id: 原始全文}。
            acl: 可选访问控制标签。

        输出 (Output):
            向量同步报告；词法索引为进程内辅助索引，不计入持久化报告。

        示例 (Example):
            syncer.sync({"a.md": "# A"})
        """
        report = self._vector_syncer.sync(sources, acl=acl)
        self._lexical_syncer.sync(sources, acl=acl)
        return report
