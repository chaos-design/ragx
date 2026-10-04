"""组合根 (Composition Root) —— 在这里完成全部依赖装配。

把「配置 → Provider → 版面分析 → 结构感知分块 → 增量同步 → 检索/生成」接好。
这是整个应用唯一进行 new 的地方，业务代码不关心如何 new。

生产链路(RagApplication)：
    read_source_documents(data/) → LayoutAnalyzer → StructureAwareChunker
    → IncrementalSyncer.sync → HybridRetriever(metadata 过滤) → RagPipeline.ask

依赖：所有节点实现 + 工厂 + 配置。
"""
from __future__ import annotations

from config.settings import Settings, load_settings, normalize_search_backend
from rag.augmentation.prompt_builder import RagPromptBuilder
from rag.embedding.service import EmbeddingService
from rag.generation.generator import Generator
from rag.ingestion.chunker import ChunkPolicy, StructureAwareChunker
from rag.ingestion.loader import read_source_documents
from rag.ingestion.layout import LayoutAwareChunker, MarkdownLayoutAnalyzer
from rag.ingestion.manifest import InMemoryManifest, JsonFileManifest
from rag.ingestion.parent_child import ParentChildChunker, ParentExpandingRetriever, ParentStore
from rag.ingestion.sync import IncrementalSyncer, SyncReport
from rag.memory.conversation import WindowBufferMemory
from rag.pipeline.orchestrator import RagPipeline
from rag.providers import (
    build_embedding_provider,
    build_llm_provider,
    resolve_provider_bundle,
)
from rag.retrieval.evaluation import EvaluationConfig, evaluate_retrieval_effect
from rag.retrieval.hybrid import FusionConfig, HybridRetriever, HybridSyncer
from rag.retrieval.lexical import (
    LexicalDocumentStore,
    LexicalRetriever,
    LexicalSyncer,
)
from rag.retrieval.reranker import build_reranker, normalize_reranker_backend
from rag.retrieval.retriever import VectorRetriever
from rag.vectorstore.factory import build_vector_store

DEFAULT_RETRIEVAL_CANDIDATE_K = 100
DEFAULT_RERANK_TOP_K = 20


def _read_sources(source: str) -> dict[str, str]:
    """把目录/文件读成 {doc_id: 全文}。doc_id 用相对路径，稳定且可溯源。

    输入 (Input):
        source: 数据源目录或单文件路径。

    输出 (Output):
        {doc_id: text} 字典。

    示例 (Example):
        _read_sources("./data")
    """
    return read_source_documents(source)


def _select_retrieval_backend(cfg: Settings) -> str:
    """选择实际检索后端。

    输入 (Input):
        cfg: 应用配置。

    输出 (Output):
        固定返回 hybrid。provider 连通性由 provider 初始化和请求结果负责暴露。

    示例 (Example):
        _select_retrieval_backend(load_settings())
    """
    return "hybrid"


class RagApplication:
    """生产入口：封装「增量建库」与「问答」两类操作，依赖全部注入。"""

    def __init__(
        self,
        cfg: Settings | None = None,
        *,
        providers: object | None = None,
    ) -> None:
        self.cfg = cfg or load_settings()

        # Provider 层：默认由配置构建，也可由外部直接注入或通过 registry 构建。
        if providers is None:
            llm_provider = build_llm_provider(self.cfg)
            embedding_provider = build_embedding_provider(self.cfg)
        else:
            provider_bundle = resolve_provider_bundle(self.cfg, providers)
            llm_provider = provider_bundle.require_llm()
            embedding_provider = provider_bundle.require_embedding()

        self._llm = llm_provider
        self._retrieval_backend = _select_retrieval_backend(self.cfg)
        self._search_backend = normalize_search_backend(self.cfg.search_backend)
        self.cfg.search_backend = self._search_backend
        self._reranker_backend = normalize_reranker_backend(self.cfg.reranker_backend)

        self._embedding = EmbeddingService(embedding_provider)
        self._store = build_vector_store(self.cfg)
        self._layout_analyzer = MarkdownLayoutAnalyzer()
        chunk_policy = ChunkPolicy(
            text_size=self.cfg.chunk_size or 512,
            text_overlap=self.cfg.chunk_overlap or 80,
        )

        if self.cfg.parent_child:
            # 父子分块：子块入向量库，父块入 ParentStore，检索时升维到父块
            self._parent_store = ParentStore()
            self._chunker = LayoutAwareChunker(
                ParentChildChunker(parent_store=self._parent_store),
                self._layout_analyzer,
            )
            base_retriever = VectorRetriever(self._embedding, self._store)
            vector_retriever = ParentExpandingRetriever(
                base_retriever,
                self._parent_store,
            )
        else:
            self._parent_store = None
            self._chunker = LayoutAwareChunker(
                StructureAwareChunker(chunk_policy),
                self._layout_analyzer,
            )
            vector_retriever = VectorRetriever(self._embedding, self._store)

        vector_syncer = IncrementalSyncer(
            self._chunker, self._embedding, self._store,
            manifest=(JsonFileManifest(self.cfg.manifest_path)
                      if self.cfg.manifest_path else InMemoryManifest()),
        )
        self._lexical_store = LexicalDocumentStore()
        lexical_retriever = LexicalRetriever(self._lexical_store)
        self._retriever = HybridRetriever(
            vector_retriever,
            lexical_retriever,
            config=FusionConfig(),
        )
        self._syncer = HybridSyncer(
            vector_syncer,
            LexicalSyncer(
                self._chunker,
                self._lexical_store,
                manifest=InMemoryManifest(),
            ),
        )
        self._reranker = build_reranker(self.cfg)

        # 编排
        self._pipeline = RagPipeline(
            retriever=self._retriever,
            prompt_builder=RagPromptBuilder(),
            generator=Generator(self._llm),
            memory=WindowBufferMemory(),
            top_k=self.cfg.top_k,
        )

    def index(self, source: str, *, acl: list[str] | None = None) -> SyncReport:
        """版面分析 → 结构感知分块 → 增量同步建库。可重复调用做增量更新。"""
        return self._syncer.sync(_read_sources(source), acl=acl)

    def ask(
        self,
        query: str,
        *,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
    ) -> dict:
        """执行问答，并在候选池内评估重排检索证据。

        Example Input:
            app.ask("报销材料几天内提交？", top_k=3)

        Example Output:
            {"answer": "...", "contexts": [{"score": 0.91, "preview": "..."}]}
        """
        # 默认只查最新版本，规避版本冲突；可叠加 acl 权限过滤
        flt = {"is_latest": True}
        if metadata_filter:
            flt.update(metadata_filter)
        rerank_top_k = top_k or DEFAULT_RERANK_TOP_K
        candidate_k = max(DEFAULT_RETRIEVAL_CANDIDATE_K, rerank_top_k)
        candidates = self._retriever.retrieve(
            query,
            top_k=candidate_k,
            metadata_filter=flt,
        )
        contexts = self._reranker.rerank(
            query,
            candidates,
            top_k=rerank_top_k,
        )
        return self._pipeline_ask_with_contexts(query, contexts)

    def evaluate_retrieval(
        self,
        query: str,
        *,
        metadata_filter: dict | None = None,
        relevant_sources: list[str] | None = None,
        relevant_doc_ids: list[str] | None = None,
        relevant_chunk_ids: list[str] | None = None,
        top_k_values: tuple[int, ...] = (1, 3, 5),
    ) -> dict:
        """执行一次检索效果评估，不触发 LLM 生成。

        Example Input:
            app.evaluate_retrieval("报销材料", relevant_sources=["rule.md"])

        Example Output:
            {"hit_at_k": {1: True, 3: True}, "mrr": 1.0, ...}
        """
        flt = {"is_latest": True}
        if metadata_filter:
            flt.update(metadata_filter)
        candidate_k = max(
            DEFAULT_RETRIEVAL_CANDIDATE_K,
            max(top_k_values, default=0),
        )
        candidates = self._retriever.retrieve(
            query,
            top_k=candidate_k,
            metadata_filter=flt,
        )
        contexts = self._reranker.rerank(query, candidates, top_k=candidate_k)
        report = evaluate_retrieval_effect(
            query,
            contexts,
            relevant_sources=relevant_sources or (),
            relevant_doc_ids=relevant_doc_ids or (),
            relevant_chunk_ids=relevant_chunk_ids or (),
            config=EvaluationConfig(top_k_values=top_k_values),
        )
        return report.to_dict()

    def _pipeline_ask_with_contexts(self, query: str, contexts) -> dict:
        """复用 pipeline 的生成链路，并返回完整证据片段。

        Example Input:
            app._pipeline_ask_with_contexts("问题", contexts)

        Example Output:
            {"answer": "...", "contexts": [{"preview": "完整 chunk 内容"}]}
        """
        # 复用 pipeline 的增强/生成/记忆，但用已过滤的 contexts
        from rag.interfaces import ChatMessage
        messages = self._pipeline._prompt_builder.build(
            query=query, contexts=contexts, history=self._pipeline._memory.history())
        answer = self._pipeline._generator.generate(messages)
        self._pipeline._memory.add(ChatMessage(role="user", content=query))
        self._pipeline._memory.add(ChatMessage(role="assistant", content=answer))
        return {
            "answer": answer,
            "contexts": [
                {
                    "source": c.document.metadata.get("source"),
                    "page": c.document.metadata.get("page"),
                    "heading_path": c.document.metadata.get("heading_path"),
                    "version": c.document.metadata.get("version"),
                    "score": round(c.score, 4),
                    "preview": c.document.content,
                }
                for c in contexts
            ],
        }


# --------------------------------------------------------------------------- #
# 向后兼容入口：旧测试/旧调用仍用 build_pipeline(source) 一次性建库              #
# --------------------------------------------------------------------------- #
def build_pipeline(
    source: str,
    cfg: Settings | None = None,
    *,
    providers: object | None = None,
) -> RagPipeline:
    app = RagApplication(cfg, providers=providers)
    app.index(source)
    return app._pipeline
