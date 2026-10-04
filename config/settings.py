"""集中配置 —— 一处定义，处处注入。

通过环境变量覆盖，便于在不同环境切换厂商而不改代码。
Provider 配置直接复用共享 agent_provider 的解析能力；RAGX 只保留索引、
分块、向量库等应用侧配置。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

AGENT_LIBRARY_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_LIBRARY_ROOT))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_DIR = PROJECT_ROOT / "data" / "stores"
DEFAULT_SQLITE_PATH = DEFAULT_STORE_DIR / "ragx-store.sqlite"
DEFAULT_MANIFEST_PATH = DEFAULT_STORE_DIR / "ragx-manifest.json"


def _first_env(names: tuple[str, ...], default: str = "") -> str:
    """读取首个非空环境变量。

    输入 (Input):
        names: 候选环境变量名，按优先级排序。
        default: 没有命中时的默认值。

    输出 (Output):
        环境变量字符串值或默认值。

    示例 (Example):
        _first_env(("OPENAI_API_KEY",), "")
    """
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量。

    输入 (Input):
        name: 环境变量名。
        default: 解析失败或未配置时的默认值。

    输出 (Output):
        解析后的整数。

    示例 (Example):
        _int_env("RAG_TOP_K", 4)
    """
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    """读取浮点数环境变量。

    输入 (Input):
        name: 环境变量名。
        default: 解析失败或未配置时的默认值。

    输出 (Output):
        解析后的浮点数。

    示例 (Example):
        _float_env("RAG_COLBERT_INTERACTION_WEIGHT", 0.95)
    """
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。

    输入 (Input):
        name: 环境变量名。
        default: 解析失败或未配置时的默认值。

    输出 (Output):
        布尔值。

    示例 (Example):
        _bool_env("RAG_PARENT_CHILD", False)
    """
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_embedding_dim(
    provider: str,
    embedding_model: str = "text-embedding-3-small",
) -> int:
    """按 provider 推导默认向量维度。

    输入 (Input):
        provider: provider 名称。
        embedding_model: embedding 模型名称。

    输出 (Output):
        默认 embedding 维度；mock 为 256，text-embedding-3-large 为 3072。

    示例 (Example):
        _default_embedding_dim("openai", "text-embedding-3-large")
    """
    if provider.lower() == "mock":
        return 256
    if "text-embedding-3-large" in embedding_model.lower():
        return 3072
    return 1536


def normalize_search_backend(value: str | None) -> str:
    """规范化候选搜索后端名称。

    输入 (Input):
        value: 配置中的搜索后端名称。

    输出 (Output):
        当前支持的 `bi_encoder`。

    示例 (Example):
        normalize_search_backend("bi-encoder")
    """
    normalized = (value or "bi_encoder").strip().lower().replace("-", "_")
    aliases = {"bi_encoder", "biencoder"}
    if normalized not in aliases:
        raise ValueError(
            "Unsupported search backend: "
            f"{value}. Expected bi_encoder."
        )
    return "bi_encoder"


@dataclass
class Settings:
    """RAGX 应用运行配置。

    输入 (Input):
        provider: agent_provider 支持的 provider 名称，如 openai/custom/mock。
        api_key: 真实 provider API Key。
        base_url: OpenAI 兼容 API Base URL。
        endpoint: OPENAI_ENDPOINT/RAG_OPENAI_ENDPOINT，支持 /responses 完整端点。
        api_version: Azure/OpenAI 兼容 API version。
        embedding_api_version: Azure/OpenAI 兼容 embedding API version。
        llm_model: 对话模型名称。
        embedding_model: 向量模型名称。
        embedding_base_url: 独立 embedding API Base URL。
        embedding_endpoint: 独立 embedding endpoint，支持 /embeddings 完整端点。
        embedding_dim: 向量维度，用于 pgvector 等强 schema 后端。
        openai_api_mode: 调用协议，支持 auto/chat/responses。
        retrieval_backend: 检索后端，生产固定为 hybrid。
        search_backend: 默认候选搜索策略，目前为 bi_encoder。
        reranker_backend: 候选重排策略，支持 cross_encoder/colbert/none。
        cross_encoder_weight: Cross-Encoder 分数权重。
        cross_encoder_retrieval_score_weight: 原召回分数在 Cross-Encoder 中的稳定项权重。
        colbert_model: 可选 HuggingFace ColBERT/Transformer 模型名。
        colbert_query_max_tokens: ColBERT 查询 token 上限。
        colbert_document_max_tokens: ColBERT 文档 token 上限。
        colbert_batch_size: ColBERT 模型编码 batch size。
        colbert_interaction_weight: ColBERT MaxSim 交互分权重。
        colbert_retrieval_score_weight: 原召回分数在 ColBERT 中的稳定项权重。
        colbert_hashing_dim: 无外部模型时本地 token 向量维度。
        top_k: 检索召回数量。
        chunk_size: 正文分块目标大小。
        chunk_overlap: 正文分块 overlap。
        vector_backend: 向量库后端，生产支持 sqlite/pgvector。
        sqlite_path: SQLite 向量库路径。
        pg_dsn: pgvector DSN。
        manifest_path: 增量同步 manifest 路径。
        parent_child: 是否启用父子分块。

    输出 (Output):
        可注入 RagApplication 的配置对象。

    示例 (Example):
        Settings(provider="openai", vector_backend="sqlite")
    """

    provider: str = "openai"
    api_key: str = ""
    base_url: str | None = None
    endpoint: str | None = None
    api_version: str = "2024-06-01"
    embedding_api_version: str = "2024-06-01"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_endpoint: str | None = None
    embedding_dim: int = 1536
    openai_api_mode: str = "auto"
    retrieval_backend: str = "hybrid"
    search_backend: str = "bi_encoder"
    reranker_backend: str = "cross_encoder"
    cross_encoder_weight: float = 0.95
    cross_encoder_retrieval_score_weight: float = 0.05
    colbert_model: str = ""
    colbert_query_max_tokens: int = 32
    colbert_document_max_tokens: int = 180
    colbert_batch_size: int = 8
    colbert_interaction_weight: float = 0.95
    colbert_retrieval_score_weight: float = 0.05
    colbert_hashing_dim: int = 64
    top_k: int = 4
    chunk_size: int = 300
    chunk_overlap: int = 50
    # 向量库后端: "sqlite"(默认，真实持久化) | "pgvector"
    vector_backend: str = "sqlite"
    sqlite_path: str = str(DEFAULT_SQLITE_PATH)
    pg_dsn: str = os.getenv("RAG_PG_DSN", "")
    # 增量同步状态表落盘路径：默认持久化，跨进程增量更新生效
    manifest_path: str = str(DEFAULT_MANIFEST_PATH)
    # 是否启用父子分块(small-to-big)
    parent_child: bool = False


def load_settings() -> Settings:
    """加载完整 RAGX 配置。

    输入 (Input):
        None。配置来源为 agent_provider 支持的 .env/环境变量，以及 RAGX
        应用侧环境变量。

    输出 (Output):
        Settings 实例。

    示例 (Example):
        cfg = load_settings()
    """
    try:  # 优先使用共享包；缺失时降级到本地 fallback
        from agent_provider import get_settings as get_provider_settings
    except ImportError:
        from rag.providers._fallback import get_settings as get_provider_settings

    provider_settings = get_provider_settings(
        env_files=_provider_env_files(),
        default_provider="openai",
    )
    return Settings(
        provider=provider_settings.provider,
        api_key=provider_settings.api_key or "",
        base_url=provider_settings.base_url,
        endpoint=provider_settings.endpoint,
        api_version=provider_settings.api_version,
        embedding_api_version=provider_settings.embedding_api_version,
        llm_model=provider_settings.llm_model,
        embedding_model=provider_settings.embedding_model,
        embedding_base_url=provider_settings.embedding_base_url,
        embedding_endpoint=provider_settings.embedding_endpoint,
        embedding_dim=provider_settings.embedding_dim,
        openai_api_mode=provider_settings.openai_api_mode,
        retrieval_backend=_first_env(("RAG_RETRIEVAL_BACKEND",), "hybrid"),
        search_backend=_first_env(("RAG_SEARCH_BACKEND",), "bi_encoder"),
        reranker_backend=_first_env(("RAG_RERANKER_BACKEND",), "cross_encoder"),
        cross_encoder_weight=_float_env("RAG_CROSS_ENCODER_WEIGHT", 0.95),
        cross_encoder_retrieval_score_weight=_float_env(
            "RAG_CROSS_ENCODER_RETRIEVAL_SCORE_WEIGHT",
            0.05,
        ),
        colbert_model=_first_env(("RAG_COLBERT_MODEL",), ""),
        colbert_query_max_tokens=_int_env("RAG_COLBERT_QUERY_MAX_TOKENS", 32),
        colbert_document_max_tokens=_int_env("RAG_COLBERT_DOCUMENT_MAX_TOKENS", 180),
        colbert_batch_size=_int_env("RAG_COLBERT_BATCH_SIZE", 8),
        colbert_interaction_weight=_float_env("RAG_COLBERT_INTERACTION_WEIGHT", 0.95),
        colbert_retrieval_score_weight=_float_env(
            "RAG_COLBERT_RETRIEVAL_SCORE_WEIGHT",
            0.05,
        ),
        colbert_hashing_dim=_int_env("RAG_COLBERT_HASHING_DIM", 64),
        top_k=_int_env("RAG_TOP_K", 4),
        chunk_size=_int_env("RAG_CHUNK_SIZE", 300),
        chunk_overlap=_int_env("RAG_CHUNK_OVERLAP", 50),
        vector_backend=_first_env(("RAG_VECTOR_BACKEND",), "sqlite"),
        sqlite_path=_first_env(("RAG_SQLITE_PATH",), str(DEFAULT_SQLITE_PATH)),
        pg_dsn=_first_env(("RAG_PG_DSN",), ""),
        manifest_path=_first_env(("RAG_MANIFEST_PATH",), str(DEFAULT_MANIFEST_PATH)),
        parent_child=_bool_env("RAG_PARENT_CHILD", False),
    )


def _provider_env_files() -> tuple[Path, ...]:
    """返回 provider 配置文件候选列表。

    输入 (Input):
        None。

    输出 (Output):
        `.env` 候选路径。`doc-rag-ingest/.env` 作为共享默认配置，ragx
        本地 `.env` 和当前工作目录 `.env` 可覆盖它。

    示例 (Example):
        _provider_env_files()
    """
    return _unique_paths(
        (
            AGENT_LIBRARY_ROOT / "doc-rag-ingest" / ".env",
            AGENT_LIBRARY_ROOT / ".env",
            AGENT_LIBRARY_ROOT / "agent_provider" / ".env",
            PROJECT_ROOT / ".env",
            Path.cwd() / ".env",
        )
    )


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """按顺序去重路径。

    输入 (Input):
        paths: 候选路径。

    输出 (Output):
        去重后的路径元组。

    示例 (Example):
        _unique_paths((Path(".env"), Path(".env")))
    """
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        unique.append(resolved)
        seen.add(resolved)
    return tuple(unique)


def ensure_real_provider(cfg: Settings) -> None:
    """确保生产入口不会静默使用测试 provider。

    mock 是**双开关**显式 opt-in：既要把 `RAG_PROVIDER` 设为 mock，
    还要把 `RAG_ALLOW_MOCK_PROVIDER` 设为真值。两个条件缺一不可，
    这样「误把 mock 带上生产」需要两次独立疏忽，概率远低于单开关。

    输入 (Input):
        cfg: Settings 实例。

    输出 (Output):
        None。若 provider 为 mock 且未显式放行则抛 ValueError。

    示例 (Example):
        ensure_real_provider(load_settings())
    """
    if cfg.provider.lower().strip() != "mock":
        return
    if _bool_env("RAG_ALLOW_MOCK_PROVIDER", default=False):
        return
    raise ValueError(
        "生产运行禁止使用测试 provider；请设置 RAG_PROVIDER=openai/custom "
        "并提供 OPENAI_API_KEY。\n"
        "若仅用于离线验证链路连通性（产物不可用于生产），"
        "需同时设置两个开关：\n"
        "  export RAG_PROVIDER=mock\n"
        "  export RAG_ALLOW_MOCK_PROVIDER=1"
    )
