"""本地 Provider 兜底实现 —— 让 ragx 无外部依赖即可导入与测试。

背景：
    历史上 RAGX 的 Provider 抽象全部来自同级目录的共享 `agent_provider` 包。
    这导致两个问题：
      1. 单独 clone 本仓库后无法运行（`ModuleNotFoundError: agent_provider`）；
      2. 单元测试必须先安装共享包才能收集。

本模块提供**行为等价**的最小实现，覆盖 RAGX 实际使用到的契约：
      • ChatMessage            —— 对话消息 DTO
      • EmbeddingProvider      —— embed(texts) -> list[list[float]]
      • LLMProvider            —— chat(messages, **kwargs) -> str
      • ProviderInitError      —— provider 初始化/能力校验失败
      • Settings / get_settings—— .env + 环境变量解析
      • Mock*Provider          —— 确定性本地实现，仅供测试与离线验证

与真实 provider 的关系：
    若安装了 `agent_provider`，RAGX 仍优先使用共享包的真实实现；
    本模块只在共享包缺失时作为 fallback 生效。真实 provider 的能力**严格更强**
    （支持 Azure、真实 API 调用、Responses 协议等），fallback 不做模拟。

Example:
    >>> from rag.providers import ChatMessage
    >>> ChatMessage(role="user", content="hi").content
    'hi'
"""
from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ChatMessage",
    "CustomProvider",
    "EmbeddingProvider",
    "LLMProvider",
    "MockEmbeddingProvider",
    "MockLLMProvider",
    "ProviderInitError",
    "Settings",
    "build_embedding_provider",
    "build_llm_provider",
    "get_settings",
]


class ProviderInitError(RuntimeError):
    """Provider 初始化或能力校验失败。"""


@dataclass(frozen=True)
class ChatMessage:
    """单条对话消息。

    Example Input:
        ChatMessage(role="user", content="报销时限是多久？")

    Example Output:
        ChatMessage(role='user', content='报销时限是多久？')
    """

    role: str
    content: str


@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本向量化契约。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text.

        Example Input:
            provider.embed(["hello"])

        Example Output:
            [[0.1, 0.2, 0.3]]
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """对话生成契约。"""

    def chat(self, messages: Sequence[ChatMessage], **kwargs: object) -> str:
        """Return a generated response.

        Example Input:
            provider.chat([ChatMessage(role="user", content="Hi")])

        Example Output:
            "Hello"
        """
        ...


# --------------------------------------------------------------------------- #
# 确定性本地实现：仅供测试与离线验证，不调用任何外部 API
# --------------------------------------------------------------------------- #
class _HashEmbedding:
    """确定性哈希向量化：相同文本恒定得到相同向量。

    用途是让检索链路在无 API Key、无模型的环境下可端到端验证；
    它**不具备语义能力**，不能作为生产 embedding 方案。
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = max(2, int(dim))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self._dim:
            payload = f"{text}:{counter}".encode()
            digest = hashlib.blake2b(payload, digest_size=32).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        vector = values[: self._dim]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class MockEmbeddingProvider:
    """确定性本地 embedding provider（测试/离线用）。"""

    def __init__(self, dim: int = 256) -> None:
        self._impl = _HashEmbedding(dim)
        self.dim = self._impl._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._impl.embed(texts)


class MockLLMProvider:
    """确定性本地 LLM provider（测试/离线用）。

    输出是固定格式的占位文本，**不是真实答案**，用于验证链路连通性。
    """

    def chat(self, messages: Sequence[ChatMessage], **kwargs: object) -> str:
        last = ""
        for message in messages:
            if message.role == "user":
                last = message.content
        if "【问题】" in last:
            last = last.split("【问题】", 1)[1].strip()
        return f"[mock-llm] 已收到 {len(messages)} 条消息，问题：{last}"


class CustomProvider:
    """由调用方注入的 provider 容器（占位实现）。

    Example Input:
        CustomProvider(embedding=my_embedding, llm=my_llm)
    """

    def __init__(self, embedding: Any = None, llm: Any = None) -> None:
        self.embedding = embedding
        self.llm = llm


# --------------------------------------------------------------------------- #
# 配置解析：与真实 agent_provider.get_settings 保持字段兼容
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """Provider 侧配置（字段与共享 agent_provider.Settings 对齐）。"""

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


def _read_env_file(path: Path) -> dict[str, str]:
    """解析单个 .env 文件（忽略注释与空行）。

    Example Input:
        _read_env_file(Path(".env"))

    Example Output:
        {"OPENAI_API_KEY": "sk-xxx"}
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            if value:
                values[key.strip()] = value
    except OSError:
        return {}
    return values


def get_settings(
    env_files: Sequence[Path] = (),
    default_provider: str = "openai",
) -> Settings:
    """加载 provider 配置：环境变量优先于 .env 文件。

    Example Input:
        get_settings(env_files=[Path(".env")], default_provider="openai")

    Example Output:
        Settings(provider="openai", llm_model="gpt-4o-mini", ...)
    """
    file_values: dict[str, str] = {}
    for env_file in env_files:
        file_values.update(_read_env_file(Path(env_file)))

    def pick(*names: str, default: str = "") -> str:
        for name in names:
            value = os.getenv(name)
            if value not in (None, ""):
                return value
        for name in names:
            value = file_values.get(name)
            if value not in (None, ""):
                return value
        return default

    def pick_int(*names: str, default: int) -> int:
        raw = pick(*names)
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    def pick_float(*names: str, default: float) -> float:
        raw = pick(*names)
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    embedding_model = pick("RAG_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")
    if "text-embedding-3-large" in embedding_model.lower():
        default_dim = 3072
    else:
        default_dim = 1536

    return Settings(
        provider=pick("RAG_PROVIDER", "OPENAI_PROVIDER", default=default_provider),
        api_key=pick("OPENAI_API_KEY", "RAG_API_KEY"),
        base_url=pick("RAG_BASE_URL", "OPENAI_BASE_URL") or None,
        endpoint=pick("RAG_ENDPOINT", "OPENAI_ENDPOINT", "RAG_OPENAI_ENDPOINT") or None,
        api_version=pick("RAG_API_VERSION", "OPENAI_API_VERSION", default="2024-06-01"),
        embedding_api_version=pick(
            "RAG_EMBEDDING_API_VERSION", "OPENAI_EMBEDDING_API_VERSION",
            default="2024-06-01",
        ),
        llm_model=pick("RAG_LLM_MODEL", "OPENAI_LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini"),
        embedding_model=embedding_model,
        embedding_base_url=pick("RAG_EMBEDDING_BASE_URL", "OPENAI_EMBEDDING_BASE_URL") or None,
        embedding_endpoint=pick(
            "RAG_EMBEDDING_ENDPOINT", "OPENAI_EMBEDDING_ENDPOINT",
            "RAG_OPENAI_EMBEDDING_ENDPOINT",
        ) or None,
        embedding_dim=pick_int("RAG_EMBEDDING_DIM", default=default_dim),
        openai_api_mode=pick("RAG_OPENAI_API_MODE", "OPENAI_API_MODE", default="auto"),
    )


def build_embedding_provider(cfg: Any = None) -> EmbeddingProvider:
    """构建真实 embedding provider。

    fallback 无法调用真实 API，因此**任何情况下都抛错**，绝不返回本地桩。
    理由：静默返回假向量会污染持久化索引，且污染在离线状态下不可见——
    等到接入真实模型时才发现索引不可用，代价远高于启动即失败。

    需要离线跑通链路时，请显式声明 `RAG_PROVIDER=mock`（走 factory 注册的
    mock 工厂，与本函数不同路径），意图明确、不会误伤生产配置。

    Example Input:
        build_embedding_provider(Settings(api_key=""))

    Example Output:
        ProviderInitError: 无法构建真实 embedding provider...
    """
    raise ProviderInitError(
        _no_real_provider_message("embedding", cfg)
    )


def _no_real_provider_message(kind: str, cfg: Any = None) -> str:
    """构造「无法构建真实 provider」的可执行错误信息。

    Example Input:
        _no_real_provider_message("llm", cfg)

    Example Output:
        "无法构建真实 llm provider：缺少 OPENAI_API_KEY ..."
    """
    api_key = getattr(cfg, "api_key", "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("RAG_API_KEY", "")
    reason = "缺少 OPENAI_API_KEY" if not api_key else "缺少 agent_provider 共享包，无法调用真实 API"
    return (
        f"无法构建真实 {kind} provider：{reason}。\n"
        "可选路径：\n"
        "  1. 配置凭据后重试：export OPENAI_API_KEY=sk-...\n"
        "  2. 安装共享包以获得完整 provider 能力：pip install agent_provider\n"
        "  3. 仅离线验证链路连通性（产物不可用于生产），需同时设置两个开关：\n"
        "       export RAG_PROVIDER=mock\n"
        "       export RAG_ALLOW_MOCK_PROVIDER=1\n"
        "  4. 注入自定义 provider："
        "RagApplication(cfg, providers=ProviderBundle(embedding=..., llm=...))"
    )


def build_llm_provider(cfg: Any = None) -> LLMProvider:
    """构建真实 LLM provider（语义同 build_embedding_provider，一律抛错）。

    静默返回桩 LLM 是最危险的一种降级：检索链路真实工作，生成环节返回
    固定占位文本，最终表现为一篇「看起来正常但内容全错」的答案。

    Example Input:
        build_llm_provider(Settings(api_key=""))

    Example Output:
        ProviderInitError: 无法构建真实 llm provider...
    """
    raise ProviderInitError(_no_real_provider_message("llm", cfg))