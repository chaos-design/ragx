"""Provider 抽象层 —— 统一大模型接入接口。

优先使用共享 `agent_provider` 包；该包缺失时自动降级到
`rag.providers._fallback` 中的本地最小实现，保证本仓库可独立导入与测试。
"""

try:  # pragma: no cover - 取决于运行环境是否安装了共享包
    from agent_provider import (
        ChatMessage,
        CustomProvider,
        EmbeddingProvider,
        LLMProvider,
        ProviderInitError,
        Settings,
        get_settings,
    )

    AGENT_PROVIDER_AVAILABLE = True
except ImportError:  # pragma: no cover - 独立运行时的正常路径
    from rag.providers._fallback import (
        ChatMessage,
        CustomProvider,
        EmbeddingProvider,
        LLMProvider,
        ProviderInitError,
        Settings,
        get_settings,
    )

    AGENT_PROVIDER_AVAILABLE = False

from rag.providers.factory import (
    EmbeddingBuilder,
    EmbeddingProviderLike,
    LLMBuilder,
    LLMProviderLike,
    ProviderBundle,
    ProviderFactory,
    build_embedding_provider,
    build_llm_provider,
    list_providers,
    register_provider,
    resolve_provider_bundle,
    unregister_provider,
)

__all__ = [
    "AGENT_PROVIDER_AVAILABLE",
    "ChatMessage",
    "CustomProvider",
    "EmbeddingBuilder",
    "EmbeddingProvider",
    "EmbeddingProviderLike",
    "LLMBuilder",
    "LLMProvider",
    "LLMProviderLike",
    "ProviderBundle",
    "ProviderFactory",
    "ProviderInitError",
    "Settings",
    "build_embedding_provider",
    "build_llm_provider",
    "get_settings",
    "list_providers",
    "register_provider",
    "resolve_provider_bundle",
    "unregister_provider",
]