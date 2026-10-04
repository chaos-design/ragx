"""OpenAI-compatible Provider compatibility exports.

真实实现来自共享 `agent_provider` 包。该包缺失时，本模块不再直接抛出
`ImportError`，而是暴露 `AVAILABLE = False` 并让调用方在**实际使用**时
得到清晰提示——这样 `import rag.providers.openai_provider` 不会阻断
整个包的导入（例如测试收集、`--help`、lint）。
"""

try:  # pragma: no cover - 取决于运行环境
    from agent_provider import (
        AzureOpenAIEmbeddingProvider,
        AzureOpenAILLMProvider,
        OpenAIEmbeddingProvider,
        OpenAILLMProvider,
    )

    AVAILABLE = True
except ImportError:  # pragma: no cover - 独立运行时的正常路径
    AVAILABLE = False

    class _MissingOpenAIProvider:
        """占位类型：实例化时给出可执行的修复指引。"""

        _label = "OpenAI-compatible provider"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                f"{self._label} 需要共享 agent_provider 包。"
                "请安装 agent_provider，或改用注入式 provider："
                "RagApplication(cfg, providers=ProviderBundle(embedding=..., llm=...))"
            )

    class OpenAIEmbeddingProvider(_MissingOpenAIProvider):
        _label = "OpenAIEmbeddingProvider"

    class OpenAILLMProvider(_MissingOpenAIProvider):
        _label = "OpenAILLMProvider"

    class AzureOpenAIEmbeddingProvider(_MissingOpenAIProvider):
        _label = "AzureOpenAIEmbeddingProvider"

    class AzureOpenAILLMProvider(_MissingOpenAIProvider):
        _label = "AzureOpenAILLMProvider"


__all__ = [
    "AVAILABLE",
    "AzureOpenAIEmbeddingProvider",
    "AzureOpenAILLMProvider",
    "OpenAIEmbeddingProvider",
    "OpenAILLMProvider",
]