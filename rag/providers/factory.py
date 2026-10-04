"""通用 Provider 解析与注册入口。

RAGX 的上层只依赖 `embed(texts)` 与 `chat(messages, **kwargs)` 两条规则：

- 直接注入：调用方传入已经初始化好的 embedding/LLM provider 实例。
- 注册构建：调用方注册一组 builder，再通过 `cfg.provider` 或 provider name 构建。
- 默认兼容：未传外部 provider 时继续走共享 `agent_provider`。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

try:  # pragma: no cover - 取决于运行环境是否安装了共享包
    from agent_provider import (
        ChatMessage,
        ProviderInitError,
    )
    from agent_provider import (
        build_embedding_provider as _agent_build_embedding_provider,
    )
    from agent_provider import (
        build_llm_provider as _agent_build_llm_provider,
    )
except ImportError:  # pragma: no cover - 独立运行时的正常路径
    from rag.providers._fallback import (
        ChatMessage,
        ProviderInitError,
    )
    from rag.providers._fallback import (
        build_embedding_provider as _agent_build_embedding_provider,
    )
    from rag.providers._fallback import (
        build_llm_provider as _agent_build_llm_provider,
    )


class EmbeddingProviderLike(Protocol):
    """Embedding provider protocol used by RAGX.

    Example Input:
        provider.embed(["hello"])

    Example Output:
        [[0.1, 0.2, 0.3]]
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text.

        Example Input:
            provider.embed(["query"])

        Example Output:
            [[0.1, 0.2]]
        """
        ...


class LLMProviderLike(Protocol):
    """LLM provider protocol used by RAGX.

    Example Input:
        provider.chat([ChatMessage(role="user", content="Hi")])

    Example Output:
        "Hello"
    """

    def chat(self, messages: Sequence[ChatMessage], **kwargs: object) -> str:
        """Return a generated response.

        Example Input:
            provider.chat([ChatMessage(role="user", content="Hi")])

        Example Output:
            "Hello"
        """
        ...


EmbeddingBuilder = Callable[[Any | None], EmbeddingProviderLike]
LLMBuilder = Callable[[Any | None], LLMProviderLike]


@dataclass(frozen=True)
class ProviderFactory:
    """A named provider factory with separate embedding and LLM builders.

    Example Input:
        ProviderFactory(build_embedding, build_llm)

    Example Output:
        A reusable factory registered under a provider name.
    """

    embedding_builder: EmbeddingBuilder
    llm_builder: LLMBuilder


@dataclass(frozen=True)
class ProviderBundle:
    """Direct provider instances supplied by external callers.

    Example Input:
        ProviderBundle(embedding=my_embedding, llm=my_llm)

    Example Output:
        A bundle that can be passed to `RagApplication`.
    """

    embedding: EmbeddingProviderLike | None = None
    llm: LLMProviderLike | None = None

    def require_embedding(self) -> EmbeddingProviderLike:
        """Return the embedding provider or raise a provider init error.

        Example Input:
            ProviderBundle(embedding=provider).require_embedding()

        Example Output:
            provider
        """
        if self.embedding is None:
            raise ProviderInitError("Embedding provider is required")
        return self.embedding

    def require_llm(self) -> LLMProviderLike:
        """Return the LLM provider or raise a provider init error.

        Example Input:
            ProviderBundle(llm=provider).require_llm()

        Example Output:
            provider
        """
        if self.llm is None:
            raise ProviderInitError("LLM provider is required")
        return self.llm


_PROVIDER_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(
    name: str,
    factory: ProviderFactory | object | None = None,
    *,
    embedding_builder: EmbeddingBuilder | None = None,
    llm_builder: LLMBuilder | None = None,
    overwrite: bool = False,
) -> None:
    """Register a provider factory by name.

    Example Input:
        register_provider("local", embedding_builder=build_emb, llm_builder=build_llm)

    Example Output:
        None. Later `Settings(provider="local")` can build both providers.
    """
    normalized_name = _normalize_provider_name(name)
    if normalized_name in _PROVIDER_REGISTRY and not overwrite:
        raise ValueError(f"Provider already registered: {normalized_name}")
    _PROVIDER_REGISTRY[normalized_name] = _coerce_factory(
        factory,
        embedding_builder=embedding_builder,
        llm_builder=llm_builder,
    )


def unregister_provider(name: str) -> None:
    """Remove a registered provider factory.

    Example Input:
        unregister_provider("local")

    Example Output:
        None. Missing names are ignored.
    """
    _PROVIDER_REGISTRY.pop(_normalize_provider_name(name), None)


def list_providers() -> tuple[str, ...]:
    """Return registered provider names.

    Example Input:
        list_providers()

    Example Output:
        ("custom", "openai")
    """
    return tuple(sorted(_PROVIDER_REGISTRY))


def resolve_provider_bundle(
    cfg: Any | None = None,
    provider: object | None = None,
) -> ProviderBundle:
    """Resolve direct, mapped, named, or registered providers into a bundle.

    Example Input:
        resolve_provider_bundle(cfg, ProviderBundle(embedding=emb, llm=llm))

    Example Output:
        ProviderBundle(embedding=emb, llm=llm)
    """
    direct_bundle = _bundle_from_input(provider)
    if direct_bundle.embedding is not None and direct_bundle.llm is not None:
        return direct_bundle
    return ProviderBundle(
        embedding=direct_bundle.embedding or build_embedding_provider(cfg, provider),
        llm=direct_bundle.llm or build_llm_provider(cfg, provider),
    )


def build_embedding_provider(
    cfg: Any | None = None,
    provider: object | None = None,
) -> EmbeddingProviderLike:
    """Build or extract an embedding provider with the same external rules.

    Example Input:
        build_embedding_provider(cfg)

    Example Output:
        An object exposing `embed(texts)`.
    """
    direct = _bundle_from_input(provider).embedding
    if direct is not None:
        return direct
    factory, factory_cfg = _select_factory(cfg, provider)
    return _ensure_embedding_provider(factory.embedding_builder(factory_cfg))


def build_llm_provider(
    cfg: Any | None = None,
    provider: object | None = None,
) -> LLMProviderLike:
    """Build or extract an LLM provider with the same external rules.

    Example Input:
        build_llm_provider(cfg)

    Example Output:
        An object exposing `chat(messages, **kwargs)`.
    """
    direct = _bundle_from_input(provider).llm
    if direct is not None:
        return direct
    factory, factory_cfg = _select_factory(cfg, provider)
    return _ensure_llm_provider(factory.llm_builder(factory_cfg))


def _coerce_factory(
    factory: ProviderFactory | object | None,
    *,
    embedding_builder: EmbeddingBuilder | None,
    llm_builder: LLMBuilder | None,
) -> ProviderFactory:
    """Convert a factory object or builder pair into `ProviderFactory`.

    Example Input:
        _coerce_factory(obj, embedding_builder=None, llm_builder=None)

    Example Output:
        ProviderFactory(...)
    """
    if isinstance(factory, ProviderFactory):
        return factory
    resolved_embedding_builder = embedding_builder
    resolved_llm_builder = llm_builder
    if factory is not None:
        resolved_embedding_builder = resolved_embedding_builder or getattr(
            factory,
            "build_embedding_provider",
            None,
        )
        resolved_llm_builder = resolved_llm_builder or getattr(
            factory,
            "build_llm_provider",
            None,
        )
    if not callable(resolved_embedding_builder) or not callable(resolved_llm_builder):
        raise TypeError(
            "Provider factory must define build_embedding_provider(cfg) and "
            "build_llm_provider(cfg), or receive both builder callables"
        )
    return ProviderFactory(
        embedding_builder=resolved_embedding_builder,
        llm_builder=resolved_llm_builder,
    )


def _bundle_from_input(provider: object | None) -> ProviderBundle:
    """Extract direct provider instances from supported external input shapes.

    Example Input:
        _bundle_from_input({"embedding": emb, "llm": llm})

    Example Output:
        ProviderBundle(embedding=emb, llm=llm)
    """
    if provider is None or isinstance(provider, str):
        return ProviderBundle()
    if isinstance(provider, ProviderBundle):
        return ProviderBundle(
            embedding=_ensure_optional_embedding_provider(provider.embedding),
            llm=_ensure_optional_llm_provider(provider.llm),
        )
    if isinstance(provider, Mapping):
        return ProviderBundle(
            embedding=_ensure_optional_embedding_provider(
                _first_mapping_value(
                    provider,
                    ("embedding", "embedding_provider"),
                )
            ),
            llm=_ensure_optional_llm_provider(
                _first_mapping_value(provider, ("llm", "llm_provider"))
            ),
        )
    if _is_factory_object(provider):
        return ProviderBundle()
    return ProviderBundle(
        embedding=(
            _ensure_embedding_provider(provider)
            if _is_embedding_provider(provider) else None
        ),
        llm=_ensure_llm_provider(provider) if _is_llm_provider(provider) else None,
    )


def _select_factory(
    cfg: Any | None,
    provider: object | None,
) -> tuple[ProviderFactory, object]:
    """Select a provider factory and config for builder execution.

    Example Input:
        _select_factory(cfg, "openai")

    Example Output:
        (ProviderFactory(...), cfg_with_provider_name)
    """
    if isinstance(provider, Mapping):
        factory = provider.get("factory")
        if factory is not None:
            return _coerce_factory(
                factory,
                embedding_builder=None,
                llm_builder=None,
            ), cfg
        if provider.get("embedding_builder") or provider.get("llm_builder"):
            return _coerce_factory(
                None,
                embedding_builder=provider.get("embedding_builder"),
                llm_builder=provider.get("llm_builder"),
            ), cfg
        name = provider.get("name") or provider.get("provider")
        if name:
            return _factory_by_name(str(name)), _with_provider_name(cfg, str(name))

    if provider is not None and _is_factory_object(provider):
        return _coerce_factory(
            provider,
            embedding_builder=None,
            llm_builder=None,
        ), cfg

    if isinstance(provider, str):
        return _factory_by_name(provider), _with_provider_name(cfg, provider)

    name = str(getattr(cfg, "provider", "") or "openai")
    return _factory_by_name(name), cfg


def _factory_by_name(name: str) -> ProviderFactory:
    """Read a provider factory by normalized name.

    Example Input:
        _factory_by_name("openai")

    Example Output:
        ProviderFactory(...)
    """
    normalized_name = _normalize_provider_name(name)
    try:
        return _PROVIDER_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(f"未知的 provider: {normalized_name}") from exc


def _normalize_provider_name(name: str) -> str:
    """Normalize provider registry keys.

    Example Input:
        _normalize_provider_name(" OpenAI ")

    Example Output:
        "openai"
    """
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("Provider name must not be empty")
    return normalized


def _first_mapping_value(mapping: Mapping[str, object], keys: tuple[str, ...]) -> object:
    """Return the first non-None mapping value for candidate keys.

    Example Input:
        _first_mapping_value({"llm": provider}, ("llm", "llm_provider"))

    Example Output:
        provider
    """
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _with_provider_name(cfg: Any | None, name: str) -> object:
    """Wrap config so registered builders see the selected provider name.

    Example Input:
        _with_provider_name(cfg, "custom")

    Example Output:
        A proxy whose `provider` attribute is "custom".
    """
    return _ProviderNameOverride(cfg, _normalize_provider_name(name))


class _ProviderNameOverride:
    """Config proxy that overrides only the provider name."""

    def __init__(self, cfg: Any | None, provider: str) -> None:
        self._cfg = cfg
        self.provider = provider

    def __getattr__(self, name: str) -> object:
        if self._cfg is None:
            raise AttributeError(name)
        return getattr(self._cfg, name)


def _is_factory_object(value: object) -> bool:
    """Return whether an object follows the provider factory rule.

    Example Input:
        _is_factory_object(factory)

    Example Output:
        True
    """
    return callable(getattr(value, "build_embedding_provider", None)) or callable(
        getattr(value, "build_llm_provider", None)
    )


def _is_embedding_provider(value: object) -> bool:
    """Return whether an object follows the embedding provider rule.

    Example Input:
        _is_embedding_provider(provider)

    Example Output:
        True
    """
    return callable(getattr(value, "embed", None))


def _is_llm_provider(value: object) -> bool:
    """Return whether an object follows the LLM provider rule.

    Example Input:
        _is_llm_provider(provider)

    Example Output:
        True
    """
    return callable(getattr(value, "chat", None))


def _ensure_optional_embedding_provider(
    provider: object | None,
) -> EmbeddingProviderLike | None:
    """Validate an optional embedding provider.

    Example Input:
        _ensure_optional_embedding_provider(provider)

    Example Output:
        provider
    """
    return None if provider is None else _ensure_embedding_provider(provider)


def _ensure_optional_llm_provider(
    provider: object | None,
) -> LLMProviderLike | None:
    """Validate an optional LLM provider.

    Example Input:
        _ensure_optional_llm_provider(provider)

    Example Output:
        provider
    """
    return None if provider is None else _ensure_llm_provider(provider)


def _ensure_embedding_provider(provider: object) -> EmbeddingProviderLike:
    """Validate the embedding provider rule.

    Example Input:
        _ensure_embedding_provider(provider)

    Example Output:
        provider
    """
    if not _is_embedding_provider(provider):
        raise TypeError("Embedding provider must expose embed(texts)")
    return provider


def _ensure_llm_provider(provider: object) -> LLMProviderLike:
    """Validate the LLM provider rule.

    Example Input:
        _ensure_llm_provider(provider)

    Example Output:
        provider
    """
    if not _is_llm_provider(provider):
        raise TypeError("LLM provider must expose chat(messages, **kwargs)")
    return provider


def _build_agent_embedding_provider(cfg: Any | None) -> EmbeddingProviderLike:
    """Build the shared default embedding provider.

    Example Input:
        _build_agent_embedding_provider(cfg)

    Example Output:
        OpenAI-compatible embedding provider.
    """
    return _agent_build_embedding_provider(cfg)


def _build_agent_llm_provider(cfg: Any | None) -> LLMProviderLike:
    """Build the shared default LLM provider.

    Example Input:
        _build_agent_llm_provider(cfg)

    Example Output:
        OpenAI-compatible LLM provider.
    """
    return _agent_build_llm_provider(cfg)


def _build_mock_embedding_provider(cfg: Any | None) -> EmbeddingProviderLike:
    """Build the deterministic local embedding provider used by tests.

    Example Input:
        _build_mock_embedding_provider(Settings(embedding_dim=64))

    Example Output:
        MockEmbeddingProvider(dim=64)
    """
    from rag.providers._fallback import MockEmbeddingProvider

    dim = int(getattr(cfg, "embedding_dim", 0) or 0)
    return MockEmbeddingProvider(dim=dim if dim > 0 else 256)


def _build_mock_llm_provider(cfg: Any | None) -> LLMProviderLike:
    """Build the deterministic local LLM provider used by tests.

    Example Input:
        _build_mock_llm_provider(cfg)

    Example Output:
        MockLLMProvider()
    """
    from rag.providers._fallback import MockLLMProvider

    return MockLLMProvider()


register_provider(
    "openai",
    embedding_builder=_build_agent_embedding_provider,
    llm_builder=_build_agent_llm_provider,
)
register_provider(
    "custom",
    embedding_builder=_build_agent_embedding_provider,
    llm_builder=_build_agent_llm_provider,
)
register_provider(
    "mock",
    embedding_builder=_build_mock_embedding_provider,
    llm_builder=_build_mock_llm_provider,
)


__all__ = [
    "EmbeddingBuilder",
    "EmbeddingProviderLike",
    "LLMBuilder",
    "LLMProviderLike",
    "ProviderBundle",
    "ProviderFactory",
    "build_embedding_provider",
    "build_llm_provider",
    "list_providers",
    "register_provider",
    "resolve_provider_bundle",
    "unregister_provider",
]
