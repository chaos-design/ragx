"""Provider factory tests for direct injection and registered builders."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from rag.app import RagApplication
from rag.providers import (
    ProviderBundle,
    build_embedding_provider,
    build_llm_provider,
    list_providers,
    register_provider,
    resolve_provider_bundle,
    unregister_provider,
)


class DirectEmbeddingProvider:
    """Embedding provider used to verify direct injection."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        """Return deterministic vectors for each input text.

        Example Input:
            provider.embed(["hello"])

        Example Output:
            [[5.0, 1.0, 0.0]]
        """
        self.calls.append(list(texts))
        return [[float(len(text) or 1), 1.0, 0.0] for text in texts]


class DirectLLMProvider:
    """LLM provider used to verify direct injection."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kwargs):
        """Return a stable answer derived from the latest user message.

        Example Input:
            provider.chat([ChatMessage(role="user", content="Hi")])

        Example Output:
            "external: Hi"
        """
        self.calls += 1
        latest = next(
            (
                message.content
                for message in reversed(messages)
                if getattr(message, "role", "") == "user"
            ),
            "",
        )
        return f"external: {latest}"


class TrackingProviderFactory:
    """Factory object that records the selected provider name."""

    def __init__(self) -> None:
        self.embedding_cfg_provider = ""
        self.llm_cfg_provider = ""
        self.embedding = DirectEmbeddingProvider()
        self.llm = DirectLLMProvider()

    def build_embedding_provider(self, cfg):
        """Build a deterministic embedding provider.

        Example Input:
            factory.build_embedding_provider(cfg)

        Example Output:
            DirectEmbeddingProvider()
        """
        self.embedding_cfg_provider = cfg.provider
        return self.embedding

    def build_llm_provider(self, cfg):
        """Build a deterministic LLM provider.

        Example Input:
            factory.build_llm_provider(cfg)

        Example Output:
            DirectLLMProvider()
        """
        self.llm_cfg_provider = cfg.provider
        return self.llm


def test_rag_application_accepts_direct_provider_bundle(tmp_path):
    """验证 RagApplication 可直接接收外部 provider 实例。"""
    source = tmp_path / "resources"
    source.mkdir()
    (source / "rule.md").write_text(
        "# 制度\n\n## 报销\n报销材料需要在 3 天内提交，并上传发票、审批单和付款证明。"
        "财务部门会根据最新制度完成校验并反馈处理结果，请务必在规定时间内"
        "完成提交流程，避免影响费用审核。\n",
        encoding="utf-8",
    )
    embedding = DirectEmbeddingProvider()
    llm = DirectLLMProvider()
    app = RagApplication(
        Settings(
            provider="unregistered-external",
            vector_backend="sqlite",
            sqlite_path=str(tmp_path / "store.sqlite"),
            manifest_path="",
        ),
        providers=ProviderBundle(embedding=embedding, llm=llm),
    )

    report = app.index(str(source))
    result = app.ask("报销材料几天内提交？")

    assert report.added == 1
    assert embedding.calls
    assert llm.calls == 1
    assert result["answer"].startswith("external:")
    assert result["contexts"]


def test_provider_registry_builds_from_cfg_provider_name():
    """验证 registry 可按 cfg.provider 构建外部 provider。"""
    factory = TrackingProviderFactory()
    register_provider("external-test", factory, overwrite=True)
    try:
        cfg = Settings(provider="external-test")

        embedding = build_embedding_provider(cfg)
        llm = build_llm_provider(cfg)

        assert embedding is factory.embedding
        assert llm is factory.llm
        assert factory.embedding_cfg_provider == "external-test"
        assert factory.llm_cfg_provider == "external-test"
        assert "external-test" in list_providers()
    finally:
        unregister_provider("external-test")


def test_provider_registry_can_be_selected_by_explicit_name():
    """验证显式 provider name 会覆盖 cfg.provider 选择。"""
    factory = TrackingProviderFactory()
    register_provider("explicit-external", factory, overwrite=True)
    try:
        cfg = Settings(provider="openai", api_key="sk-test")

        bundle = resolve_provider_bundle(cfg, "explicit-external")

        assert bundle.embedding is factory.embedding
        assert bundle.llm is factory.llm
        assert factory.embedding_cfg_provider == "explicit-external"
        assert factory.llm_cfg_provider == "explicit-external"
    finally:
        unregister_provider("explicit-external")


def test_provider_mapping_accepts_direct_instances():
    """验证 mapping 形式可直接注入外部 provider 实例。"""
    embedding = DirectEmbeddingProvider()
    llm = DirectLLMProvider()

    bundle = resolve_provider_bundle(
        Settings(provider="openai", api_key="sk-test"),
        {"embedding_provider": embedding, "llm_provider": llm},
    )

    assert bundle.embedding is embedding
    assert bundle.llm is llm


def test_provider_mapping_validates_direct_provider_rules():
    """验证外部传入对象必须满足统一 provider 规则。"""
    with pytest.raises(TypeError, match="Embedding provider must expose embed"):
        resolve_provider_bundle(
            Settings(provider="openai", api_key="sk-test"),
            {"embedding": object(), "llm": DirectLLMProvider()},
        )


def test_register_provider_rejects_incomplete_factory():
    """验证注册 provider 时必须同时提供 embedding 与 LLM builder。"""
    with pytest.raises(TypeError, match="build_embedding_provider"):
        register_provider("broken-external", embedding_builder=lambda cfg: object())
