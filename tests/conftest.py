"""Shared test doubles for RAGX tests."""

from __future__ import annotations

import pytest


class FakeEmbeddingProvider:
    """Deterministic embedding provider for local tests."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
            ascii_words = sum(1 for char in text if char.isascii() and char.isalnum())
            checksum = sum(ord(char) for char in text) % 997
            vectors.append(
                [float(len(text) or 1), float(cjk), float(ascii_words), float(checksum)]
            )
        return vectors


class FakeLLMProvider:
    """Deterministic chat provider for local tests."""

    def chat(self, messages, **kwargs):
        latest = next(
            (
                message.content
                for message in reversed(messages)
                if getattr(message, "role", "") == "user"
            ),
            "",
        )
        return f"(test) {latest}"


@pytest.fixture(autouse=True)
def patch_rag_app_providers(monkeypatch):
    """Patch RagApplication provider builders for tests that do not override them."""
    try:
        from rag import app as app_module
    except ImportError:
        return
    monkeypatch.setattr(
        app_module,
        "build_embedding_provider",
        lambda cfg: FakeEmbeddingProvider(),
    )
    monkeypatch.setattr(app_module, "build_llm_provider", lambda cfg: FakeLLMProvider())
