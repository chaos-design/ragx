"""Provider healthcheck script tests with fully isolated providers."""
from __future__ import annotations

from types import SimpleNamespace

from scripts import healthcheck_provider as healthcheck


class _FakeLLM:
    """Minimal LLM test double."""

    def chat(self, messages):
        """Return a deterministic answer and keep the request shape observable."""
        assert messages[-1].role == "user"
        return "在吗"


class _FakeEmbedding:
    """Minimal embedding test double."""

    def embed(self, texts):
        """Return one deterministic vector per text."""
        assert texts == ["连通性测试"]
        return [[1.0, 2.0, 3.0]]


def test_main_success_without_embedding_check(monkeypatch, capsys):
    """目标：LLM 连通成功且未启用 embedding 时返回 0 并打印成功摘要。"""
    cfg = SimpleNamespace(
        provider="openai",
        llm_model="chat-model",
        embedding_model="embed-model",
        base_url="https://example.test/v1",
    )
    calls = {"ensure": 0, "llm": 0, "embedding": 0}

    monkeypatch.setattr(healthcheck, "load_settings", lambda: cfg)
    monkeypatch.setattr(
        healthcheck,
        "ensure_real_provider",
        lambda received: calls.__setitem__("ensure", calls["ensure"] + 1),
    )
    monkeypatch.setattr(
        healthcheck,
        "build_llm_provider",
        lambda received: calls.__setitem__("llm", calls["llm"] + 1) or _FakeLLM(),
    )
    monkeypatch.setattr(
        healthcheck,
        "build_embedding_provider",
        lambda received: calls.__setitem__("embedding", calls["embedding"] + 1)
        or _FakeEmbedding(),
    )
    monkeypatch.setattr(healthcheck, "_check_embedding_enabled", lambda: False)

    exit_code = healthcheck.main()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == {"ensure": 1, "llm": 1, "embedding": 0}
    assert "[LLM OK]" in output
    assert "[结果] 真实大模型连通成功" in output


def test_main_success_with_embedding_check(monkeypatch, capsys):
    """目标：开启 embedding 自检时应构建 embedding provider 并报告返回维度。"""
    cfg = SimpleNamespace(
        provider="openai",
        llm_model="chat-model",
        embedding_model="embed-model",
        base_url=None,
    )
    calls = {"embedding": 0}

    monkeypatch.setattr(healthcheck, "load_settings", lambda: cfg)
    monkeypatch.setattr(healthcheck, "ensure_real_provider", lambda received: None)
    monkeypatch.setattr(healthcheck, "build_llm_provider", lambda received: _FakeLLM())
    monkeypatch.setattr(
        healthcheck,
        "build_embedding_provider",
        lambda received: calls.__setitem__("embedding", calls["embedding"] + 1)
        or _FakeEmbedding(),
    )
    monkeypatch.setattr(healthcheck, "_check_embedding_enabled", lambda: True)

    exit_code = healthcheck.main()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls["embedding"] == 1
    assert "[Embedding OK] 返回维度=3" in output


def test_main_returns_error_when_validation_fails(monkeypatch, capsys):
    """目标：真实 provider 校验失败时脚本应返回 1，不继续构建 LLM。"""
    cfg = SimpleNamespace(
        provider="mock",
        llm_model="chat-model",
        embedding_model="embed-model",
        base_url=None,
    )

    monkeypatch.setattr(healthcheck, "load_settings", lambda: cfg)
    monkeypatch.setattr(
        healthcheck,
        "ensure_real_provider",
        lambda received: (_ for _ in ()).throw(ValueError("mock disabled")),
    )

    exit_code = healthcheck.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "[失败] 连通测试异常：ValueError: mock disabled" in output
