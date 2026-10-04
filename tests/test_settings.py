"""配置加载验证：真实 provider 默认值、环境变量覆盖与 mock 拦截。"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings as settings_module  # noqa: E402
from config.settings import (  # noqa: E402
    Settings,
    ensure_real_provider,
    load_settings,
    normalize_search_backend,
)


def test_load_settings_uses_agent_provider_and_app_env():
    with patch.dict(
        os.environ,
        {
            "RAG_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "RAG_BASE_URL": "https://example.test/v1",
            "RAG_LLM_MODEL": "chat-model",
            "RAG_EMBEDDING_API_VERSION": "2024-03-01-preview",
            "RAG_EMBEDDING_MODEL": "embed-model",
            "RAG_EMBEDDING_BASE_URL": "https://embed.example.test/v1",
            "RAG_EMBEDDING_ENDPOINT": "https://embed.example.test/v1/embeddings",
            "RAG_EMBEDDING_DIM": "999",
            "RAG_VECTOR_BACKEND": "sqlite",
            "RAG_SEARCH_BACKEND": "bi_encoder",
            "RAG_RERANKER_BACKEND": "colbert",
            "RAG_CROSS_ENCODER_WEIGHT": "0.91",
            "RAG_CROSS_ENCODER_RETRIEVAL_SCORE_WEIGHT": "0.09",
            "RAG_COLBERT_MODEL": "colbert-ir/colbertv2.0",
            "RAG_COLBERT_QUERY_MAX_TOKENS": "24",
            "RAG_COLBERT_DOCUMENT_MAX_TOKENS": "160",
            "RAG_COLBERT_BATCH_SIZE": "4",
            "RAG_COLBERT_INTERACTION_WEIGHT": "0.9",
            "RAG_COLBERT_RETRIEVAL_SCORE_WEIGHT": "0.1",
            "RAG_COLBERT_HASHING_DIM": "48",
            "RAG_TOP_K": "7",
            "RAG_CHUNK_SIZE": "128",
            "RAG_CHUNK_OVERLAP": "16",
            "RAG_PARENT_CHILD": "true",
        },
    ):
        cfg = load_settings()

    assert cfg.provider == "openai"
    assert cfg.api_key == "sk-test"
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.llm_model == "chat-model"
    assert cfg.embedding_api_version == "2024-03-01-preview"
    assert cfg.embedding_model == "embed-model"
    assert cfg.embedding_base_url == "https://embed.example.test/v1"
    assert cfg.embedding_endpoint == "https://embed.example.test/v1/embeddings"
    assert cfg.embedding_dim == 999
    assert cfg.vector_backend == "sqlite"
    assert cfg.search_backend == "bi_encoder"
    assert cfg.reranker_backend == "colbert"
    assert cfg.cross_encoder_weight == 0.91
    assert cfg.cross_encoder_retrieval_score_weight == 0.09
    assert cfg.colbert_model == "colbert-ir/colbertv2.0"
    assert cfg.colbert_query_max_tokens == 24
    assert cfg.colbert_document_max_tokens == 160
    assert cfg.colbert_batch_size == 4
    assert cfg.colbert_interaction_weight == 0.9
    assert cfg.colbert_retrieval_score_weight == 0.1
    assert cfg.colbert_hashing_dim == 48
    assert cfg.top_k == 7
    assert cfg.chunk_size == 128
    assert cfg.chunk_overlap == 16
    assert cfg.parent_child is True
    print("✓ settings: provider/app env merged")


def test_load_settings_invalid_int_falls_back():
    with patch.dict(os.environ, {"RAG_PROVIDER": "openai", "RAG_TOP_K": "bad"}):
        cfg = load_settings()
    assert cfg.top_k == 4
    print("✓ settings: invalid integer fallback ok")


def test_normalize_search_backend_accepts_only_bi_encoder():
    """验证默认候选搜索配置是 bi-encoder，未知后端会显式失败。"""
    assert normalize_search_backend("bi-encoder") == "bi_encoder"
    assert normalize_search_backend(None) == "bi_encoder"
    try:
        normalize_search_backend("colbert")
    except ValueError as exc:
        assert "Unsupported search backend" in str(exc)
    else:
        raise AssertionError("unsupported search backend should fail")


def test_load_settings_infers_large_embedding_dim(monkeypatch):
    """验证 RAGX 复用 agent_provider 推导出的真实 embedding 维度。

    输入 (Input):
        monkeypatch: pytest 隔离 fixture。

    输出 (Output):
        None。断言 text-embedding-3-large 默认维度为 3072。

    示例 (Example):
        OPENAI_EMBEDDING_MODEL=text-embedding-3-large。
    """
    monkeypatch.setattr(settings_module, "_provider_env_files", lambda: ())
    with patch.dict(
        os.environ,
        {
            "RAG_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        },
        clear=True,
    ):
        cfg = load_settings()

    assert cfg.embedding_dim == 3072
    assert cfg.embedding_model == "text-embedding-3-large"


def test_provider_env_files_include_shared_agent_provider_env(monkeypatch, tmp_path):
    """验证 RAGX 显式传给 agent_provider 的候选列表包含共享 .env。

    输入 (Input):
        monkeypatch: pytest 隔离 fixture。
        tmp_path: 临时根目录。

    输出 (Output):
        None。断言 agent_provider/.env 会被纳入加载顺序。

    示例 (Example):
        _provider_env_files() 包含 AGENT_LIBRARY_ROOT / "agent_provider" / ".env"。
    """
    root = tmp_path / "agent-library"
    ragx_root = root / "ragx"
    shared_env = root / "agent_provider" / ".env"
    ragx_env = ragx_root / ".env"
    cwd_env = tmp_path / "workspace" / ".env"
    for path in (shared_env, ragx_env, cwd_env):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

    monkeypatch.setattr(settings_module, "AGENT_LIBRARY_ROOT", root)
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", ragx_root)
    monkeypatch.chdir(cwd_env.parent)

    env_files = settings_module._provider_env_files()

    assert shared_env in env_files
    assert env_files.index(shared_env) < env_files.index(ragx_env)
    assert env_files.index(shared_env) < env_files.index(cwd_env)


def test_ensure_real_provider_rejects_test_provider():
    try:
        ensure_real_provider(Settings(provider="mock"))
    except ValueError as exc:
        assert "禁止使用测试 provider" in str(exc)
    else:
        raise AssertionError("test provider should be rejected")

    ensure_real_provider(Settings(provider="openai"))
    print("✓ settings: test provider rejected, real provider accepted")


if __name__ == "__main__":
    test_load_settings_uses_agent_provider_and_app_env()
    test_load_settings_invalid_int_falls_back()
    test_ensure_real_provider_rejects_test_provider()
    print("\n✓✓✓ settings tests passed")
