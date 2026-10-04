"""真实大模型连通自检 —— 无 key 或 mock 配置时明确失败。

用法：
    RAG_PROVIDER=openai OPENAI_API_KEY=sk-xxx \
    OPENAI_ENDPOINT=https://example/api/modelhub/online/responses \
    OPENAI_MODEL=gpt-5.4 python scripts/healthcheck_provider.py

机制：读 config(.env/环境变量) → 装配共享 Provider → 真发一次最小 LLM 请求。
如需同时验证 embedding，设置 RAG_CHECK_EMBEDDING=true。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:  # 优先共享包，缺失时降级到本地 fallback
    from agent_provider import (
        build_embedding_provider,
        build_llm_provider,
    )
except ImportError:  # pragma: no cover - 独立运行时的正常路径
    from rag.providers._fallback import (
        build_embedding_provider,
        build_llm_provider,
    )

from config.settings import ensure_real_provider, load_settings
from rag.interfaces import ChatMessage


def main() -> int:
    cfg = load_settings()
    print(f"[配置] provider={cfg.provider} llm={cfg.llm_model} "
          f"embed={cfg.embedding_model} base_url={cfg.base_url}")

    try:
        ensure_real_provider(cfg)
        llm = build_llm_provider(cfg)
        ans = llm.chat([ChatMessage(role="user", content="只回复两个字：在吗")])
        print(f"[LLM OK] 回复={ans[:50]!r}")
        if _check_embedding_enabled():
            emb = build_embedding_provider(cfg)
            vec = emb.embed(["连通性测试"])
            print(f"[Embedding OK] 返回维度={len(vec[0])}")
        print("[结果] 真实大模型连通成功 ✓")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[失败] 连通测试异常：{type(e).__name__}: {e}")
        return 1


def _check_embedding_enabled() -> bool:
    """判断是否执行 embedding 连通检查。

    输入 (Input):
        None。读取 RAG_CHECK_EMBEDDING 环境变量。

    输出 (Output):
        True 表示额外验证 embedding provider。

    示例 (Example):
        _check_embedding_enabled()
    """
    return os.getenv("RAG_CHECK_EMBEDDING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


if __name__ == "__main__":
    raise SystemExit(main())
