"""Prompt builder regression tests."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.augmentation.prompt_builder import RagPromptBuilder  # noqa: E402
from rag.interfaces import Document, ScoredDocument  # noqa: E402


def test_prompt_builder_uses_professional_grounded_system_prompt():
    """验证系统提示词强调专业性、证据边界和禁止编造。"""
    builder = RagPromptBuilder()
    messages = builder.build(
        query="企业知识库如何回答？",
        contexts=[
            ScoredDocument(
                document=Document(
                    id="doc-1",
                    content="只能依据上下文回答。",
                    metadata={"source": "guide.md"},
                ),
                score=0.91,
            )
        ],
        history=[],
    )

    system_prompt = messages[0].content
    user_prompt = messages[-1].content
    assert "专业问答助手" in system_prompt
    assert "严格依据【上下文】" in system_prompt
    assert "无法从现有资料中确认" in system_prompt
    assert "不得补充、推测或编造" in system_prompt
    assert "来源:guide.md" in user_prompt
    assert "只能依据上下文回答" in user_prompt


def test_prompt_builder_handles_empty_contexts_with_clear_marker():
    """验证无检索结果时提示词保留清晰上下文占位。"""
    messages = RagPromptBuilder().build(
        query="没有资料的问题",
        contexts=[],
        history=[],
    )

    assert "（无检索结果）" in messages[-1].content
    assert "没有资料的问题" in messages[-1].content
