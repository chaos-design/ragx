"""Rich 终端渲染层 —— 统一 CLI 欢迎页、回答、历史与检索证据展示。

本模块只负责展示，不读取输入、不调用 RAG、不维护会话状态。这样 `ChatCLI`
仍然只做编排，展示样式可以独立测试和替换。
"""
from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from typing import Any

try:
    from rich import box
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError:
    box = Console = Markdown = Panel = Table = Text = None

_RICH_AVAILABLE = Console is not None


class ChatRenderer:
    """RAG CLI 展示适配器，Rich 不可用时自动降级为纯文本。"""

    def __init__(
        self,
        *,
        width: int | None = None,
        preview_limit: int = 0,
        force_plain: bool = False,
    ) -> None:
        """初始化渲染器。

        输入 (Input):
            width: 可选固定终端宽度，None 表示跟随当前 stdout。
            preview_limit: 检索片段预览最大字符数，0 表示不截断。
            force_plain: 强制使用纯文本渲染，便于测试或最小依赖运行。

        输出 (Output):
            None。

        示例 (Example):
            ChatRenderer(width=100, preview_limit=80)
        """
        self._width = width
        self._preview_limit = preview_limit
        self._use_rich = _RICH_AVAILABLE and not force_plain

    def render_welcome(self, commands: Iterable[str]) -> None:
        """展示 CLI 欢迎区和可用命令。

        输入 (Input):
            commands: 退出命令集合。

        输出 (Output):
            None。

        示例 (Example):
            renderer.render_welcome({"exit", "quit"})
        """
        command_text = " / ".join(sorted(commands))
        if not self._use_rich:
            print(
                "[RAGX] RAG 多轮对话验证 CLI"
                f" | 退出: {command_text} | 会话: clear 清空 / history 查看"
            )
            return
        body = Text(no_wrap=True, overflow="ellipsis")
        body.append("RAG 多轮对话验证 CLI", style="bold")
        body.append(" | 退出: ")
        body.append(command_text, style="cyan")
        body.append(" | 会话: ")
        body.append("clear", style="cyan")
        body.append(" 清空 / ")
        body.append("history", style="cyan")
        body.append(" 查看")
        self._console().print(
            Panel.fit(
                body,
                title="RAGX",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def render_status(self, message: str, *, title: str = "状态") -> None:
        """展示状态提示。

        输入 (Input):
            message: 状态文本。
            title: 面板标题。

        输出 (Output):
            None。

        示例 (Example):
            renderer.render_status("已清空会话历史", title="会话")
        """
        if not self._use_rich:
            print(f"[{title}] {message}")
            return
        self._console().print(
            Panel(
                Text(message),
                title=title,
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def render_goodbye(self) -> None:
        """展示退出提示。

        输入 (Input):
            None。

        输出 (Output):
            None。

        示例 (Example):
            renderer.render_goodbye()
        """
        self.render_status("再见。", title="退出")

    def render_history(self, turns: Sequence[Any]) -> None:
        """展示会话历史。

        输入 (Input):
            turns: 包含 user、assistant 字段的历史轮次。

        输出 (Output):
            None。

        示例 (Example):
            renderer.render_history(session.turns)
        """
        if not turns:
            self.render_status("暂无会话历史。", title="历史")
            return
        if not self._use_rich:
            print("[历史] 会话历史")
            for index, turn in enumerate(turns, 1):
                question = _clip(_clean(getattr(turn, "user", "")), 80)
                answer = _clip(_clean(getattr(turn, "assistant", "")), 120)
                print(f"{index}. 用户: {question}")
                print(f"   AI: {answer}")
            return

        table = Table(
            title="会话历史",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            expand=True,
        )
        table.add_column("#", justify="right", width=4, style="cyan")
        table.add_column("用户问题", ratio=1)
        table.add_column("AI 回答", ratio=2)
        for index, turn in enumerate(turns, 1):
            table.add_row(
                str(index),
                _clip(_clean(getattr(turn, "user", "")), 80),
                _clip(_clean(getattr(turn, "assistant", "")), 120),
            )
        self._console().print(table)

    def render_answer(self, text: str, contexts: Sequence[dict]) -> None:
        """展示回答和检索证据。

        输入 (Input):
            text: 模型回答文本。
            contexts: RagApplication 返回的上下文片段。

        输出 (Output):
            None。

        示例 (Example):
            renderer.render_answer("答案", [{"source": "doc.md"}])
        """
        if not self._use_rich:
            print("\n[回答]")
            print(text or "（空响应）")
            self.render_contexts(contexts)
            return
        self._console().print(
            Panel(
                Markdown(text or "（空响应）"),
                title="回答",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self.render_contexts(contexts)

    def render_contexts(self, contexts: Sequence[dict]) -> None:
        """展示检索命中的溯源片段。

        输入 (Input):
            contexts: RagApplication 返回的上下文片段。

        输出 (Output):
            None。

        示例 (Example):
            renderer.render_contexts([{"source": "doc.md", "score": 1.2}])
        """
        if not contexts:
            self.render_status("未返回检索片段。", title="检索证据")
            return
        if not self._use_rich:
            print("\n[检索证据]")
            for index, context in enumerate(contexts, 1):
                preview = _clip(_clean(context.get("preview", "")), self._preview_limit)
                heading = _clean(_format_heading(context.get("heading_path")))
                print(
                    f"{index}. score={_format_score(context.get('score'))} "
                    f"source={_format_source(context)} heading={heading}"
                )
                if preview:
                    print(f"   {preview}")
            return

        table = Table(
            title="检索证据",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            expand=True,
        )
        table.add_column("#", justify="right", width=4, style="cyan")
        table.add_column("score", justify="right", width=8)
        table.add_column("来源", ratio=1)
        table.add_column("章节", ratio=1)
        table.add_column("片段预览", ratio=3, overflow="fold")

        for index, context in enumerate(contexts, 1):
            preview = _clean(context.get("preview", ""))
            table.add_row(
                str(index),
                _format_score(context.get("score")),
                _format_source(context),
                _clip(_clean(_format_heading(context.get("heading_path"))), 80),
                _clip(preview, self._preview_limit),
            )
        self._console().print(table)

    def render_index_report(self, report: Any) -> None:
        """展示知识库索引同步报告。

        输入 (Input):
            report: SyncReport 或具备同名字段的对象。

        输出 (Output):
            None。

        示例 (Example):
            renderer.render_index_report(report)
        """
        metrics = (
            ("新增", getattr(report, "added", 0)),
            ("更新", getattr(report, "updated", 0)),
            ("跳过", getattr(report, "skipped", 0)),
            ("删除", getattr(report, "deleted", 0)),
            ("复用", getattr(report, "reused_chunks", 0)),
            ("重嵌入", getattr(report, "reembedded_chunks", 0)),
        )
        if not self._use_rich:
            summary = " / ".join(f"{label} {value}" for label, value in metrics)
            print(f"[索引] 知识库同步完成 {summary}")
            return

        body = Text(no_wrap=True, overflow="ellipsis")
        for index, (label, value) in enumerate(metrics):
            if index:
                body.append(" / ")
            body.append(f"{label} ", style="cyan")
            body.append(str(value))
        self._console().print(
            Panel.fit(
                body,
                title="[索引] 知识库同步完成",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def _console(self) -> Any:
        """创建绑定当前 stdout 的 Console，便于测试中重定向捕获。

        输入 (Input):
            None。

        输出 (Output):
            Rich Console 实例。

        示例 (Example):
            renderer._console()
        """
        return Console(
            file=sys.stdout,
            width=self._width,
            highlight=False,
            soft_wrap=True,
            color_system="auto",
        )


def _clean(value: Any) -> str:
    """压平展示文本中的多余空白。

    输入 (Input):
        value: 任意可转字符串的值。

    输出 (Output):
        去掉连续空白后的字符串。

    示例 (Example):
        _clean("a\\n b")
    """
    return " ".join(str(value or "").split())


def _clip(text: str, limit: int) -> str:
    """按字符数截断文本。

    输入 (Input):
        text: 原始文本。
        limit: 最大长度。

    输出 (Output):
        不超过 limit 的展示文本。

    示例 (Example):
        _clip("abcdef", 4)
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_score(score: Any) -> str:
    """格式化检索分数。

    输入 (Input):
        score: 数字或字符串分数。

    输出 (Output):
        保留四位有效小数的展示文本。

    示例 (Example):
        _format_score(1.23456)
    """
    if isinstance(score, (int, float)):
        return f"{score:.4f}"
    return str(score or "")


def _format_heading(value: Any) -> str:
    """格式化 heading_path 字段。

    输入 (Input):
        value: 字符串、列表或空值。

    输出 (Output):
        可读章节路径。

    示例 (Example):
        _format_heading(["A", "B"])
    """
    if isinstance(value, (list, tuple)):
        return " > ".join(str(item) for item in value)
    return str(value or "")


def _format_source(context: dict) -> str:
    """格式化来源字段。

    输入 (Input):
        context: 单条检索上下文。

    输出 (Output):
        包含 source/page/version 的来源文本。

    示例 (Example):
        _format_source({"source": "doc.md", "page": 1})
    """
    source = str(context.get("source") or "?")
    extras: list[str] = []
    if context.get("page"):
        extras.append(f"p{context['page']}")
    if context.get("version"):
        extras.append(f"v{context['version']}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return _clip(source + suffix, 80)
