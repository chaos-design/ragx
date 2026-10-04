"""CLI 工具验证:各模块解耦、可独立单测(无需 TTY)。

覆盖:
  ① EditBuffer 纯逻辑:任意光标位置增删 / 左右移动 / 行首行尾
  ② Spinner 非 TTY 优雅降级 + start/stop 不抛错
  ④ ChatService 适配器:Echo / Rag(用桩 app)
  ⑤ Session 多轮状态 + history 滑窗
  编排 ChatCLI:注入假 read_line/spinner,跑通多轮 + 命令,验证零耦合
"""
from __future__ import annotations

import builtins
import importlib.util
import os
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli.app import ChatCLI  # noqa: E402
from cli.line_editor import EditBuffer, _render, read_line  # noqa: E402
from cli.rag_service import (ChatAnswer, ChatService, EchoChatService,  # noqa: E402
                             RagChatService)
from cli.rendering import ChatRenderer  # noqa: E402
from cli.session import Session  # noqa: E402
from cli.spinner import Spinner  # noqa: E402


def test_edit_buffer_insert_and_cursor():
    buf = EditBuffer()
    buf.insert("helo")
    assert buf.text == "helo" and buf.cursor == 4
    # 光标移到 'l' 与 'o' 之间(index 3),在任意位置插入
    buf.left()                       # 光标 -> 3
    buf.insert("l")                  # "hello"
    assert buf.text == "hello" and buf.cursor == 4
    print("✓ EditBuffer: 任意位置插入 ok")


def test_edit_buffer_delete_backspace_bounds():
    buf = EditBuffer("abc")
    buf.home()
    assert buf.cursor == 0 and buf.backspace() is False  # 行首退格无效
    assert buf.delete() is True and buf.text == "bc"     # 删除光标右侧 'a'
    buf.end()
    assert buf.delete() is False                          # 行尾 delete 无效
    assert buf.backspace() is True and buf.text == "b"
    print("✓ EditBuffer: 增删边界 ok")


def test_edit_buffer_navigation():
    buf = EditBuffer("12345")
    buf.home()
    assert buf.cursor == 0
    buf.right()
    buf.right()
    assert buf.cursor == 2
    buf.end()
    assert buf.cursor == 5
    assert buf.right() is False                           # 越界保护
    print("✓ EditBuffer: 光标导航 ok")


def test_spinner_no_tty_graceful():
    sp = Spinner("测试")
    # 沙箱无 TTY:start 只打印一次、stop 直接返回,均不应抛错
    sp.start()
    sp.stop()
    with Spinner("ctx"):
        pass
    print("✓ Spinner: 无 TTY 优雅降级 ok")


def test_spinner_tty_spin_and_clear_paths():
    class FakeStdout:
        def __init__(self):
            self.buffer = []

        def isatty(self):
            return True

        def write(self, text):
            self.buffer.append(text)

        def flush(self):
            pass

    fake_stdout = FakeStdout()
    with patch.object(sys, "stdout", fake_stdout):
        sp = Spinner("测试", interval=0.001)
        sp.start()
        time.sleep(0.003)
        sp.stop(clear=True)

    assert any("测试" in item for item in fake_stdout.buffer)
    assert "\r\x1b[K" in fake_stdout.buffer
    print("✓ Spinner: TTY start/stop/clear paths ok")


def test_echo_chat_service():
    svc = EchoChatService()
    assert isinstance(svc, ChatService)                   # Protocol 鸭子类型
    ans = svc.answer("你好")
    assert isinstance(ans, ChatAnswer) and "你好" in ans.text and ans.contexts == []
    print("✓ EchoChatService: 实现 ChatService 协议 ok")


def test_rag_chat_service_adapter():
    class StubApp:
        def ask(self, q):
            return {"answer": f"回答:{q}", "contexts": [{"source": "x.md", "score": 0.9}]}

    svc = RagChatService(StubApp())
    assert isinstance(svc, ChatService)
    ans = svc.answer("报销")
    assert ans.text == "回答:报销" and ans.contexts[0]["source"] == "x.md"
    print("✓ RagChatService: 归一化适配 ok")


def test_echo_chat_service_delay_branch():
    svc = EchoChatService(delay=0.001)
    assert "延迟" in svc.answer("延迟").text
    print("✓ EchoChatService: delay branch ok")


def test_session_state_and_history_window():
    s = Session()
    assert s.count == 0
    s.add_turn("q1", "a1", [{"source": "d"}])
    s.add_turn("q2", "a2")
    assert s.count == 2 and s.turns[0].contexts[0]["source"] == "d"
    # history:每轮 2 条消息;滑窗只取最近 1 轮
    assert len(s.history()) == 4
    win = s.history(max_turns=1)
    assert len(win) == 2 and win[0].content == "q2"
    s.clear()
    assert s.count == 0
    print("✓ Session: 多轮状态 + history 滑窗 ok")


def test_chat_cli_orchestration_decoupled():
    """注入假 read_line/spinner,验证编排层不依赖任何真实终端/RAG。"""
    scripted = iter(["第一个问题", "history", "clear", "exit"])
    calls = {"spin_start": 0, "spin_stop": 0}

    class FakeSpinner:
        def __init__(self, text):
            pass

        def start(self):
            calls["spin_start"] += 1
            return self

        def stop(self, clear=True):
            calls["spin_stop"] += 1

    cli = ChatCLI(
        EchoChatService(),
        read_line=lambda prompt: next(scripted),
        spinner_factory=lambda text: FakeSpinner(text),
        session=Session(),
    )
    cli.run()
    # 只有"第一个问题"进入问答(history/clear 是命令),clear 后历史清空
    assert calls["spin_start"] == 1 and calls["spin_stop"] == 1
    assert cli.session.count == 0       # 被 clear 清空
    print("✓ ChatCLI: 注入式编排 + 命令处理 + 完全解耦 ok")


def test_chat_cli_default_dependencies_and_rendering():
    cli = ChatCLI(EchoChatService(), session=Session())
    assert cli.session.count == 0
    output = StringIO()
    with redirect_stdout(output):
        cli._render_answer("答案", [])
        cli._render_answer(
            "答案",
            [{"source": "doc.md", "score": 0.9, "preview": "片段"}],
        )
    out = output.getvalue()
    assert "回答" in out and "答案" in out
    assert "检索证据" in out
    assert "未返回检索片段" in out
    assert "doc.md" in out and "片段" in out
    print("✓ ChatCLI: default injection and rendering ok")


def test_renderer_context_preview_is_not_truncated():
    long_preview = "完整片段开始 " + ("关键证据 " * 30) + "完整片段结束TAIL"
    output = StringIO()

    with redirect_stdout(output):
        ChatRenderer(width=100).render_contexts(
            [
                {
                    "source": "doc.md",
                    "score": 0.912345,
                    "heading_path": "章节",
                    "preview": long_preview,
                }
            ]
        )

    out = output.getvalue()
    assert "完整片段开始" in out
    assert "完整片段结束TAIL" in out
    print("✓ ChatRenderer: context preview keeps full snippet")


def test_renderer_welcome_is_single_line_content():
    output = StringIO()
    with redirect_stdout(output):
        ChatRenderer(width=120).render_welcome({"exit", "quit"})

    body_lines = [
        line for line in output.getvalue().splitlines()
        if "RAG 多轮对话验证 CLI" in line
    ]
    assert len(body_lines) == 1
    assert "退出:" in body_lines[0]
    assert "clear" in body_lines[0] and "history" in body_lines[0]
    print("✓ ChatRenderer: welcome content single line ok")


def test_renderer_index_report_is_compact():
    class Report:
        added = 1
        updated = 2
        skipped = 3
        deleted = 4
        reused_chunks = 5
        reembedded_chunks = 6

    output = StringIO()
    with redirect_stdout(output):
        ChatRenderer(width=120).render_index_report(Report())

    body_lines = [
        line for line in output.getvalue().splitlines()
        if "新增 1" in line
    ]
    assert "[索引] 知识库同步完成" in output.getvalue()
    assert len(body_lines) == 1
    assert "更新 2" in body_lines[0]
    assert "跳过 3" in body_lines[0]
    assert "删除 4" in body_lines[0]
    assert "复用 5" in body_lines[0]
    assert "重嵌入 6" in body_lines[0]
    print("✓ ChatRenderer: index report single line ok")


def test_renderer_plain_fallback_outputs_without_rich():
    """验证强制纯文本模式可覆盖回答、证据和索引报告。"""

    class Report:
        added = 1
        updated = 0
        skipped = 0
        deleted = 0
        reused_chunks = 2
        reembedded_chunks = 3

    output = StringIO()
    renderer = ChatRenderer(width=120, force_plain=True)

    with redirect_stdout(output):
        renderer.render_welcome({"exit"})
        renderer.render_answer(
            "答案",
            [{"source": "doc.md", "score": 0.8, "preview": "证据"}],
        )
        renderer.render_index_report(Report())

    out = output.getvalue()
    assert "[RAGX] RAG 多轮对话验证 CLI" in out
    assert "[回答]" in out and "答案" in out
    assert "[检索证据]" in out and "doc.md" in out and "证据" in out
    assert "[索引] 知识库同步完成" in out
    assert "复用 2" in out and "重嵌入 3" in out
    print("✓ ChatRenderer: plain fallback output ok")


def test_rendering_module_imports_when_rich_is_missing(monkeypatch):
    """验证 rich 缺失时 rendering 模块仍可导入并纯文本输出。"""
    module_path = Path(__file__).resolve().parents[1] / "cli" / "rendering.py"
    real_import = builtins.__import__

    def block_rich(name, *args, **kwargs):
        """模拟当前 Python 环境未安装 rich。

        输入 (Input):
            name: import 目标模块名。

        输出 (Output):
            非 rich 模块走真实 import，rich 抛 ModuleNotFoundError。

        示例 (Example):
            block_rich("rich")
        """
        if name == "rich" or name.startswith("rich."):
            raise ModuleNotFoundError("No module named 'rich'", name="rich")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_rich)
    spec = importlib.util.spec_from_file_location("_plain_rendering", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    output = StringIO()
    with redirect_stdout(output):
        module.ChatRenderer().render_answer("答案", [])

    out = output.getvalue()
    assert "[回答]" in out
    assert "答案" in out
    assert "[检索证据] 未返回检索片段。" in out
    print("✓ ChatRenderer: import without rich ok")


def test_chat_cli_empty_input_and_eof():
    scripted = iter(["", "quit"])
    cli = ChatCLI(
        EchoChatService(),
        read_line=lambda prompt: next(scripted),
        session=Session(),
    )

    cli_eof = ChatCLI(
        EchoChatService(),
        read_line=lambda prompt: (_ for _ in ()).throw(EOFError),
        session=Session(),
    )
    output = StringIO()
    with redirect_stdout(output):
        cli.run()
        cli_eof.run()
    out = output.getvalue()
    assert "再见" in out
    print("✓ ChatCLI: empty input and EOF branches ok")


def test_read_line_non_tty_and_render():
    class FakeStdin:
        def isatty(self):
            return False

    class FakeStdout:
        def __init__(self):
            self.buffer = []

        def write(self, text):
            self.buffer.append(text)

        def flush(self):
            pass

    with patch.object(sys, "stdin", FakeStdin()):
        with patch("builtins.input", lambda prompt: f"{prompt}value"):
            assert read_line("P> ") == "P> value"

    buf = EditBuffer("abc")
    buf.left()
    fake_stdout = FakeStdout()
    with patch.object(sys, "stdout", fake_stdout):
        _render(buf, "P> ")

    assert any("P> abc" in item for item in fake_stdout.buffer)
    assert any(item == "\x1b[1D" for item in fake_stdout.buffer)
    print("✓ line_editor: non-tty input and render paths ok")


def test_read_line_tty_uses_prompt_toolkit_session():
    class FakeStdin:
        def isatty(self):
            return True

    class FakeSession:
        def __init__(self):
            self.prompts = []

        def prompt(self, message):
            self.prompts.append(message)
            return "中间插入后删除"

    fake_session = FakeSession()
    with patch.object(sys, "stdin", FakeStdin()):
        with patch("cli.line_editor._get_prompt_session", lambda: fake_session):
            assert read_line("你> ") == "中间插入后删除"

    assert fake_session.prompts == ["你> "]
    print("✓ line_editor: TTY uses prompt_toolkit session ok")


if __name__ == "__main__":
    test_edit_buffer_insert_and_cursor()
    test_edit_buffer_delete_backspace_bounds()
    test_edit_buffer_navigation()
    test_spinner_no_tty_graceful()
    test_spinner_tty_spin_and_clear_paths()
    test_echo_chat_service()
    test_rag_chat_service_adapter()
    test_echo_chat_service_delay_branch()
    test_session_state_and_history_window()
    test_chat_cli_orchestration_decoupled()
    test_chat_cli_default_dependencies_and_rendering()
    test_renderer_context_preview_is_not_truncated()
    test_renderer_welcome_is_single_line_content()
    test_renderer_index_report_is_compact()
    test_chat_cli_empty_input_and_eof()
    test_read_line_non_tty_and_render()
    test_read_line_tty_uses_prompt_toolkit_session()
    print("\n✓✓✓ all CLI tests passed")
