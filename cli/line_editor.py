"""模块①终端输入交互 —— 行编辑器。

解耦设计：拆成两层
  - EditBuffer：纯逻辑(光标移动/任意位置增删)，零 IO，可单测。
  - read_line()：TTY 驱动层，使用 prompt_toolkit 提供成熟行编辑能力。
TTY 不可用(无终端/管道)时 read_line 自动降级为标准 input()。

对外接口：
  EditBuffer.insert(ch) / backspace() / delete() / left() / right()
            / home() / end() / text -> str / cursor -> int
  read_line(prompt) -> str

依赖：标准库 sys；TTY 交互路径依赖 prompt_toolkit。
"""
from __future__ import annotations

import sys
from typing import Protocol


class _PromptSessionLike(Protocol):
    """prompt_toolkit PromptSession 的最小协议。"""

    def prompt(self, message: str) -> str:
        """读取一行输入。

        输入 (Input):
            message: 提示符文本。

        输出 (Output):
            用户输入的文本。

        示例 (Example):
            session.prompt("你> ")
        """
        ...


_PROMPT_SESSION: _PromptSessionLike | None = None


class EditBuffer:
    """光标行缓冲：纯内存逻辑，不做任何 IO。"""

    def __init__(self, text: str = "") -> None:
        self._chars: list[str] = list(text)
        self._cursor: int = len(self._chars)

    @property
    def text(self) -> str:
        return "".join(self._chars)

    @property
    def cursor(self) -> int:
        return self._cursor

    def insert(self, s: str) -> None:
        for ch in s:
            self._chars.insert(self._cursor, ch)
            self._cursor += 1

    def backspace(self) -> bool:
        """删除光标左侧字符。返回是否删除成功。"""
        if self._cursor > 0:
            self._chars.pop(self._cursor - 1)
            self._cursor -= 1
            return True
        return False

    def delete(self) -> bool:
        """删除光标右侧字符(Delete 键)。"""
        if self._cursor < len(self._chars):
            self._chars.pop(self._cursor)
            return True
        return False

    def left(self) -> bool:
        if self._cursor > 0:
            self._cursor -= 1
            return True
        return False

    def right(self) -> bool:
        if self._cursor < len(self._chars):
            self._cursor += 1
            return True
        return False

    def home(self) -> None:
        self._cursor = 0

    def end(self) -> None:
        self._cursor = len(self._chars)


def _render(buf: EditBuffer, prompt: str) -> None:
    """重绘当前行并把光标定位到正确列(ANSI)。"""
    # \r 回到行首, \x1b[K 清到行尾
    sys.stdout.write("\r\x1b[K" + prompt + buf.text)
    # 光标回退到 buf.cursor 位置
    move_back = len(buf.text) - buf.cursor
    if move_back > 0:
        sys.stdout.write(f"\x1b[{move_back}D")
    sys.stdout.flush()


def _get_prompt_session() -> _PromptSessionLike:
    """返回进程级 prompt_toolkit 会话，复用内存历史。

    输入 (Input):
        None。

    输出 (Output):
        prompt_toolkit PromptSession 实例。

    示例 (Example):
        session = _get_prompt_session()
    """
    global _PROMPT_SESSION
    if _PROMPT_SESSION is None:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory
        except ImportError as exc:  # pragma: no cover - 依赖缺失时才触发
            raise RuntimeError(
                "CLI 交互输入依赖 prompt_toolkit，请先执行 "
                "`python3 -m pip install -r requirements.txt`。"
            ) from exc
        _PROMPT_SESSION = PromptSession(history=InMemoryHistory())
    return _PROMPT_SESSION


def read_line(prompt: str = "> ") -> str:
    """读取一行，支持成熟终端行编辑能力。

    TTY 环境使用 prompt_toolkit，支持左右方向键、任意位置插入/删除、
    Home/End、中文宽字符光标定位和输入历史。无 TTY 时降级为 input()，
    保证非交互环境(测试/管道)可用。

    输入 (Input):
        prompt: 提示符文本。

    输出 (Output):
        用户提交的单行文本。

    示例 (Example):
        query = read_line("你> ")
    """
    if not sys.stdin.isatty():
        return input(prompt)
    return _get_prompt_session().prompt(prompt)
