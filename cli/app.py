"""CLI 编排层 —— 串联五大模块，自身不含任何业务逻辑。

解耦设计：ChatCLI 只依赖"接口/构件"，通过构造函数注入:
  - read_line：输入函数(终端输入交互 ①) —— 可替换为任意 callable(prompt)->str
  - spinner_factory：Loading 渲染 ② —— 可替换/可关闭
  - chat：ChatService(RAG 调用逻辑 ④) —— 背后可为真实 RAG 或 Echo
  - session：会话状态 ⑤
编排层不 import 具体实现，只认接口，因此任意模块可独立替换。

对外接口：
  ChatCLI(chat, *, read_line=..., spinner_factory=..., session=...).run()

依赖：cli.rag_service.ChatService(类型约定)、cli.session.Session。
       默认值惰性引用 line_editor/spinner，便于无依赖替换。
"""
from __future__ import annotations

from typing import Callable

from cli.rag_service import ChatService
from cli.rendering import ChatRenderer
from cli.session import Session

COMMANDS = {"exit", "quit", ":q"}


class ChatCLI:
    def __init__(
        self,
        chat: ChatService,
        *,
        read_line: Callable[[str], str] | None = None,
        spinner_factory: Callable[[str], object] | None = None,
        renderer: ChatRenderer | None = None,
        session: Session | None = None,
        prompt: str = "你> ",
    ) -> None:
        # 惰性默认:不注入时才引用具体实现，注入时彻底解耦
        if read_line is None:
            from cli.line_editor import read_line as _rl
            read_line = _rl
        if spinner_factory is None:
            from cli.spinner import Spinner
            spinner_factory = lambda text: Spinner(text)  # noqa: E731
        self._chat = chat
        self._read_line = read_line
        self._spinner_factory = spinner_factory
        self._renderer = renderer or ChatRenderer()
        self._session = session or Session()
        self._prompt = prompt

    @property
    def session(self) -> Session:
        return self._session

    def _handle_command(self, query: str) -> bool:
        """处理内置命令，返回 True 表示调用方应跳过问答。

        输入 (Input):
            query: 用户输入文本。

        输出 (Output):
            True 表示已处理命令；False 表示应进入问答链路。

        示例 (Example):
            cli._handle_command("history")
        """
        low = query.lower()
        if low in COMMANDS:
            raise _Quit
        if low in {"clear", ":clear"}:
            self._session.clear()
            self._renderer.render_status("已清空会话历史。", title="会话")
            return True
        if low in {"history", ":h"}:
            self._renderer.render_history(self._session.turns)
            return True
        return False

    def ask_once(self, query: str) -> str:
        """执行单轮问答并写入会话。

        输入 (Input):
            query: 用户问题。

        输出 (Output):
            模型回答文本。

        示例 (Example):
            cli.ask_once("报销要几天内提交？")
        """
        sp = self._spinner_factory("AI 思考中")
        start = getattr(sp, "start", None)
        if callable(start):
            start()
        try:
            ans = self._chat.answer(query)
        finally:
            stop = getattr(sp, "stop", None)
            if callable(stop):
                stop()
        self._session.add_turn(query, ans.text, ans.contexts)
        return ans.text

    def _render_answer(self, text: str, contexts: list[dict]) -> None:
        """展示回答与检索上下文。

        输入 (Input):
            text: 模型回答文本。
            contexts: 检索上下文列表。

        输出 (Output):
            None。

        示例 (Example):
            cli._render_answer("答案", [{"source": "doc.md"}])
        """
        self._renderer.render_answer(text, contexts)

    def run(self) -> None:
        """启动交互式对话循环。

        输入 (Input):
            None。

        输出 (Output):
            None。

        示例 (Example):
            ChatCLI(chat).run()
        """
        self._renderer.render_welcome(COMMANDS)
        while True:
            try:
                query = self._read_line(self._prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self._renderer.render_goodbye()
                break
            if not query:
                continue
            try:
                if self._handle_command(query):
                    continue
            except _Quit:
                self._renderer.render_goodbye()
                break
            text = self.ask_once(query)
            contexts = self._session.turns[-1].contexts
            self._render_answer(text, contexts)


class _Quit(Exception):
    """内部信号:退出循环。"""
