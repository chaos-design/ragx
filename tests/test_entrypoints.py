"""入口脚本测试：覆盖 chat_cli.py 与 main.py 的装配和输出路径。"""
from __future__ import annotations

import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace

RAGX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(RAGX_ROOT, ".."))
for path in (RAGX_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:  # 优先共享包，缺失时降级到本地 fallback
    from agent_provider import ProviderInitError  # noqa: E402
except ImportError:  # pragma: no cover - 独立运行时的正常路径
    from rag.providers._fallback import ProviderInitError  # noqa: E402

import chat_cli  # noqa: E402
import main as main_module  # noqa: E402
from cli.rag_service import EchoChatService, RagChatService  # noqa: E402
from config.settings import Settings  # noqa: E402


class _Report:
    added = 1
    updated = 2
    skipped = 3
    deleted = 4
    reused_chunks = 5
    reembedded_chunks = 6


class _FakeRagApplication:
    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.indexed_sources = []
        _FakeRagApplication.instances.append(self)

    def index(self, source):
        self.indexed_sources.append(source)
        return _Report()

    def ask(self, query):
        return {
            "answer": f"answer:{query}",
            "contexts": [
                {
                    "source": "doc.md",
                    "page": 1,
                    "heading_path": ["A"],
                    "version": 1,
                    "score": 0.9,
                    "preview": "preview",
                }
            ],
        }


def test_chat_cli_build_chat_echo():
    args = SimpleNamespace(echo=True, delay=0.01, persist=None)
    service = chat_cli.build_chat(args)
    assert isinstance(service, EchoChatService)
    assert "ping" in service.answer("ping").text
    print("✓ chat_cli: echo service branch ok")


def test_chat_cli_build_chat_rag_with_persist(monkeypatch, tmp_path):
    _FakeRagApplication.instances.clear()
    cfg = Settings(provider="openai", api_key="sk-test")

    import config.settings as settings_module
    import rag.app as app_module

    monkeypatch.setattr(settings_module, "load_settings", lambda: cfg)
    monkeypatch.setattr(settings_module, "ensure_real_provider", lambda value: None)
    monkeypatch.setattr(app_module, "RagApplication", _FakeRagApplication)

    args = SimpleNamespace(echo=False, delay=0.0, persist=str(tmp_path))
    output = StringIO()
    with redirect_stdout(output):
        service = chat_cli.build_chat(args)

    assert isinstance(service, RagChatService)
    assert cfg.vector_backend == "sqlite"
    assert cfg.sqlite_path == os.path.join(str(tmp_path), "vectors.db")
    assert cfg.manifest_path == os.path.join(str(tmp_path), "manifest.json")
    assert _FakeRagApplication.instances[-1].indexed_sources == [chat_cli.DATA_DIR]
    assert "[索引]" in output.getvalue()
    print("✓ chat_cli: rag persist branch ok")


def test_chat_cli_main_single_question(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chat_cli.py", "--echo", "hello"])
    output = StringIO()
    with redirect_stdout(output):
        chat_cli.main()

    out = output.getvalue()
    assert "回答" in out
    assert "(echo) 你说的是：hello" in out
    assert "检索证据" in out
    print("✓ chat_cli: main single question ok")


def test_chat_cli_main_provider_init_error(monkeypatch):
    """Provider 缺失配置时应输出修复指引而不是裸 traceback。"""

    def fail_build_chat(args):
        """模拟 provider 初始化失败。

        输入 (Input):
            args: CLI 参数。

        输出 (Output):
            抛出 ProviderInitError。

        示例 (Example):
            fail_build_chat(SimpleNamespace())
        """
        raise ProviderInitError("OpenAI provider requires OPENAI_API_KEY")

    monkeypatch.setattr(sys, "argv", ["chat_cli.py"])
    monkeypatch.setattr(chat_cli, "build_chat", fail_build_chat)
    error = StringIO()

    with redirect_stderr(error):
        try:
            chat_cli.main()
        except SystemExit as exc:
            assert exc.code == 2

    message = error.getvalue()
    assert "[配置错误]" in message
    assert "export OPENAI_API_KEY" in message
    assert "python chat_cli.py --echo" in message
    print("✓ chat_cli: provider init error message ok")


def test_chat_cli_main_runtime_dependency_error(monkeypatch):
    """运行依赖缺失时应输出当前 Python 和安装命令，而不是裸 traceback。"""

    def fail_build_chat(args):
        """模拟 PDF 依赖缺失导致建库失败。

        输入 (Input):
            args: CLI 参数。

        输出 (Output):
            抛出 RuntimeError。

        示例 (Example):
            fail_build_chat(SimpleNamespace())
        """
        raise RuntimeError("读取 PDF 需要安装 PyMuPDF：python3 -m pip install pymupdf")

    monkeypatch.setattr(sys, "argv", ["chat_cli.py"])
    monkeypatch.setattr(chat_cli, "build_chat", fail_build_chat)
    error = StringIO()

    with redirect_stderr(error):
        try:
            chat_cli.main()
        except SystemExit as exc:
            assert exc.code == 2

    message = error.getvalue()
    assert "[运行依赖错误]" in message
    assert "当前 Python:" in message
    assert "paddlepaddle" in message
    assert "-m pip install -r requirements.txt" in message
    assert "python3 chat_cli.py --echo" in message
    print("✓ chat_cli: runtime dependency error message ok")


def test_chat_cli_main_provider_runtime_error(monkeypatch):
    """Provider 请求失败时应输出配置排查项，而不是裸 traceback。"""

    import config.settings as settings_module

    class FakeProviderStatusError(Exception):
        status_code = 404

    def fail_build_chat(args):
        """模拟 embedding provider 返回 404。

        输入 (Input):
            args: CLI 参数。

        输出 (Output):
            抛出 provider status error。

        示例 (Example):
            fail_build_chat(SimpleNamespace())
        """
        raise FakeProviderStatusError("404 page not found")

    monkeypatch.setattr(sys, "argv", ["chat_cli.py"])
    monkeypatch.setattr(chat_cli, "build_chat", fail_build_chat)
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda: SimpleNamespace(
            endpoint="https://example.test/v1/responses",
            embedding_endpoint=None,
            embedding_base_url=None,
            embedding_model="text-embedding-3-large",
        ),
    )
    error = StringIO()

    with redirect_stderr(error):
        try:
            chat_cli.main()
        except SystemExit as exc:
            assert exc.code == 2

    message = error.getvalue()
    assert "[模型服务错误]" in message
    assert "404 page not found" in message
    assert "未配置独立 embedding endpoint/base URL" in message
    assert "OPENAI_EMBEDDING_ENDPOINT" in message
    assert "OPENAI_EMBEDDING_MODEL" in message
    assert "python3 chat_cli.py --echo" in message
    print("✓ chat_cli: provider runtime error message ok")


def test_chat_cli_main_question_provider_runtime_error(monkeypatch):
    """单轮问答阶段的 provider 错误也应被 CLI 统一捕获。"""

    class FailingChat:
        def answer(self, query):
            """模拟 chat provider 运行时失败。

            输入 (Input):
                query: 用户问题。

            输出 (Output):
                抛出 provider 404。

            示例 (Example):
                FailingChat().answer("hi")
            """
            error = RuntimeError("provider returned 404 page not found")
            raise error

    monkeypatch.setattr(sys, "argv", ["chat_cli.py", "hello"])
    monkeypatch.setattr(chat_cli, "build_chat", lambda args: FailingChat())
    error = StringIO()

    with redirect_stderr(error), redirect_stdout(StringIO()):
        try:
            chat_cli.main()
        except SystemExit as exc:
            assert exc.code == 2

    assert "[模型服务错误]" in error.getvalue()
    print("✓ chat_cli: question provider runtime error message ok")


def test_main_single_question(monkeypatch):
    _FakeRagApplication.instances.clear()
    cfg = Settings(provider="openai", api_key="sk-test")

    monkeypatch.setattr(main_module, "load_settings", lambda: cfg)
    monkeypatch.setattr(main_module, "ensure_real_provider", lambda value: None)
    monkeypatch.setattr(main_module, "RagApplication", _FakeRagApplication)
    monkeypatch.setattr(sys, "argv", ["main.py", "hello", "rag"])

    output = StringIO()
    with redirect_stdout(output):
        main_module.main()

    out = output.getvalue()
    assert "[索引] 新增 1 文档 / 更新 2" in out
    assert "答> answer:hello rag" in out
    assert "src=doc.md" in out
    assert _FakeRagApplication.instances[-1].indexed_sources == [main_module.DATA_DIR]
    print("✓ main: single question branch ok")


def test_main_interactive_exit(monkeypatch):
    cfg = Settings(provider="openai", api_key="sk-test")

    monkeypatch.setattr(main_module, "load_settings", lambda: cfg)
    monkeypatch.setattr(main_module, "ensure_real_provider", lambda value: None)
    monkeypatch.setattr(main_module, "RagApplication", _FakeRagApplication)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")

    output = StringIO()
    with redirect_stdout(output):
        main_module.main()

    assert "RAG 交互问答" in output.getvalue()
    print("✓ main: interactive exit branch ok")
