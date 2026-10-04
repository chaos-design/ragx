"""CLI 入口(组合根) —— 唯一 new 对象的地方,把五大模块装配起来。

用法:
    OPENAI_API_KEY=sk-xxx python chat_cli.py
    python chat_cli.py "一次性问题"     # 单次提问(真实 provider + data/ 建库)
    python chat_cli.py --echo          # 仅用于 CLI 交互联调，不进入生产 RAG 链路

模块装配:
    line_editor.read_line  ①终端输入交互
    spinner.Spinner        ②Loading 渲染
    rag.providers(经 RagApplication)  ③LLM Provider 层
    rag_service.{Rag,Echo}ChatService ④RAG 调用逻辑
    session.Session        ⑤会话状态管理
    app.ChatCLI            编排(仅注入,不含业务)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:  # 优先共享包，缺失时降级到本地 fallback
    from agent_provider import ProviderInitError
except ImportError:  # pragma: no cover - 独立运行时的正常路径
    from rag.providers._fallback import ProviderInitError

from cli.app import ChatCLI
from cli.rag_service import EchoChatService, RagChatService
from cli.rendering import ChatRenderer
from cli.session import Session

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_chat(args, renderer: ChatRenderer | None = None):
    """按 CLI 参数装配 ChatService。

    输入 (Input):
        args: argparse 解析出的命名空间。
        renderer: 可选 Rich 展示器，用于输出索引报告。

    输出 (Output):
        ChatService 实例，可能是真实 RAG 或 Echo 假后端。

    示例 (Example):
        build_chat(argparse.Namespace(echo=True, delay=0, persist=None))
    """
    if args.echo:
        return EchoChatService(delay=args.delay)
    from config.settings import ensure_real_provider, load_settings
    from rag.app import RagApplication
    cfg = load_settings()
    ensure_real_provider(cfg)
    # --persist:覆盖默认持久化目录，仍复用真实 provider 配置
    if args.persist:
        cfg.vector_backend = "sqlite"
        cfg.sqlite_path = os.path.join(args.persist, "vectors.db")
        cfg.manifest_path = os.path.join(args.persist, "manifest.json")
    app = RagApplication(cfg)
    report = app.index(DATA_DIR)
    (renderer or ChatRenderer()).render_index_report(report)
    return RagChatService(app)


def main() -> None:
    """解析命令行参数并启动单轮或交互式 CLI。

    输入 (Input):
        None。参数来自 sys.argv。

    输出 (Output):
        None。

    示例 (Example):
        OPENAI_API_KEY=sk-xxx python chat_cli.py "报销要几天内提交？"
    """
    parser = argparse.ArgumentParser(description="RAG 多轮对话验证 CLI")
    parser.add_argument("question", nargs="*", help="一次性提问(留空进入交互)")
    parser.add_argument("--echo", action="store_true", help="使用 Echo 假后端联调")
    parser.add_argument("--delay", type=float, default=0.0, help="Echo 模拟延迟秒数")
    parser.add_argument("--persist", metavar="DIR", default=None,
                        help="持久化目录:sqlite 向量库 + manifest 落盘,增量更新跨运行生效")
    args = parser.parse_args()

    try:
        chat = build_chat(args)
    except ProviderInitError as exc:
        print(_provider_init_error_message(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        if _is_provider_runtime_error(exc):
            print(_provider_runtime_error_message(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        print(_runtime_error_message(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        if _is_provider_runtime_error(exc):
            print(_provider_runtime_error_message(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        raise

    try:
        if args.question:
            cli = ChatCLI(chat, session=Session())
            text = cli.ask_once(" ".join(args.question))
            cli._render_answer(text, cli.session.turns[-1].contexts)
            return

        ChatCLI(chat, session=Session()).run()
    except Exception as exc:
        if _is_provider_runtime_error(exc):
            print(_provider_runtime_error_message(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        raise


def _provider_init_error_message(exc: ProviderInitError) -> str:
    """生成 provider 初始化失败时的用户提示。

    输入 (Input):
        exc: ProviderInitError 异常。

    输出 (Output):
        包含修复命令的错误提示文本。

    示例 (Example):
        _provider_init_error_message(ProviderInitError("missing key"))
    """
    return (
        f"[配置错误] {exc}\n"
        "真实 RAG 对话需要配置模型 API Key：\n"
        "  1. export OPENAI_API_KEY=你的真实Key\n"
        "  2. 仅验证 CLI 交互时运行: python chat_cli.py --echo\n"
    )


def _runtime_error_message(exc: RuntimeError) -> str:
    """生成运行依赖或建库失败时的用户提示。

    输入 (Input):
        exc: RuntimeError 异常。

    输出 (Output):
        包含当前 Python 和依赖安装命令的错误提示文本。

    示例 (Example):
        _runtime_error_message(RuntimeError("读取 PDF 需要安装 PyMuPDF"))
    """
    return (
        f"[运行依赖错误] {exc}\n"
        f"当前 Python: {sys.executable}\n"
        "请在同一个 Python 环境中安装 RAGX 运行依赖"
        "（含 PyMuPDF、PaddleOCR 与 paddlepaddle）：\n"
        f"  {sys.executable} -m pip install -r requirements.txt\n"
        "如果只想验证 CLI 交互，不需要真实建库：\n"
        "  python3 chat_cli.py --echo\n"
    )


def _is_provider_runtime_error(exc: Exception) -> bool:
    """判断异常是否来自模型服务请求阶段。

    输入 (Input):
        exc: 任意运行时异常。

    输出 (Output):
        bool；True 表示应转换为 provider 配置提示。

    Example Input:
        _is_provider_runtime_error(OpenAIError("404 page not found"))

    Example Output:
        True
    """
    error_type = f"{type(exc).__module__}.{type(exc).__name__}".lower()
    message = str(exc).lower()
    status_code = _exception_attr(exc, "status_code")
    return (
        "openai" in error_type
        or "provider" in error_type
        or status_code in {400, 401, 403, 404, 429, 500, 502, 503, 504}
        or "404 page not found" in message
    )


def _exception_attr(exc: Exception, name: str) -> Any:
    """Safely read an exception attribute.

    输入 (Input):
        exc: 异常对象。
        name: 属性名。

    输出 (Output):
        Any: 属性值；读取失败时返回 None。

    Example Input:
        _exception_attr(exc, "status_code")

    Example Output:
        404
    """
    try:
        return getattr(exc, name)
    except Exception:  # noqa: BLE001 - 探测可选属性，任何异常都等价于「该属性不可用」
        return None


def _provider_runtime_error_message(exc: Exception) -> str:
    """生成模型服务请求失败时的用户提示。

    输入 (Input):
        exc: OpenAI-compatible provider 请求异常。

    输出 (Output):
        包含 404/provider 配置排查项的错误提示文本。

    Example Input:
        _provider_runtime_error_message(RuntimeError("404 page not found"))

    Example Output:
        "[模型服务错误] Provider 请求失败：404 page not found..."
    """
    status_code = _exception_attr(exc, "status_code")
    status = f"status={status_code}, " if status_code else ""
    config_diagnostics = _provider_config_diagnostics()
    return (
        f"[模型服务错误] Provider 请求失败：{status}{exc}\n"
        f"当前 Python: {sys.executable}\n"
        "真实 RAG 启动会先为 data/ 文档创建 embeddings；404 通常表示 "
        "embedding endpoint/base URL 或模型部署名不匹配。\n"
        f"{config_diagnostics}"
        "请检查以下配置是否指向同一个可用的 OpenAI 兼容服务：\n"
        "  1. OPENAI_BASE_URL / OPENAI_ENDPOINT / RAG_OPENAI_ENDPOINT\n"
        "  2. OPENAI_EMBEDDING_BASE_URL / OPENAI_EMBEDDING_ENDPOINT / "
        "RAG_OPENAI_EMBEDDING_ENDPOINT\n"
        "  3. OPENAI_EMBEDDING_MODEL 是否为服务端已部署的 embedding 模型\n"
        "如果只想验证 CLI 交互，不需要真实建库：\n"
        "  python3 chat_cli.py --echo\n"
    )


def _provider_config_diagnostics() -> str:
    """Return non-secret provider configuration diagnostics for CLI errors.

    输入 (Input):
        None。配置来自当前 RAGX 设置加载链路。

    输出 (Output):
        str: 不包含 API Key 的配置诊断；读取失败时返回空字符串。

    Example Input:
        _provider_config_diagnostics()

    Example Output:
        "当前配置诊断：未配置独立 embedding endpoint/base URL。\\n"
    """
    try:
        from config.settings import load_settings

        cfg = load_settings()
    except Exception:  # noqa: BLE001 - 仅用于补充提示信息，配置读取失败不应中断 CLI
        return ""
    endpoint = str(getattr(cfg, "endpoint", "") or "").rstrip("/")
    embedding_endpoint = getattr(cfg, "embedding_endpoint", None)
    embedding_base_url = getattr(cfg, "embedding_base_url", None)
    if not endpoint.endswith("/responses") or embedding_endpoint or embedding_base_url:
        return ""
    embedding_model = getattr(cfg, "embedding_model", "") or "<未配置>"
    return (
        "当前配置诊断：检测到 OPENAI_ENDPOINT/RAG_OPENAI_ENDPOINT 指向 "
        "`/responses`，但未配置独立 embedding endpoint/base URL。\n"
        f"当前 embedding model: {embedding_model}\n"
        "请在当前加载的 .env 中补充 `OPENAI_EMBEDDING_ENDPOINT` 或 "
        "`OPENAI_EMBEDDING_BASE_URL`，不要让 embedding 请求复用仅支持 "
        "chat/responses 的 endpoint。\n"
    )


if __name__ == "__main__":
    main()
