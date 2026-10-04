"""PDF 文档读取与 OCR 兜底。"""
from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag.ingestion.chunker import Block, BlockType
from rag.ingestion.layout import analyze_pdf_text_page

from .utils import bounded_env_int

CJK_PUNCTUATION = r"，。！？；：、（）《》“”‘’"
CJK_TEXT_CLUSTER = rf"\u4e00-\u9fff{CJK_PUNCTUATION}"


def read_pdf(path: Path) -> str:
    """读取带文本层的 PDF。

    输入 (Input):
        path: PDF 文件路径。

    输出 (Output):
        按页拼接的文本，页码以 Markdown 标题注入，便于后续检索溯源。

    Example Input:
        read_pdf(Path("agent-library/ragx/data/resources/statement.pdf"))

    Example Output:
        "# statement.pdf\\n\\n## Page 1\\n..."
    """
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError("读取 PDF 需要安装 PyMuPDF：python3 -m pip install pymupdf") from exc

    pages: list[str] = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            blocks = analyze_pdf_text_page(page, page_no)
            text = _format_pdf_page(blocks)
            if not text:
                text = _ocr_pdf_page(page, path, page_no)
            if text:
                pages.append(f"## Page {page_no}\n{text}")
    if not pages:
        raise ValueError(f"PDF 未抽取到可索引文本: {path}")
    return f"# {path.name}\n\n" + "\n\n".join(pages)


def _format_pdf_page(blocks: list[Block]) -> str:
    """把按阅读顺序排序后的 PDF Block 格式化为 Markdown 文本。

    输入 (Input):
        blocks: `analyze_pdf_text_page` 返回的页面块。

    输出 (Output):
        页面 Markdown 文本。标题块保留为 Markdown 标题，正文按块空行分隔。

    Example Input:
        _format_pdf_page([Block(BlockType.HEADING, "标题"), Block(BlockType.TEXT, "正文")])

    Example Output:
        "### 标题\\n\\n正文"
    """
    parts: list[str] = []
    for block in blocks:
        text = _normalize_pdf_text(block.text)
        if not text:
            continue
        if block.type is BlockType.HEADING:
            parts.append(f"### {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _ocr_pdf_page(page: Any, path: Path, page_no: int) -> str:
    """对无文本层 PDF 页执行真实 OCR。

    输入 (Input):
        page: PyMuPDF Page 实例。
        path: PDF 文件路径，用于错误提示。
        page_no: 页码。

    输出 (Output):
        OCR 文本。若 OCR 未识别出内容则返回空字符串。

    Example Input:
        _ocr_pdf_page(page, Path("agent-library/ragx/data/resources/statement.pdf"), 1)

    Example Output:
        "识别出的页面文本"
    """
    _paddle_ocr_client()

    image_path = ""
    prepared_path = ""
    try:
        pixmap = page.get_pixmap(matrix=_ocr_matrix(), alpha=False)
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
            dir=os.getcwd(),
        ) as handle:
            image_path = handle.name
        pixmap.save(image_path)
        text = _run_paddle_ocr(image_path, path, page_no)
        if text:
            return text
        prepared_path = _prepare_ocr_image(image_path)
        if prepared_path and prepared_path != image_path:
            return _run_paddle_ocr(prepared_path, path, page_no)
        return ""
    finally:
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
        if prepared_path and os.path.exists(prepared_path):
            os.unlink(prepared_path)


def _run_paddle_ocr(
    image_path: str,
    path: Path,
    page_no: int,
) -> str:
    """调用 PaddleOCR 识别图片。

    输入 (Input):
        image_path: 待识别图片路径。
        path: 原 PDF 路径，用于错误提示。
        page_no: 页码。

    输出 (Output):
        OCR 文本。

    Example Input:
        _run_paddle_ocr("page.png", Path("a.pdf"), 1)

    Example Output:
        "text"
    """
    client = _paddle_ocr_client()
    try:
        result = _call_paddle_ocr(client, image_path)
    except Exception as exc:
        raise RuntimeError(
            f"OCR 失败: {path} page={page_no}, error={exc}"
        ) from exc
    return _extract_paddle_text(result)


def _prepare_ocr_image(image_path: str) -> str:
    """生成更适合 OCR 的预处理图片。

    输入 (Input):
        image_path: 原始渲染图片路径。

    输出 (Output):
        预处理后的临时图片路径。若 Pillow 不可用则返回空字符串。

    Example Input:
        _prepare_ocr_image("page.png")

    Example Output:
        "/current/working/dir/tmpxxx.png"
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover - 取决于运行环境
        return ""

    with Image.open(image_path) as source:
        image = source.convert("L")
    width, height = image.size
    crop_box = (
        int(width * 0.10),
        int(height * 0.08),
        int(width * 0.90),
        int(height * 0.92),
    )
    image = image.crop(crop_box)
    image = ImageOps.autocontrast(image)
    threshold = _ocr_threshold()
    image = image.point(lambda pixel: 0 if pixel < threshold else 255)
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
        dir=os.getcwd(),
    ) as handle:
        prepared_path = handle.name
    image.save(prepared_path)
    return prepared_path


def _ocr_threshold() -> int:
    """读取 OCR 二值化阈值。

    输入 (Input):
        None。可通过 RAG_OCR_THRESHOLD 覆盖。

    输出 (Output):
        0-255 范围内的阈值。

    Example Input:
        _ocr_threshold()

    Example Output:
        170
    """
    return bounded_env_int("RAG_OCR_THRESHOLD", 170, 0, 255)


def _ocr_lang() -> str:
    """选择 OCR 语言。

    输入 (Input):
        None。可通过 RAG_OCR_LANG 覆盖。

    输出 (Output):
        PaddleOCR 语言参数。默认 `ch`。

    Example Input:
        _ocr_lang()

    Example Output:
        "ch"
    """
    configured = os.getenv("RAG_OCR_LANG")
    return configured.strip() if configured and configured.strip() else "ch"


@lru_cache(maxsize=1)
def _paddle_ocr_client() -> Any:
    """创建并缓存 PaddleOCR 客户端。

    输入 (Input):
        None。语言由 `RAG_OCR_LANG` 控制，默认 `ch`。

    输出 (Output):
        PaddleOCR 客户端。

    Example Input:
        _paddle_ocr_client()

    Example Output:
        PaddleOCR(...)
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        if _is_missing_paddle_runtime(exc):
            raise RuntimeError(_paddle_runtime_dependency_message()) from exc
        raise RuntimeError(
            "PDF 页面无文本层，且未安装 PaddleOCR。请安装 PaddleOCR "
            "后重试，或先提供可抽取文本的 PDF。"
        ) from exc

    try:
        return _create_paddle_ocr(PaddleOCR, _ocr_lang())
    except RuntimeError as exc:
        if _is_missing_paddle_runtime(exc):
            raise RuntimeError(_paddle_runtime_dependency_message()) from exc
        raise


def _is_missing_paddle_runtime(exc: BaseException) -> bool:
    """Detect PaddleOCR failures caused by a missing PaddlePaddle runtime.

    输入 (Input):
        exc: PaddleOCR import or initialization exception.

    输出 (Output):
        bool: True 表示异常可归因于缺少 paddlepaddle 运行时。

    Example Input:
        _is_missing_paddle_runtime(ModuleNotFoundError("No module named 'paddle'"))

    Example Output:
        True
    """
    missing_name = str(getattr(exc, "name", "") or "").lower()
    message = str(exc).lower()
    return (
        missing_name in {"paddle", "paddlepaddle"}
        or "paddlepaddle" in message
        or "paddle_static" in message
    )


def _paddle_runtime_dependency_message() -> str:
    """Return an actionable PaddlePaddle dependency error message.

    输入 (Input):
        None.

    输出 (Output):
        str: 面向 CLI 用户的依赖修复提示。

    Example Input:
        _paddle_runtime_dependency_message()

    Example Output:
        "PDF 页面无文本层，PaddleOCR 已安装但缺少 paddlepaddle 运行时。..."
    """
    return (
        "PDF 页面无文本层，PaddleOCR 已安装但缺少 paddlepaddle 运行时。"
        "请在当前 Python 环境安装 requirements.txt 中的 paddlepaddle 后重试。"
    )


def _create_paddle_ocr(paddle_ocr_cls: Any, lang: str) -> Any:
    """兼容不同 PaddleOCR 版本创建客户端。

    输入 (Input):
        paddle_ocr_cls: `paddleocr.PaddleOCR` 类。
        lang: PaddleOCR 语言参数。

    输出 (Output):
        PaddleOCR 客户端实例。

    Example Input:
        _create_paddle_ocr(PaddleOCR, "ch")

    Example Output:
        PaddleOCR(...)
    """
    configs = (
        {"lang": lang, "use_angle_cls": True, "show_log": False},
        {"lang": lang, "use_textline_orientation": True},
        {"lang": lang},
    )
    last_error: TypeError | ValueError | None = None
    for config in configs:
        try:
            return paddle_ocr_cls(**config)
        except (TypeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return paddle_ocr_cls()


def _call_paddle_ocr(client: Any, image_path: str) -> Any:
    """调用 PaddleOCR 客户端并兼容 v2/v3 API。

    输入 (Input):
        client: PaddleOCR 客户端。
        image_path: 待识别图片路径。

    输出 (Output):
        PaddleOCR 原始识别结果。

    Example Input:
        _call_paddle_ocr(client, "page.png")

    Example Output:
        [[..., ("text", 0.99)]]
    """
    if hasattr(client, "ocr"):
        try:
            return client.ocr(image_path, cls=True)
        except TypeError:
            return client.ocr(image_path)
    if hasattr(client, "predict"):
        return client.predict(image_path)
    raise TypeError("PaddleOCR client does not expose ocr or predict")


def _extract_paddle_text(result: Any) -> str:
    """从 PaddleOCR 结果中提取文本。

    输入 (Input):
        result: PaddleOCR v2/v3 原始返回值。

    输出 (Output):
        按行拼接后的 OCR 文本。

    Example Input:
        [[[[0, 0]], ("hello", 0.99)]]

    Example Output:
        "hello"
    """
    lines: list[str] = []
    _collect_paddle_text(result, lines)
    normalized_lines = [_normalize_pdf_text(line) for line in lines]
    return "\n".join(line for line in normalized_lines if line).strip()


def _normalize_pdf_text(text: str) -> str:
    """Normalize extracted PDF text without changing meaningful word spacing.

    输入 (Input):
        text: PDF text layer or OCR line text.

    输出 (Output):
        str: 规范化后的文本，移除中文字符之间的误识别空格。

    Example Input:
        _normalize_pdf_text("记 忆 系 统 and Agent")

    Example Output:
        "记忆系统 and Agent"
    """
    compacted = re.sub(
        rf"(?<=[{CJK_TEXT_CLUSTER}])\s+(?=[{CJK_TEXT_CLUSTER}])",
        "",
        text.strip(),
    )
    return re.sub(rf"(?<=[{CJK_PUNCTUATION}])\s+(?=\S)", "", compacted)


def _collect_paddle_text(value: Any, lines: list[str]) -> None:
    """递归收集 PaddleOCR 返回结构中的文本。"""
    if value is None:
        return
    if isinstance(value, str):
        return
    if isinstance(value, dict):
        _collect_paddle_mapping_text(value, lines)
        return
    if isinstance(value, (list, tuple)):
        if _append_text_score_pair(value, lines):
            return
        for item in value:
            _collect_paddle_text(item, lines)
        return
    for method_name in ("to_dict", "json"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            _collect_paddle_text(method(), lines)
            return
        except TypeError:
            continue


def _collect_paddle_mapping_text(value: dict[Any, Any], lines: list[str]) -> None:
    """从 PaddleOCR 字典结构中提取文本。"""
    for key in ("rec_texts", "texts"):
        entries = value.get(key)
        if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes)):
            for entry in entries:
                text = str(entry).strip()
                if text:
                    lines.append(text)
    for key in ("text", "transcription"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    for item in value.values():
        _collect_paddle_text(item, lines)


def _append_text_score_pair(value: list[Any] | tuple[Any, ...], lines: list[str]) -> bool:
    """识别 `(text, score)` 或 `(box, (text, score))` 结构。"""
    if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], (int, float)):
        lines.append(value[0].strip())
        return True
    if len(value) >= 2 and isinstance(value[1], (list, tuple)):
        nested = value[1]
        if (
            nested
            and isinstance(nested[0], str)
            and not isinstance(value[0], (dict, str))
            and (len(nested) == 1 or isinstance(nested[1], (int, float)))
        ):
            lines.append(nested[0].strip())
            return True
    return False


def _ocr_matrix() -> Any:
    """返回 OCR 渲染矩阵。

    输入 (Input):
        None。可通过 RAG_OCR_ZOOM 覆盖缩放倍数。

    输出 (Output):
        PyMuPDF Matrix 实例。

    Example Input:
        _ocr_matrix()

    Example Output:
        fitz.Matrix(2.0, 2.0)
    """
    import fitz

    try:
        zoom = float(os.getenv("RAG_OCR_ZOOM", "2.0"))
    except ValueError:
        zoom = 2.0
    return fitz.Matrix(zoom, zoom)
