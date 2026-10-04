"""真实数据读取验证：目录枚举、Markdown、DOCX 与异常边界。"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.ingestion.chunker import Block, BlockType
from rag.ingestion.loader import (
    TextFileLoader,
    read_source_documents,
)
from rag.ingestion.loader.pdf import (
    _call_paddle_ocr,
    _create_paddle_ocr,
    _extract_paddle_text,
    _format_pdf_page,
    _normalize_pdf_text,
    _ocr_lang,
    _ocr_matrix,
    _ocr_pdf_page,
    _ocr_threshold,
    _paddle_ocr_client,
    _paddle_runtime_dependency_message,
    _prepare_ocr_image,
    _run_paddle_ocr,
    read_pdf,
)

DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>AI Agent Engineer</w:t></w:r></w:p>
    <w:p><w:r><w:t>真实 DOCX 数据进入索引。</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>字段</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>说明</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>provider</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>agent_provider</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_docx(path: str) -> None:
    with zipfile.ZipFile(path, "w") as docx:
        docx.writestr("word/document.xml", DOCX_XML)


def test_read_source_documents_with_real_docx_and_markdown():
    with tempfile.TemporaryDirectory() as root:
        _write(os.path.join(root, "rag-sample.md"), "# RAG\n真实 Markdown 数据。\n")
        _write(os.path.join(root, "raw.json"), '{"title": "未知后缀也进入索引"}')
        _write_docx(os.path.join(root, "ai-agent-engineer.docx"))
        os.makedirs(os.path.join(root, "stores"))
        _write(os.path.join(root, "stores", "ignored.md"), "# should not index")

        documents = read_source_documents(root)

    assert sorted(documents) == ["ai-agent-engineer.docx", "rag-sample.md", "raw.json"]
    assert "真实 Markdown 数据" in documents["rag-sample.md"]
    assert "未知后缀也进入索引" in documents["raw.json"]
    assert "AI Agent Engineer" in documents["ai-agent-engineer.docx"]
    assert "| provider | agent_provider |" in documents["ai-agent-engineer.docx"]
    print("✓ loader: markdown/docx real source reading ok")


def test_read_source_documents_reads_unknown_suffix_as_text():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "raw.json")
        _write(path, '{"name": "RAGX"}')
        documents = read_source_documents(path)

    assert documents == {"raw.json": '{"name": "RAGX"}'}
    print("✓ loader: unknown suffix single file read as text")


def test_read_source_documents_single_file_and_missing_path():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "single.txt")
        _write(path, "真实 TXT 数据")
        documents = read_source_documents(path)
        assert documents == {"single.txt": "真实 TXT 数据"}

        try:
            read_source_documents(os.path.join(root, "missing.md"))
        except FileNotFoundError as exc:
            assert "数据源不存在" in str(exc)
        else:
            raise AssertionError("missing source should raise FileNotFoundError")
    print("✓ loader: single file and missing path boundaries ok")


def test_read_source_documents_rejects_empty_directory():
    with tempfile.TemporaryDirectory() as root:
        try:
            read_source_documents(root)
        except ValueError as exc:
            assert "未包含可读取的真实文档" in str(exc)
        else:
            raise AssertionError("empty directory should raise ValueError")
    print("✓ loader: empty directory rejected")


def test_text_file_loader_keeps_legacy_chunking_contract():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "legacy.md")
        _write(path, "abcdef")
        docs = TextFileLoader(chunk_size=3, chunk_overlap=1).load(root)

    assert [doc.content for doc in docs] == ["abc", "cde", "ef"]
    assert all(doc.metadata["source"].endswith("legacy.md") for doc in docs)
    print("✓ loader: legacy TextFileLoader chunking ok")


def test_read_pdf_uses_layout_order_for_double_columns(monkeypatch, tmp_path: Path):
    """验证真实 PDF loader 会复用版面排序，避免双栏文本左右交错。"""

    def block(text: str, bbox: tuple[int, int, int, int], size: int = 10) -> dict:
        return {
            "type": 0,
            "bbox": bbox,
            "lines": [{"spans": [{"text": text, "size": size}]}],
        }

    raw = {
        "blocks": [
            block("标题", (40, 20, 560, 50), size=18),
            block("左一", (40, 100, 260, 130)),
            block("右一", (330, 100, 550, 130)),
            block("左二", (40, 160, 260, 190)),
            block("右二", (330, 160, 550, 190)),
        ]
    }

    class FakePage:
        rect = SimpleNamespace(width=600)

        def get_text(self, mode):
            assert mode == "dict"
            return raw

    class FakeDoc:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([FakePage()])

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(open=lambda path: FakeDoc()),
    )

    text = read_pdf(tmp_path / "double-column.pdf")

    assert text.index("左一") < text.index("左二") < text.index("右一") < text.index("右二")
    assert "### 标题" in text


def test_read_pdf_ocr_fallback_and_empty_document(monkeypatch, tmp_path: Path):
    """验证 PDF 页无文本块时会走 OCR，完全无内容时抛出明确错误。"""

    class EmptyTextPage:
        rect = SimpleNamespace(width=600)

        def get_text(self, mode):
            assert mode == "dict"
            return {"blocks": []}

    class OnePageDoc:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([EmptyTextPage()])

    class EmptyDoc(OnePageDoc):
        def __iter__(self):
            return iter([])

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(open=lambda path: OnePageDoc()),
    )
    with patch("rag.ingestion.loader.pdf._ocr_pdf_page", return_value="OCR fallback"):
        text = read_pdf(tmp_path / "scan.pdf")
    assert "OCR fallback" in text

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(open=lambda path: EmptyDoc()),
    )
    try:
        read_pdf(tmp_path / "empty.pdf")
    except ValueError as exc:
        assert "PDF 未抽取到可索引文本" in str(exc)
    else:
        raise AssertionError("empty PDF should raise ValueError")


def test_format_pdf_page_skips_empty_blocks():
    """验证 PDF Block 格式化会保留标题并跳过空块。"""
    text = _format_pdf_page(
        [
            Block(BlockType.HEADING, "记 忆 系 统"),
            Block(BlockType.TEXT, ""),
            Block(BlockType.TEXT, "中 文 正 文"),
        ]
    )

    assert text == "### 记忆系统\n\n中文正文"


def test_ocr_pdf_page_uses_paddleocr_sdk(monkeypatch):
    class FakePixmap:
        def save(self, path):
            _write(path, "image")

    class FakePage:
        def get_pixmap(self, matrix, alpha):
            assert matrix is not None
            assert alpha is False
            return FakePixmap()

    created: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def ocr(self, image_path, cls=True):
            assert image_path.endswith(".png")
            assert cls is True
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("OCR text", 0.99)]]

    _paddle_ocr_client.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=FakePaddleOCR),
    )
    with patch("rag.ingestion.loader.pdf._ocr_matrix", return_value=object()):
        text = _ocr_pdf_page(FakePage(), Path("scan.pdf"), 1)
    _paddle_ocr_client.cache_clear()

    assert text == "OCR text"
    assert created["kwargs"]["lang"] == "ch"
    print("✓ loader: OCR fallback uses PaddleOCR SDK")


def test_ocr_pdf_page_retries_with_preprocessed_image(tmp_path: Path):
    """验证首次 OCR 为空时会重试预处理图片并清理临时文件。"""

    class FakePixmap:
        def save(self, path):
            _write(path, "image")

    class FakePage:
        def get_pixmap(self, matrix, alpha):
            assert matrix is not None
            assert alpha is False
            return FakePixmap()

    prepared = tmp_path / "prepared.png"
    prepared.write_text("prepared", encoding="utf-8")

    with (
        patch("rag.ingestion.loader.pdf._paddle_ocr_client", return_value=object()),
        patch("rag.ingestion.loader.pdf._ocr_matrix", return_value=object()),
        patch(
            "rag.ingestion.loader.pdf._prepare_ocr_image",
            return_value=str(prepared),
        ),
        patch(
            "rag.ingestion.loader.pdf._run_paddle_ocr",
            side_effect=["", "prepared text"],
        ) as run,
    ):
        text = _ocr_pdf_page(FakePage(), Path("scan.pdf"), 1)

    assert text == "prepared text"
    assert run.call_count == 2
    assert not prepared.exists()


def test_ocr_pdf_page_requires_paddleocr(monkeypatch):
    _paddle_ocr_client.cache_clear()
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    try:
        _ocr_pdf_page(object(), Path("scan.pdf"), 1)
    except RuntimeError as exc:
        assert "未安装 PaddleOCR" in str(exc)
    else:
        raise AssertionError("missing PaddleOCR should raise RuntimeError")
    finally:
        _paddle_ocr_client.cache_clear()
    print("✓ loader: OCR requires PaddleOCR")


def test_paddle_ocr_client_reports_missing_paddlepaddle(monkeypatch):
    """验证 PaddleOCR 已安装但缺少 paddlepaddle 时提示可执行修复命令。"""

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            raise RuntimeError(
                "Engine 'paddle_static' is unavailable because dependency "
                "'paddlepaddle' is not installed."
            )

    _paddle_ocr_client.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=FakePaddleOCR),
    )
    try:
        _paddle_ocr_client()
    except RuntimeError as exc:
        message = str(exc)
        assert "缺少 paddlepaddle 运行时" in message
        assert "requirements.txt" in message
    else:
        raise AssertionError("missing paddlepaddle should raise RuntimeError")
    finally:
        _paddle_ocr_client.cache_clear()

    assert "paddlepaddle" in _paddle_runtime_dependency_message()


def test_ocr_lang_prefers_config_then_default():
    with patch.dict(os.environ, {"RAG_OCR_LANG": "eng"}):
        assert _ocr_lang() == "eng"
    with patch.dict(os.environ, {"RAG_OCR_LANG": "  "}):
        assert _ocr_lang() == "ch"
    with patch.dict(os.environ, {}, clear=True):
        assert _ocr_lang() == "ch"
    print("✓ loader: OCR language selection ok")


def test_create_paddle_ocr_compatibility_fallbacks():
    calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "use_angle_cls" in kwargs:
                raise TypeError("old arg unsupported")
            self.kwargs = kwargs

    client = _create_paddle_ocr(FakePaddleOCR, "en")

    assert client.kwargs == {"lang": "en", "use_textline_orientation": True}
    assert calls[0]["use_angle_cls"] is True


def test_run_paddle_ocr_success_and_failure():
    class FakeClient:
        def ocr(self, image_path, cls=True):
            assert cls is True
            return [[[[0, 0]], (" text ", 0.99)]]

    with patch("rag.ingestion.loader.pdf._paddle_ocr_client", return_value=FakeClient()):
        assert _run_paddle_ocr("page.png", Path("a.pdf"), 1) == "text"

    class FailingClient:
        def ocr(self, image_path, cls=True):
            raise ValueError("bad image")

    with patch("rag.ingestion.loader.pdf._paddle_ocr_client", return_value=FailingClient()):
        try:
            _run_paddle_ocr("page.png", Path("a.pdf"), 2)
        except RuntimeError as exc:
            assert "bad image" in str(exc)
        else:
            raise AssertionError("PaddleOCR failure should raise RuntimeError")
    print("✓ loader: PaddleOCR success/failure paths ok")


def test_paddle_ocr_result_parsing_and_predict_api():
    assert _extract_paddle_text({"rec_texts": ["A", "B"]}) == "A\nB"
    assert _extract_paddle_text([{"text": "C"}, ("D", 0.8)]) == "C\nD"
    assert _extract_paddle_text({"rec_texts": ["记 忆 系 统", "AI Agent"]}) == (
        "记忆系统\nAI Agent"
    )
    assert _normalize_pdf_text("知 识 库 ： Agent") == "知识库：Agent"

    class PredictClient:
        def predict(self, image_path):
            return [{"rec_texts": ["predict text"]}]

    assert _extract_paddle_text(_call_paddle_ocr(PredictClient(), "page.png")) == (
        "predict text"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None,
    reason="Pillow 未安装（可选 OCR 图像预处理依赖）",
)
def test_prepare_ocr_image_and_threshold_env():
    from PIL import Image

    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "source.png")
        Image.new("RGB", (100, 120), color=(180, 180, 180)).save(source)
        with patch.dict(os.environ, {"RAG_OCR_THRESHOLD": "999"}):
            assert _ocr_threshold() == 255
        with patch.dict(os.environ, {"RAG_OCR_THRESHOLD": "bad"}):
            assert _ocr_threshold() == 170

        prepared = _prepare_ocr_image(source)
        try:
            assert os.path.exists(prepared)
            assert prepared.endswith(".png")
        finally:
            if prepared and os.path.exists(prepared):
                os.unlink(prepared)
    print("✓ loader: OCR image preprocessing and threshold env ok")


def test_ocr_matrix_uses_configured_zoom(monkeypatch):
    """验证 OCR 渲染矩阵支持环境变量，并在非法输入时回退默认值。"""
    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(Matrix=lambda x, y: (x, y)),
    )

    with patch.dict(os.environ, {"RAG_OCR_ZOOM": "3.5"}):
        assert _ocr_matrix() == (3.5, 3.5)
    with patch.dict(os.environ, {"RAG_OCR_ZOOM": "bad"}):
        assert _ocr_matrix() == (2.0, 2.0)


if __name__ == "__main__":
    test_read_source_documents_with_real_docx_and_markdown()
    test_read_source_documents_reads_unknown_suffix_as_text()
    test_read_source_documents_single_file_and_missing_path()
    test_read_source_documents_rejects_empty_directory()
    test_text_file_loader_keeps_legacy_chunking_contract()
    test_ocr_lang_prefers_config_then_default()
    test_create_paddle_ocr_compatibility_fallbacks()
    test_run_paddle_ocr_success_and_failure()
    test_paddle_ocr_result_parsing_and_predict_api()
    test_prepare_ocr_image_and_threshold_env()
    print("\n✓✓✓ loader tests passed")
