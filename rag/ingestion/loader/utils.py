"""loader 通用工具函数。"""
from __future__ import annotations

import os
from pathlib import Path

IGNORED_SOURCE_DIRS = {"stores", "__pycache__"}


def iter_source_paths(source_path: Path) -> list[Path]:
    """枚举数据源文件。

    输入 (Input):
        source_path: 已解析的源路径。

    输出 (Output):
        按路径排序后的文件列表。

    Example Input:
        iter_source_paths(Path("agent-library/ragx/data/resources"))

    Example Output:
        [Path("agent-library/ragx/data/resources/rag-sample.md")]
    """
    if source_path.is_file():
        return [source_path]

    paths: list[Path] = []
    for root, dirs, files in os.walk(source_path):
        dirs[:] = [
            dirname
            for dirname in dirs
            if dirname not in IGNORED_SOURCE_DIRS and not dirname.startswith(".")
        ]
        root_path = Path(root)
        paths.extend(root_path / filename for filename in files)
    return sorted(paths)


def make_doc_id(path: Path, source_path: Path) -> str:
    """生成稳定 doc_id。

    输入 (Input):
        path: 文件路径。
        source_path: 原始源路径。

    输出 (Output):
        目录读取时返回相对路径，单文件读取时返回文件名。

    Example Input:
        make_doc_id(Path("agent-library/ragx/data/a.md"), Path("agent-library/ragx/data"))

    Example Output:
        "a.md"
    """
    if source_path.is_dir():
        return path.relative_to(source_path).as_posix()
    return path.name


def read_text_file(path: Path) -> str:
    """按 UTF-8 文本读取文件。

    输入 (Input):
        path: 源文件路径。

    输出 (Output):
        文件文本，无法解码的字节会被忽略。

    Example Input:
        read_text_file(Path("agent-library/ragx/data/resources/rag-sample.md"))

    Example Output:
        "# RAG\\n..."
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """读取整数环境变量并限制取值范围。

    输入 (Input):
        name: 环境变量名。
        default: 缺省值或非法值回退值。
        minimum: 允许的最小值。
        maximum: 允许的最大值。

    输出 (Output):
        位于 [minimum, maximum] 区间内的整数。

    Example Input:
        bounded_env_int("RAG_OCR_THRESHOLD", 170, 0, 255)

    Example Output:
        170
    """
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))
