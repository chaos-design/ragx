"""同步状态表 (Manifest) —— doc_id → {source_hash, version, updated_at}。

解耦设计：把"增量更新依赖的状态"从 IncrementalSyncer 中抽出为可替换组件。
  - ManifestStore：协议(load/save 全量字典)，syncer 只依赖它。
  - InMemoryManifest：默认,进程内,等价旧行为(测试/一次性建库)。
  - JsonFileManifest：落地为 JSON 文件,使"文档级跳过/版本递增"跨进程生效
    —— 配合持久化向量库(sqlite/pgvector),CLI 重启后真正做增量更新而非全量重建。

生产环境应替换为数据库表(MySQL/PG),此处给出零依赖的文件实现作示范。

对外接口：
  ManifestStore.load() -> dict[str, dict]
  ManifestStore.save(manifest: dict[str, dict]) -> None

依赖：仅标准库 json/os。
"""
from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class ManifestStore(Protocol):
    def load(self) -> dict[str, dict]: ...
    def save(self, manifest: dict[str, dict]) -> None: ...


class InMemoryManifest:
    """进程内状态:不持久化。等价于旧 syncer 内置 dict。"""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    def load(self) -> dict[str, dict]:
        return self._data

    def save(self, manifest: dict[str, dict]) -> None:
        self._data = manifest


class JsonFileManifest:
    """JSON 文件状态:跨进程持久化,让增量更新在 CLI 重启后仍生效。"""

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> dict[str, dict]:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, manifest: dict[str, dict]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)) or ".", exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)  # 原子替换,避免半写状态
