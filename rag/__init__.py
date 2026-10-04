"""RAGX package."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_agent_library_on_path() -> None:
    """Ensure the shared agent-library package root is importable.

    Input:
        None.

    Output:
        None. The agent-library root is added to sys.path when necessary.

    Example:
        _ensure_agent_library_on_path()
    """
    agent_library_root = Path(__file__).resolve().parents[2]
    root_text = str(agent_library_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


_ensure_agent_library_on_path()
