"""模块②Loading 渲染 —— 终端动态等待状态。

解耦设计：Spinner 只负责"在后台线程渲染动画"，不感知 LLM/RAG。
支持 with 上下文 与 显式 start/stop。无 TTY 时静默(只打印一次文字)。

对外接口：
  Spinner(text).start() / stop()
  with Spinner("思考中"): ...

依赖：标准库 threading/itertools/sys/time。
"""
from __future__ import annotations

import itertools
import sys
import threading
import time


class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, text: str = "AI 思考中", interval: float = 0.08) -> None:
        self._text = text
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tty = sys.stdout.isatty()

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{frame} {self._text}…")
            sys.stdout.flush()
            time.sleep(self._interval)

    def start(self) -> "Spinner":
        if not self._tty:
            sys.stdout.write(f"{self._text}…\n")  # 非交互环境只提示一次
            sys.stdout.flush()
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, clear: bool = True) -> None:
        if not self._tty:
            return
        self._stop.set()
        if self._thread:
            self._thread.join()
        if clear:
            sys.stdout.write("\r\x1b[K")  # 清除 loading 行
            sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
