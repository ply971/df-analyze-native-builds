# Pure, framework-free log-capture helpers used by desktop/runner.py.
#
# No threading, no DearPyGui dependency here on purpose -- this is the part
# of the desktop app that's cheapest to unit-test in isolation.
from __future__ import annotations

import queue
import re


class QueueWriter:
    """A file-like object that just enqueues whatever is written to it.

    Used to redirect sys.stdout/sys.stderr into a background-thread-safe
    queue for the duration of a df-analyze run, so the GUI's log widget can
    poll the queue from the main/render thread instead of blocking on the
    run itself.
    """

    encoding = "utf-8"

    def __init__(self, q: "queue.Queue[str]") -> None:
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


_SPLIT_RE = re.compile(r"(\r|\n)")


class LineBuffer:
    """Turns a stream of raw text chunks (as produced by print()/tqdm's
    carriage-return-based progress updates) into a bounded, display-ready
    string.

    A '\\n' commits the current in-progress line to history. A '\\r'
    (tqdm-style in-place redraw) discards the in-progress line instead of
    committing it, so a progress bar's dozens of redraws collapse into one
    continuously-overwritten line rather than spamming the log with
    near-duplicate committed lines. Only a final '\\n' (e.g. tqdm's closing
    update) actually commits a progress line to history.
    """

    def __init__(self, max_lines: int = 300) -> None:
        self._lines: list[str] = []
        self._current: str = ""
        self._max_lines = max_lines

    def feed(self, chunk: str) -> None:
        for part in _SPLIT_RE.split(chunk):
            if part == "\n":
                self._lines.append(self._current)
                self._lines = self._lines[-self._max_lines :]
                self._current = ""
            elif part == "\r":
                self._current = ""
            elif part:
                self._current += part

    def render(self) -> str:
        return "\n".join([*self._lines, self._current])
