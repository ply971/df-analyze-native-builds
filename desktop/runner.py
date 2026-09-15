"""Run analysis in a cancellable process in source and packaged applications."""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

from df_analyze import gui_common as gc

from .log_stream import QueueWriter

DoneResult = Tuple[str, str]
ANALYZE_SCRIPT = Path(__file__).resolve().parents[1] / "df-analyze.py"
_IN_PROCESS_LOCK = threading.Lock()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Stop the analysis and the joblib/model workers it started."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass


@dataclass
class _ProcessState:
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: Optional[subprocess.Popen[str]] = None
    stopping: bool = False

    def stop_if_requested(self) -> None:
        with self.lock:
            process = self.process
            if (
                not self.cancel_requested.is_set()
                or self.stopping
                or process is None
                or process.poll() is not None
            ):
                return
            self.stopping = True
        threading.Thread(
            target=_terminate_process,
            args=(process,),
            daemon=True,
            name="df-analyze-cancel",
        ).start()


@dataclass
class RunHandle:
    thread: threading.Thread
    log_queue: "queue.Queue[str]"
    done_queue: "queue.Queue[DoneResult]"
    _process_state: Optional[_ProcessState] = field(default=None, repr=False)

    @property
    def can_cancel(self) -> bool:
        state = self._process_state
        return (
            state is not None
            and self.thread.is_alive()
            and not state.cancel_requested.is_set()
        )

    def cancel(self) -> bool:
        """Request cancellation without blocking the GUI; false if unavailable."""
        state = self._process_state
        if state is None or not self.can_cancel:
            return False
        state.cancel_requested.set()
        self.log_queue.put("\nCancelling analysis and its worker processes...\n")
        state.stop_if_requested()
        return True


def start_run(
    cfg: gc.RunConfig,
    main_fn: Optional[Callable[[], None]] = None,
) -> RunHandle:
    """Return progress queues consumed by the GUI's render loop.

    ``main_fn`` is an injectable in-process entry point for tests. Real source
    runs use a subprocess; packaged runs dispatch the same executable in worker
    mode so they also support cancellation without changing GUI process globals.
    """
    log_q: "queue.Queue[str]" = queue.Queue()
    done_q: "queue.Queue[DoneResult]" = queue.Queue()
    argv = ["df-analyze.py", *gc.build_argv(cfg)]
    state = None
    if main_fn is None:
        state = _ProcessState()
        thread = threading.Thread(
            target=_run_process,
            args=(argv[1:], log_q, done_q, state),
            daemon=True,
            name="df-analyze-run",
        )
    else:
        thread = threading.Thread(
            target=_run_worker,
            args=(argv, main_fn, log_q, done_q),
            daemon=True,
            name="df-analyze-run",
        )
    handle = RunHandle(thread, log_q, done_q, state)
    thread.start()
    return handle


def _run_process(
    args: list[str],
    log_q: "queue.Queue[str]",
    done_q: "queue.Queue[DoneResult]",
    state: _ProcessState,
    script: Optional[Path] = None,
) -> None:
    script = script or ANALYZE_SCRIPT
    process: Optional[subprocess.Popen[str]] = None
    try:
        if state.cancel_requested.is_set():
            done_q.put(("cancelled", "Analysis cancelled."))
            return
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        # Avoid multiplying BLAS threads across each joblib worker.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        log_q.put("Starting analysis engine. The first import can take a moment...\n")
        if getattr(sys, "frozen", False):
            mode = "--embedding-worker" if script.name == "df-embed.py" else "--analysis-worker"
            command = [sys.executable, mode, *args]
        else:
            command = [sys.executable, "-u", str(script), *args]
        process = subprocess.Popen(
            command,
            cwd=script.parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        with state.lock:
            state.process = process
        state.stop_if_requested()
        if process.stdout is not None:
            with process.stdout:
                for line in process.stdout:
                    log_q.put(line)
        code = process.wait()
        if state.cancel_requested.is_set():
            done_q.put(("cancelled", "Analysis cancelled."))
        elif code == 0:
            done_q.put(("success", ""))
        else:
            done_q.put(
                (
                    "error",
                    f"Analysis exited with code {code}. See the run log for details.",
                )
            )
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _terminate_process(process)
        if state.cancel_requested.is_set():
            done_q.put(("cancelled", "Analysis cancelled."))
        else:
            done_q.put(
                ("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            )


def start_embedding(cfg) -> RunHandle:
    """Run the embedding CLI with the same log and cancellation lifecycle."""
    args = cfg.argv()
    log_q = queue.Queue()
    done_q = queue.Queue()
    state = _ProcessState()
    thread = threading.Thread(target=_run_process,
                              args=(args, log_q, done_q, state, ANALYZE_SCRIPT.with_name("df-embed.py")), daemon=True)
    handle = RunHandle(thread, log_q, done_q, state)
    thread.start()
    return handle


def _run_worker(
    argv: list[str],
    main_fn: Optional[Callable[[], None]],
    log_q: "queue.Queue[str]",
    done_q: "queue.Queue[DoneResult]",
) -> None:
    # Redirecting process globals requires serialization for frozen/injected runs.
    with _IN_PROCESS_LOCK:
        old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
        stream = QueueWriter(log_q)
        warning_logger = logging.getLogger("py.warnings")
        old_handlers = list(warning_logger.handlers)
        warning_logger.handlers.clear()
        sys.argv, sys.stdout, sys.stderr = argv, stream, stream
        result: DoneResult
        try:
            if main_fn is None:
                # Preserve the entry script's torch-before-transformers ordering.
                import torch  # noqa: F401
                from joblib import parallel_config

                from df_analyze._main import main

                # Joblib configuration is thread-local. Set it in this worker,
                # where frozen builds cannot safely spawn more GUI executables.
                with parallel_config(backend="threading"):
                    main()
            else:
                main_fn()
            result = ("success", "")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            result = (
                "error",
                f"df-analyze exited early (code={code}). This usually means "
                "invalid or conflicting CLI arguments -- check the log above.",
            )
        except BaseException as exc:
            result = ("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        finally:
            sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr
            warning_logger.handlers[:] = old_handlers
        # Completion follows restoration so UI output is never captured by an
        # already completed run's stream.
        done_q.put(result)
