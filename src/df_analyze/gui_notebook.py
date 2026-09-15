"""A real, local Jupyter kernel shared by the desktop and web notebooks.

Code executes in a separate Python process, never in a GUI callback. Imported
notebooks are data until the user explicitly runs a cell. This is a local coding
environment with the user's filesystem permissions, not a security sandbox.
"""
from __future__ import annotations

import atexit
import copy
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from df_analyze import gui_common as gc

_SESSIONS = set()


def run_analysis(config: gc.RunConfig, **changes):
    """Run the real pipeline from a cell; return its leaderboard DataFrame."""
    import torch  # noqa: F401 -- preserve CLI import order
    from df_analyze._main import main

    cfg = replace(config, **changes)
    errors = gc.validate_config(cfg)
    if errors:
        raise ValueError("\n".join(errors))
    cfg = replace(cfg, outdir=cfg.outdir / ("notebook_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]))
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    (cfg.outdir / "studio_experiment.json").write_text(gc.config_to_json(cfg), encoding="utf-8")
    if cfg.pipeline_steps or cfg.drop_duplicates:
        frame = gc.load_full_dataset(cfg.data_path, cfg.separator, cfg.input_mode)
        if cfg.drop_duplicates:
            frame, _ = gc.drop_duplicate_rows(frame)
        frame, summaries = gc.apply_transform_pipeline(frame, cfg.pipeline_steps)
        if frame.empty:
            raise ValueError("Preparation removed all rows.")
        output = cfg.outdir / ("prepared.csv" if cfg.input_mode == "Annotated spreadsheet" else "prepared.parquet")
        gc.save_prepared_dataset(frame, output, cfg)
        tests = gc.prepare_external_test_files(cfg)
        print("\n".join(summaries))
        cfg = replace(cfg, data_path=output, separator=",", test_files=tests)
    original = sys.argv
    try:
        sys.argv = ["df-analyze.py", *gc.build_argv(cfg)]
        main()
    finally:
        sys.argv = original
    results = gc.find_latest_run(cfg.outdir)
    if results is None:
        raise RuntimeError(f"No performance table was produced. Inspect {cfg.outdir}")
    print(f"Results saved to {results}")
    return gc.load_results_table(results)


def context_code(cfg: gc.RunConfig | None) -> str:
    imports = (
        "from pathlib import Path\nimport numpy as np\nimport pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from df_analyze import gui_common as gc\n"
        "from df_analyze.gui_notebook import run_analysis\n"
    )
    if cfg is None:
        return imports + "\n# Load a dataset in the app, then click 'Insert app dataset and settings'.\n"
    return (imports + f"\ncfg = gc.config_from_json({gc.config_to_json(cfg)!r})\n"
            "raw_df = gc.load_full_dataset(cfg.data_path, cfg.separator, cfg.input_mode)\n"
            "df = raw_df.copy()\n"
            "if cfg.drop_duplicates:\n    df, removed = gc.drop_duplicate_rows(df)\n"
            "    print(f'Removed {removed:,} duplicate rows')\n"
            "df, preparation_log = gc.apply_transform_pipeline(df, cfg.pipeline_steps)\n"
            "for message in preparation_log:\n    print(message)\n"
            "print(f'Loaded {len(df):,} rows and {len(df.columns)} columns')\ndf.head()")


SNIPPETS = {
    "Data quality": (
        "# Run the app dataset setup cell first. df includes the preparation recipe.\n"
        "display(gc.dataset_profile(df))\n"
        "print(f'Rows: {len(df):,} | Columns: {len(df.columns):,}')\n"
        "print(f'Missing cells: {int(df.isna().sum().sum()):,}')\n"
        "try:\n    print(f'Duplicate rows: {int(df.duplicated().sum()):,}')\n"
        "except TypeError:\n    print('Duplicate detection is unavailable for nested values.')"
    ),
    "Target distribution": (
        "# Run the app dataset setup cell first.\n"
        "target = df[cfg.target]\n"
        "print(f'Target: {cfg.target} | Missing labels: {int(target.isna().sum()):,}')\n"
        "if cfg.mode == 'classify':\n"
        "    counts = target.value_counts(dropna=False)\n"
        "    display(counts.rename('count').to_frame().assign(percent=counts / len(target) * 100))\n"
        "    counts.head(30).plot.bar(title='Target distribution (up to 30 classes)')\n"
        "else:\n    display(target.describe())\n"
        "    pd.to_numeric(target, errors='coerce').dropna().hist(bins=30)\n"
        "    plt.title(f'Distribution of {cfg.target}')\n"
        "plt.tight_layout()\nplt.show()"
    ),
    "Saved results": (
        "# Read the latest completed run from the app's output folder.\n"
        "results_dir = gc.find_latest_run(cfg.outdir)\n"
        "if results_dir is None:\n    print('No completed results found. Run an experiment first.')\n"
        "else:\n    print(f'Results: {results_dir}')\n"
        "    scores = gc.load_results_table(results_dir)\n"
        "    display(gc.filtered_sorted_view(scores, cfg.htune_metric))"
    ),
}


def cell(source="", kind="code"):
    result = {"id": uuid4().hex[:8], "cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def starter(cfg: gc.RunConfig | None = None):
    return [
        cell("# The discovery journal\n\nA question, a few lines of Python, and room for a surprise.\n\n"
             "| 01 / Observe | 02 / Explore | 03 / Reflect |\n"
             "| :--- | :--- | :--- |\n"
             "| Load your data and look for patterns. | Test an idea with a table or a plot. | Record what changed your mind. |\n\n"
             "> **My research question:** What would I like to learn from this dataset?\n\n"
             "Start with the setup cell below. Variables stay available until you restart Python.", "markdown"),
        cell(context_code(cfg)),
        cell("# 01 / Meet the data\nif 'df' in globals():\n    display(df.describe(include='all'))\nelse:\n    print('Insert and run the app dataset setup cell first.')"),
        cell("# 02 / Follow a pattern\nif 'df' in globals():\n"
             "    numeric = df.select_dtypes(include='number')\n"
             "    if not numeric.empty:\n"
             "        column = numeric.columns[0]\n"
             "        with plt.rc_context({'axes.spines.top': False, 'axes.spines.right': False}):\n"
             "            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#faf9f5')\n"
             "            ax.set_facecolor('#faf9f5')\n"
             "            ax.hist(numeric[column].dropna(), bins=24, color='#459d89', edgecolor='#faf9f5', linewidth=1.5)\n"
             "            ax.set(title=f'A closer look at {column}', xlabel=str(column), ylabel='Observations')\n"
             "            ax.grid(axis='y', alpha=0.15)\n"
             "            ax.set_axisbelow(True)\n"
             "            fig.tight_layout()\n            plt.show()\n"
             "    else:\n        print('No numeric columns yet. Try the target distribution helper.')"),
        cell("## 03 / What did you notice?\n\n"
             "**Observation** — Describe a pattern, an outlier, or something unexpected.\n\n"
             "**Next question** — What would help you check whether that pattern holds?\n\n"
             "**Experiment notes** — Record the settings and results you want to revisit.", "markdown"),
        cell("# Run df-analyze with the app's settings, or override any RunConfig field.\n# scores = run_analysis(cfg, models=['dummy'], feat_select=['none'], htune_trials=1, no_preds=True)\n# scores"),
    ]


class NotebookSession:
    def __init__(self, root: Path, cfg: gc.RunConfig | None = None, python: str | None = None):
        self.root = Path(root).resolve()
        self.python = python or sys.executable
        self._bundled_kernel = python is None and getattr(sys, "frozen", False)
        self.working_dir = Path.home() / "df-analyze-notebooks" if self._bundled_kernel else self.root
        self.cells = starter(cfg)
        self.metadata = {"kernelspec": {"name": "python3", "display_name": "Python (df-analyze)", "language": "python"}}
        self._deleted = []
        self.lock = threading.RLock()
        self.manager = self.client = None
        self.thread = None
        self.busy = False
        self.closed = False
        self.stopping = False
        self.status = "Ready. Run a cell to start Python."
        self.prompt = None
        self.revision = 0
        self._displays = {}
        _SESSIONS.add(self)

    def snapshot(self):
        with self.lock:
            return {"cells": copy.deepcopy(self.cells), "busy": self.busy or self.stopping,
                    "status": self.status, "prompt": copy.deepcopy(self.prompt), "revision": self.revision,
                    "can_undo_delete": bool(self._deleted)}

    def edit(self, index, source, kind=None):
        with self.lock:
            if self.busy:
                raise ValueError("Wait for the cell to finish before editing.")
            target = self.cells[index]
            if kind and kind != target["cell_type"]:
                self.cells[index] = cell(source, kind)
            else:
                if target["source"] != source:
                    target["source"] = source
                    self.revision += 1

    def add(self, kind="code", source=""):
        with self.lock:
            if self.busy:
                raise ValueError("Wait for execution to finish before adding cells.")
            self.cells.append(cell(source, kind))
            self.revision += 1

    def remove(self, index):
        with self.lock:
            if self.busy:
                raise ValueError("Wait for execution to finish before removing cells.")
            self._deleted.append((index, self.cells.pop(index)))
            self._deleted = self._deleted[-20:]
            self.revision += 1

    def undo_delete(self):
        with self.lock:
            if self.busy or self.stopping:
                raise ValueError("Wait for execution to finish before restoring cells.")
            if not self._deleted:
                return None
            index, entry = self._deleted.pop()
            index = min(index, len(self.cells))
            self.cells.insert(index, entry)
            self.revision += 1
            return index

    def duplicate(self, index):
        with self.lock:
            if self.busy or self.stopping:
                raise ValueError("Wait for execution to finish before duplicating cells.")
            entry = copy.deepcopy(self.cells[index])
            entry["id"] = uuid4().hex[:8]
            if entry["cell_type"] == "code":
                entry.update(outputs=[], execution_count=None)
            self.cells.insert(index + 1, entry)
            self.revision += 1
            return index + 1

    def move(self, index, delta):
        with self.lock:
            if self.busy:
                raise ValueError("Wait for execution to finish before moving cells.")
            destination = index + delta
            if 0 <= destination < len(self.cells):
                self.cells[index], self.cells[destination] = self.cells[destination], self.cells[index]
                self.revision += 1

    def to_ipynb(self):
        import nbformat

        with self.lock:
            notebook = nbformat.from_dict({"nbformat": 4, "nbformat_minor": 5,
                "metadata": copy.deepcopy(self.metadata),
                "cells": copy.deepcopy(self.cells)})
        return nbformat.writes(notebook)

    def load_ipynb(self, text):
        import nbformat

        notebook = nbformat.reads(text, as_version=4)
        nbformat.validate(notebook)
        cells = json.loads(nbformat.writes(notebook))["cells"]
        for entry in cells:
            entry["id"] = uuid4().hex[:8]
            entry["source"] = "".join(entry["source"]) if isinstance(entry["source"], list) else entry["source"]
        with self.lock:
            if self.busy:
                raise ValueError("Interrupt execution before loading another notebook.")
            self.cells = cells
            self.metadata = copy.deepcopy(dict(notebook.metadata))
            self._deleted.clear()
            self._displays.clear()
            self.revision += 1
            self.status = "Notebook loaded. Cells have not been executed. Restart Python for a clean namespace."

    def _start(self):
        if self.manager is not None:
            return
        from jupyter_client.manager import KernelManager
        from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

        if not self.python:
            raise ValueError("A Python environment with ipykernel and df-analyze is required.")
        command = [self.python]
        command += ["--notebook-kernel"] if self._bundled_kernel else ["-m", "ipykernel_launcher"]
        command += ["-f", "{connection_file}"]

        class AppKernelSpecManager(KernelSpecManager):
            def get_kernel_spec(self, kernel_name):
                # Do not depend on a python3 kernelspec installed on the user's
                # computer. The app explicitly owns its kernel command.
                return KernelSpec(argv=command, display_name="Python (df-analyze)", language="python")

        manager = KernelManager(kernel_name="df-analyze", kernel_spec_manager=AppKernelSpecManager())
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(self.root), str(self.root / "src"), env.get("PYTHONPATH", "")])
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        self.manager = manager
        try:
            self.working_dir.mkdir(parents=True, exist_ok=True)
            manager.start_kernel(cwd=str(self.working_dir), env=env, **kwargs)
            self.client = manager.blocking_client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=45)
        except Exception:
            self._shutdown()
            raise

    def execute(self, index=None, *, scope="cell"):
        with self.lock:
            if self.busy or self.stopping or self.closed:
                raise ValueError("Python is already running or this notebook is closed.")
            if scope not in {"cell", "through", "from"}:
                raise ValueError("Choose cell, through, or from execution scope.")
            if index is not None and not 0 <= index < len(self.cells):
                raise ValueError("Select an existing cell to run.")
            if index is None:
                indices = list(range(len(self.cells)))
            elif scope == "through":
                indices = list(range(index + 1))
            elif scope == "from":
                indices = list(range(index, len(self.cells)))
            else:
                indices = [index]
            indices = [i for i in indices if self.cells[i]["cell_type"] == "code"]
            if not indices:
                self.status = "No Python cells selected. Add a code cell to run Python."
                return
            self.busy = True
            self.status = "Starting Python..." if self.manager is None else "Running..."
            self.thread = threading.Thread(target=self._execute, args=(indices,), daemon=True, name="notebook-execution")
            self.thread.start()

    def _execute(self, indices):
        try:
            self._start()
            client, manager = self.client, self.manager
            if client is None or manager is None:
                raise RuntimeError("Python stopped before execution could begin.")
            for index in indices:
                if self.closed:
                    break
                with self.lock:
                    target = self.cells[index]
                    target["outputs"] = []
                    self.status = f"Running cell {index + 1}..."
                    source = target["source"]
                    self.revision += 1
                msg_id = client.execute(source, allow_stdin=True, stop_on_error=True)
                failed, clear_next = False, False
                while not self.closed:
                    try:
                        request = client.get_stdin_msg(timeout=0)
                        if request.get("parent_header", {}).get("msg_id") == msg_id:
                            with self.lock:
                                self.prompt = request["content"]
                    except queue.Empty:
                        pass
                    try:
                        message = client.get_iopub_msg(timeout=.1)
                    except queue.Empty:
                        if not manager.is_alive():
                            raise RuntimeError("Python stopped unexpectedly. Restart the kernel to continue.")
                        continue
                    if message.get("parent_header", {}).get("msg_id") != msg_id:
                        continue
                    kind, content = message["msg_type"], message["content"]
                    with self.lock:
                        if kind == "status" and content["execution_state"] == "idle":
                            break
                        if kind == "execute_input":
                            target["execution_count"] = content["execution_count"]
                        elif kind == "clear_output":
                            clear_next = content.get("wait", False)
                            if not clear_next:
                                target["outputs"] = []
                        elif kind in ("stream", "display_data", "execute_result", "error", "update_display_data"):
                            if clear_next:
                                target["outputs"] = []
                                clear_next = False
                            output = {"output_type": "display_data" if kind == "update_display_data" else kind,
                                      **{key: value for key, value in content.items() if key != "transient"}}
                            display_id = content.get("transient", {}).get("display_id")
                            if kind == "update_display_data":
                                for previous in self._displays.get(display_id, []):
                                    previous.update(data=output["data"], metadata=output.get("metadata", {}))
                            elif kind == "stream" and target["outputs"] and target["outputs"][-1].get("output_type") == "stream" and target["outputs"][-1].get("name") == output.get("name"):
                                previous = target["outputs"][-1]
                                previous["text"] += output["text"]
                            else:
                                target["outputs"].append(output)
                                if display_id:
                                    self._displays.setdefault(display_id, []).append(output)
                            failed = failed or kind == "error"
                            self.revision += 1
                # Consume shell replies as well as IOPub output so long-lived
                # notebooks do not accumulate an unused reply queue.
                try:
                    while client.get_shell_msg(timeout=1).get("parent_header", {}).get("msg_id") != msg_id:
                        pass
                except queue.Empty:
                    pass
                if failed:
                    self.status = "Cell raised an error. Fix it and run again."
                    break
            else:
                self.status = "Execution complete."
        except Exception as error:
            self.status = "Python stopped. Variables cleared." if self.stopping else f"Notebook error: {error}"
        finally:
            with self.lock:
                self.busy = False
                self.prompt = None
                self.revision += 1

    def interrupt(self):
        if self.manager is not None:
            self.manager.interrupt_kernel()

    def stop(self):
        """Force-stop native calls that cannot respond promptly to interrupts."""
        if self.stopping:
            return
        self.stopping = True
        self.status = "Stopping Python and clearing variables..."

        def shutdown():
            try:
                self._shutdown()
                if self.thread is not None:
                    self.thread.join(5)
                self._displays.clear()
            finally:
                self.busy = False
                self.stopping = False
                self.prompt = None
                self.status = "Python stopped. Variables cleared; run setup cells again."
                self.revision += 1
        threading.Thread(target=shutdown, daemon=True, name="notebook-stop").start()

    def answer(self, text):
        with self.lock:
            if self.prompt is not None and self.client is not None:
                self.client.input(text)
                self.prompt = None

    def restart(self):
        if self.busy:
            raise ValueError("Interrupt the running cell before restarting Python.")
        self._shutdown()
        self._displays.clear()
        self.status = "Python restarted. Variables cleared; run setup cells again."
        self.revision += 1

    def clear_outputs(self):
        with self.lock:
            if self.busy:
                raise ValueError("Wait for execution to finish before clearing outputs.")
            for target in self.cells:
                if target["cell_type"] == "code":
                    target.update(outputs=[], execution_count=None)
            self._displays.clear()
            self.revision += 1

    def _shutdown(self):
        if self.manager is not None:
            import psutil

            children = []
            provisioner = self.manager.provisioner
            if provisioner is not None and getattr(provisioner, "pid", None):
                try:
                    children = psutil.Process(provisioner.pid).children(recursive=True)
                except psutil.Error:
                    pass
            for process in children:
                try:
                    process.terminate()
                except psutil.Error:
                    pass
            try:
                self.manager.shutdown_kernel(now=True)
            finally:
                if self.client is not None:
                    self.client.stop_channels()
                self.manager = self.client = None
                _, alive = psutil.wait_procs(children, timeout=1)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.Error:
                        pass

    def close(self):
        self.closed = True
        self._shutdown()
        _SESSIONS.discard(self)


@atexit.register
def _close_sessions():
    for session in list(_SESSIONS):
        try:
            session.close()
        except Exception:
            pass
