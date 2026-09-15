"""Embedding controls; computation runs outside the render thread."""
import queue
from pathlib import Path

import dearpygui.dearpygui as dpg

from desktop import dpg_context as ui
from desktop.log_stream import LineBuffer
from desktop.runner import start_embedding
from df_analyze.gui_embedding import ACTIONS, INPUT_HELP, EmbeddingConfig

_job = None
_log = LineBuffer()
_output = None


def running():
    return _job is not None


def cancel():
    if _job is not None:
        _job.cancel()


def config():
    return EmbeddingConfig(**{key: dpg.get_value("embedding_" + key) for key in EmbeddingConfig.__dataclass_fields__})


def poll():
    global _job
    if _job is None:
        return
    while True:
        try:
            _log.feed(_job.log_queue.get_nowait())
        except queue.Empty:
            break
    dpg.set_value("embedding_log", _log.render())
    try:
        status, message = _job.done_queue.get_nowait()
    except queue.Empty:
        return
    _job = None
    dpg.configure_item("embedding_controls", enabled=True)
    dpg.configure_item("embedding_cancel", enabled=False)
    dpg.set_value("embedding_status", message or "Completed")
    dpg.configure_item("embedding_load", enabled=status == "success" and _output is not None and _output.is_file())


def build(is_analysis_running, load_dataset):
    def selected(sender, data, destination):
        path = data.get("file_path_name", "")
        if path:
            dpg.set_value(destination, path)

    def start():
        global _job, _log, _output
        try:
            if _job is not None or is_analysis_running():
                raise ValueError("Wait for or cancel the active task first.")
            cfg = config()
            cfg.argv()
            _log = LineBuffer()
            _output = cfg.output_path() if cfg.action == ACTIONS[0] else None
            _job = start_embedding(cfg)
            dpg.configure_item("embedding_controls", enabled=False)
            dpg.configure_item("embedding_cancel", enabled=_job.can_cancel)
            dpg.configure_item("embedding_load", enabled=False)
            dpg.set_value("embedding_status", "Running; progress appears below.")
        except (ValueError, OSError) as error:
            dpg.set_value("embedding_status", str(error))

    def load():
        if _output is not None and _output.is_file():
            dpg.set_value("input_mode_combo", "Data table")
            dpg.set_value("delimiter_combo", "Comma")
            load_dataset(Path(_output))

    with ui.file_dialog(show=False, tag="embedding_file_dialog", callback=selected,
                        user_data="embedding_data_path", width=800, height=480):
        dpg.add_file_extension(".parquet")
    with ui.group(tag="page_embeddings", show=False):
        dpg.add_text("Text and image embeddings")
        dpg.add_text(INPUT_HELP, wrap=850)
        with ui.group(tag="embedding_controls"):
            dpg.add_combo(ACTIONS, tag="embedding_action", label="Action", default_value=ACTIONS[0], width=300)
            dpg.add_combo(["nlp", "vision"], tag="embedding_modality", label="Text / image modality", default_value="nlp", width=200)
            with ui.group(horizontal=True):
                dpg.add_input_text(tag="embedding_data_path", label="Input Parquet", width=500)
                dpg.add_button(label="Browse", callback=lambda: dpg.show_item("embedding_file_dialog"))
            dpg.add_input_text(tag="embedding_outpath", label="Output Parquet (blank = embedded.parquet)", width=500)
            dpg.add_input_text(tag="embedding_name", label="Dataset name (optional)", width=300)
            dpg.add_input_int(tag="embedding_batch_size", label="Batch size", default_value=2, min_value=1, min_clamped=True, width=200)
            dpg.add_input_int(tag="embedding_limit_samples", label="First N samples (0 = all)", default_value=0, min_value=0, min_clamped=True, width=200)
            dpg.add_button(label="Start embedding task", callback=start)
        dpg.add_button(label="Cancel embedding task", tag="embedding_cancel", enabled=False, callback=cancel)
        dpg.add_button(label="Load embedded dataset for analysis", tag="embedding_load", enabled=False, callback=load)
        dpg.add_text("Ready", tag="embedding_status", wrap=850)
        dpg.add_input_text(tag="embedding_log", multiline=True, readonly=True, height=380, width=-1)
