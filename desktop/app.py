from __future__ import annotations

import argparse
import contextlib
import math
import multiprocessing
import queue
import subprocess
import sys
import time
import webbrowser
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    for import_root in (ROOT, ROOT / "src"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd

# Agg is the non-interactive backend: no display server needed, just render
# to an in-memory buffer we hand to DPG as a texture. Charts are rendered
# with matplotlib/seaborn (already project dependencies) rather than DPG's
# native implot widgets so they look like standard data-science plots
# (anti-aliased, real box plots, colorbars) instead of basic bars/lines.
# sklearn itself is NOT imported here -- PCA's ~3.5s import cost is deferred
# to first use on the PCA tab, same "pay for it only if you use it"
# philosophy as the ML stack in runner.py.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from desktop import dpg_context as ui
from desktop import advanced_settings
from desktop import embedding_panel
from desktop.notebook_panel import NotebookPanel
from desktop.dpg_context import ItemId
from desktop.log_stream import LineBuffer
from desktop.runner import RunHandle, start_run
from desktop.workflow_canvas import WorkflowCanvas
from df_analyze import gui_common as gc
from df_analyze._constants import VERSION
from df_analyze.gui_workflow import import_workflow

# Only the worker imports the ML stack: dataset exploration starts quickly.
_current_run: RunHandle | None = None
_line_buffer = LineBuffer()
_data_path: Path | None = None
_notebook: NotebookPanel | None = None
_loaded_input_settings: tuple[str, str] | None = None
_preview = pd.DataFrame()
_profile = pd.DataFrame()
_roles: dict[str, str] = {}
_result_dirs: dict[str, Path] = {}
_result_dir: Path | None = None
_result_table = pd.DataFrame()
_artifacts: dict[str, Path] = {}
_run_outdir: Path | None = None
_run_started = 0.0
_workflow: WorkflowCanvas | None = None
_preview_limit = 2000
# A chainable, no-code data-manipulation pipeline (see gc.TransformStep) --
# _pipeline_preview is _preview run through _pipeline_steps, recomputed live
# on every add/remove so the Data preparation tab always shows real,
# up-to-date results, never a static simulation.
_pipeline_steps: list[gc.TransformStep] = []
_pipeline_preview = pd.DataFrame()
# Multi-target analysis: which extra columns (besides target_combo's value)
# are checked, persisted across the checklist's rebuilds the same way _roles
# persists across render_roles() rebuilds.
_extra_targets: set[str] = set()
_per_target_table: pd.DataFrame | None = None

# Lightweight, always-on per-frame animation: count-up KPI numbers and a
# pulsing "analysis running" indicator. Pure Python state ticked once per
# frame from the main render loop (see _tick_animations/run_gui) -- no
# threads, no DPG animation API (DPG doesn't have one), just interpolating
# a value over wall-clock time and writing it back with dpg.set_value().
_animations: dict[str, dict[str, Any]] = {}
_kpi_raw: dict[str, float] = {}

MODEL_LABELS = {
    "dummy": "Baseline control", "knn": "Nearest neighbors", "lr": "Logistic regression",
    "elastic": "Elastic net", "lgbm": "LightGBM", "catboost": "CatBoost",
    "rf": "Random forest", "sgd": "Linear SGD", "mlp": "Neural network (MLP)",
    "gandalf": "GANDALF neural network",
}
ROLE_OPTIONS = ["Automatic", "Categorical", "Ordinal", "Exclude"]


def _text(tag: str) -> str:
    value = dpg.get_value(tag)
    return "" if value is None else str(value)


def _flag(tag: str) -> bool:
    return bool(dpg.get_value(tag))


def _integer(tag: str) -> int:
    return int(dpg.get_value(tag))


def _number(tag: str) -> float:
    return float(dpg.get_value(tag))


def _set_status(message: str, color: tuple[int, int, int]) -> None:
    dpg.set_value("status_text", message)
    dpg.configure_item("status_text", color=color)


def _animate_number(tag: str, target: float, formatter: Callable[[float], str], duration: float = 0.6) -> None:
    """Count a KPI/score display up (or down) from its last value to `target`
    over `duration` seconds, ticked by _tick_animations() every frame."""
    start = _kpi_raw.get(tag, 0.0)
    _kpi_raw[tag] = target
    if not dpg.does_item_exist(tag):
        return
    if abs(target - start) < 1e-12:
        dpg.set_value(tag, formatter(target))
        return
    _animations[tag] = {"from": start, "to": target, "t0": time.monotonic(), "dur": duration, "fmt": formatter}


def _tick_animations() -> None:
    """Advance every in-flight number animation by one frame. Safe to call
    every frame unconditionally, including headless (no viewport needed)."""
    if not _animations:
        return
    now = time.monotonic()
    finished = []
    for tag, anim in _animations.items():
        if not dpg.does_item_exist(tag):
            finished.append(tag)
            continue
        elapsed = now - anim["t0"]
        frac = 1.0 if anim["dur"] <= 0 else min(1.0, elapsed / anim["dur"])
        eased = 1 - (1 - frac) ** 3  # ease-out cubic -- fast start, gentle settle
        value = anim["from"] + (anim["to"] - anim["from"]) * eased
        dpg.set_value(tag, anim["fmt"](value))
        if frac >= 1.0:
            finished.append(tag)
    for tag in finished:
        _animations.pop(tag, None)


def _update_status_pulse() -> None:
    """A small breathing dot beside the sidebar status line while an
    analysis is running -- a lightweight "something is alive" signal
    alongside the existing spinner and elapsed-time counter. Safe to call
    every frame unconditionally, including headless (no viewport needed)."""
    if not dpg.does_item_exist("status_pulse_dot"):
        return
    if _current_run is not None:
        alpha = int(140 + 100 * math.sin(time.monotonic() * 4.0))
        alpha = max(0, min(255, alpha))
        dpg.configure_item("status_pulse_dot", color=(0, 0, 0, 0), fill=(*ACCENT_LIGHT, alpha))
    else:
        dpg.configure_item("status_pulse_dot", color=(0, 0, 0, 0), fill=(0, 0, 0, 0))


def on_nav_selected(sender: ItemId = 0, app_data: Any = None, user_data: str = "overview") -> None:
    for key, _ in NAV_ITEMS:
        dpg.set_value(f"nav_{key}", key == user_data)
        dpg.configure_item(f"page_{key}", show=key == user_data)
    if user_data == "workflow" and _workflow is not None:
        _workflow.refresh_labels()


def on_show_dialog(sender: ItemId, app_data: Any, user_data: str) -> None:
    dpg.show_item(user_data)


def _selected_models(prefix: str, names: list[str]) -> list[str]:
    return [name for name in names if _flag(f"{prefix}_{name}")]


def _active_feat_select() -> list[str]:
    selected = [name for name in gc.FEAT_SELECT_METHODS if _flag(f"feat_{name}")]
    return [name for name in selected if name != "none"] or ["none"]


def _form_run_config() -> gc.RunConfig:
    if _data_path is None:
        raise ValueError("Load a dataset in Data explorer first.")
    separator = _text("delimiter_custom") if _text("delimiter_combo") == "Custom" else gc.DELIMITERS.get(_text("delimiter_combo"), ",")
    if _loaded_input_settings != (_text("input_mode_combo"), separator):
        raise ValueError("Input format or delimiter changed. Reload the dataset to refresh its columns.")
    mode = _text("mode_radio")
    return gc.RunConfig(
        data_path=_data_path, target=_text("target_combo"), mode=mode,
        models=_selected_models("clf", gc.CLASSIFIERS) if mode == "classify" else _selected_models("reg", gc.REGRESSORS),
        feat_select=_active_feat_select(), embed_model=_text("embed_model_combo"),
        wrapper_method=_text("wrapper_method_combo"), wrapper_model=_text("wrapper_model_combo"),
        n_feat_wrapper=_text("n_feat_wrapper_input"),
        redundant_wrapper_selection=_flag("redundant_wrapper_checkbox"),
        redundant_threshold=_number("redundant_threshold_input"),
        redundant_corr_threshold=_number("redundant_corr_threshold_input"),
        filter_method=_text("filter_method_combo"), n_feat_filter=_text("n_feat_filter_input"),
        filter_assoc_cont_classify=_text("filter_assoc_cont_combo"),
        filter_assoc_cat_classify=_text("filter_assoc_cat_combo"),
        filter_pred_classify=_text("filter_pred_combo"),
        filter_assoc_cont_regress=_text("filter_assoc_cont_combo"),
        filter_assoc_cat_regress=_text("filter_assoc_cat_combo"),
        filter_pred_regress=_text("filter_pred_combo"),
        norm=_text("norm_combo"), nan=_text("nan_combo"), htune_trials=_integer("htune_trials_input"),
        htune_metric=_text("metric_combo"), test_val_size=_number("test_val_size_slider"),
        seed=_integer("seed_input"), outdir=Path(_text("outdir_input")).expanduser(),
        extra_args=_text("extra_args_input"),
        categoricals=[col for col, role in _roles.items() if role == "Categorical"],
        ordinals=[col for col, role in _roles.items() if role == "Ordinal"],
        drops=[col for col, role in _roles.items() if role == "Exclude"],
        grouper=_text("grouper_combo") or None,
        no_preds=_flag("no_preds_checkbox"), adaptive_error=_flag("adaptive_error_checkbox"),
        drop_duplicates=_flag("drop_duplicates_checkbox"),
        pipeline_steps=list(_pipeline_steps),
        extra_targets=sorted(_extra_targets) if _flag("multi_target_checkbox") else [],
        mt_agg_strategy=_text("mt_agg_strategy_combo"),
        mt_top_k=_text("mt_top_k_input"),
        input_mode=_text("input_mode_combo"),
        separator=_text("delimiter_custom") if _text("delimiter_combo") == "Custom" else gc.DELIMITERS.get(_text("delimiter_combo"), ","),
        test_files=[line.strip() for line in _text("test_files_input").splitlines() if line.strip()],
        tests_method=_text("tests_method_combo"),
        advanced_options=advanced_settings.read(),
    )


def _build_run_config() -> gc.RunConfig:
    cfg = _form_run_config()
    return _workflow.flow.compile(cfg) if _workflow is not None and _workflow.active else cfg


def _canvas_config() -> gc.RunConfig:
    if _data_path is not None:
        return _form_run_config()
    mode = _text("mode_radio") or "classify"
    names = gc.CLASSIFIERS if mode == "classify" else gc.REGRESSORS
    return gc.RunConfig(data_path=Path("Select a dataset"), target="target", mode=mode,
                        models=_selected_models("clf" if mode == "classify" else "reg", names),
                        feat_select=_active_feat_select(), htune_trials=_integer("htune_trials_input"),
                        htune_metric=_text("metric_combo"))


def _table(parent: str, frame: pd.DataFrame, limit: int = 100) -> None:
    dpg.delete_item(parent, children_only=True)
    with ui.table(parent=parent, header_row=True, row_background=True,
                  borders_innerH=True, borders_outerH=True, scrollX=True,
                  policy=dpg.mvTable_SizingFixedFit):
        for col in frame.columns:
            dpg.add_table_column(label=str(col), width_fixed=True, init_width_or_weight=150)
        for values in frame.head(limit).itertuples(index=False, name=None):
            with ui.table_row():
                for value in values:
                    cell = str(value)
                    dpg.add_text(cell[:110] + ("..." if len(cell) > 110 else ""))


def _load_data(path: Path) -> bool:
    global _data_path, _preview, _profile, _roles, _pipeline_steps, _extra_targets
    global _loaded_input_settings
    try:
        delimiter = _text("delimiter_custom") if _text("delimiter_combo") == "Custom" else gc.DELIMITERS.get(_text("delimiter_combo"), ",")
        frame = gc.load_preview(path, max_rows=_preview_limit, separator=delimiter,
                                input_mode=_text("input_mode_combo") or "Data table")
        if frame.empty or not len(frame.columns):
            raise ValueError("The dataset contains no rows or columns.")
        profile = gc.dataset_profile(frame)
    except Exception as exc:
        _set_status(f"Cannot load data: {exc}", ERROR)
        return False
    _data_path, _preview, _profile = path, frame, profile
    _loaded_input_settings = (_text("input_mode_combo"), delimiter)
    _roles = {}
    # A preparation pipeline is built against one dataset's columns -- like
    # roles, it doesn't carry over to a newly loaded/different dataset.
    _pipeline_steps = []
    _extra_targets = set()
    columns = [str(col) for col in frame.columns]
    dpg.set_value("data_path_display", str(path))
    dpg.configure_item("target_combo", items=columns)
    preferred = next((col for col in columns if col.lower() in {"target", "diagnosis", "label", "outcome"}), columns[-1])
    dpg.set_value("target_combo", preferred)
    dpg.configure_item("grouper_combo", items=["", *columns])
    dpg.set_value("grouper_combo", "")
    dpg.configure_item("distribution_combo", items=columns)
    dpg.set_value("distribution_combo", preferred)
    dpg.configure_item("role_column_combo", items=columns)
    dpg.set_value("role_column_combo", columns[0])
    numeric = [str(col) for col in frame.select_dtypes(include="number").columns]
    for tag in ("scatter_x_combo", "scatter_y_combo"):
        dpg.configure_item(tag, items=numeric)
    dpg.set_value("scatter_x_combo", numeric[0] if numeric else "")
    dpg.set_value("scatter_y_combo", numeric[1] if len(numeric) > 1 else numeric[0] if numeric else "")

    categorical_like = _categorical_like_columns()
    for tag in ("hist_combo", "viz_relationship_x_combo", "viz_relationship_y_combo", "box_value_combo", "trend_combo"):
        dpg.configure_item(tag, items=numeric)
    dpg.set_value("hist_combo", numeric[0] if numeric else "")
    dpg.set_value("viz_relationship_x_combo", numeric[0] if numeric else "")
    dpg.set_value("viz_relationship_y_combo", numeric[1] if len(numeric) > 1 else numeric[0] if numeric else "")
    dpg.set_value("box_value_combo", numeric[0] if numeric else "")
    dpg.set_value("trend_combo", numeric[0] if numeric else "")
    for tag in ("cat_combo", "box_group_combo", "pca_color_combo"):
        dpg.configure_item(tag, items=categorical_like)
    dpg.set_value("cat_combo", categorical_like[0] if categorical_like else "")
    dpg.set_value("box_group_combo", categorical_like[0] if categorical_like else "")
    # Default PCA coloring to the chosen target when it's categorical-like
    # (the common case, and what the reference dashboard mockup showed),
    # falling back to the first available categorical-like column.
    pca_default = preferred if preferred in categorical_like else (categorical_like[0] if categorical_like else "")
    dpg.set_value("pca_color_combo", pca_default)
    missing = float(frame.isna().mean().mean() * 100)
    dpg.set_value("dataset_summary", f"{path.name}  |  {len(frame):,} preview rows  |  {len(columns)} columns")
    dpg.set_value("preview_note", f"Exploration uses up to {_preview_limit:,} rows; analysis uses the full dataset. Table shows up to 100 rows and 30 columns.")
    messages = []
    if missing:
        messages.append(f"Missing cells: {missing:.1f}% of the preview. Review heavily incomplete columns.")
    constants = profile.loc[profile["unique"] <= 1, "column"].astype(str).tolist()
    if constants:
        messages.append("Constant or empty columns: " + ", ".join(constants[:12]))
    duplicates = int(frame.duplicated().sum())
    messages.append(f"Duplicate preview rows: {duplicates:,}. Check whether these are repeated observations.")
    messages.append("Review IDs, post-outcome measurements, and target leakage before training.")
    dpg.set_value("quality_notes", "\n".join(messages))
    on_preview_filter()
    _table("profile_container", profile, limit=150)
    render_roles()
    render_multi_target_checklist()
    _refresh_pipeline_preview()
    on_distribution_changed()
    on_scatter_changed()
    _render_missing_chart()
    on_histogram_changed()
    on_categorical_changed()
    on_viz_relationship_changed()
    on_box_changed()
    on_trend_changed()
    _render_heatmap()
    on_pca_changed()
    _set_status("Dataset ready", SUCCESS)
    on_config_changed()
    # Registered last, right before control returns to the render loop --
    # everything above (chart rendering, PCA's one-time sklearn import) is
    # exactly the kind of slow, blocking work that would otherwise eat the
    # count-up's short window before a single frame gets a chance to tick
    # it, making the animation invisible for anything but a trivial load.
    _animate_number("kpi_rows", float(len(frame)), lambda v: f"{int(round(v)):,}")
    _animate_number("kpi_columns", float(len(columns)), lambda v: str(int(round(v))))
    _animate_number("kpi_missing", missing, lambda v: f"{v:.1f}%")
    _animate_number("kpi_numeric", float(len(numeric)), lambda v: str(int(round(v))))
    return True


def on_preview_limit_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    """Reload the current dataset at a new preview-row cap. Exploration
    only ever looks at this bounded preview; the analysis itself always
    reads the complete file regardless of this setting."""
    global _preview_limit
    _preview_limit = _integer("preview_limit_input")
    if _data_path is not None:
        _load_data(_data_path)


def on_data_file_selected(sender: ItemId, app_data: Any) -> None:
    value = app_data.get("file_path_name", "") if isinstance(app_data, dict) else ""
    if value:
        _load_data(Path(str(value)))


def on_load_path(sender: ItemId = 0, app_data: Any = None) -> None:
    value = _text("data_path_display").strip()
    if value:
        _load_data(Path(value).expanduser())


def on_test_file_selected(sender: ItemId, app_data: Any) -> None:
    if not isinstance(app_data, dict):
        return
    paths = list(app_data.get("selections", {}).values())
    if not paths and app_data.get("file_path_name"):
        paths = [app_data["file_path_name"]]
    existing = [line for line in _text("test_files_input").splitlines() if line]
    dpg.set_value("test_files_input", "\n".join(dict.fromkeys([*existing, *map(str, paths)])))
    on_config_changed()


def on_preview_filter(sender: ItemId = 0, app_data: Any = None) -> None:
    frame = _preview.iloc[:, :30]
    term = _text("preview_search").strip()
    if term and not frame.empty:
        mask = frame.astype(str).apply(lambda col: col.str.contains(term, case=False, regex=False)).any(axis=1)
        frame = frame.loc[mask]
    _table("preview_container", frame)
    dpg.set_value("preview_count", f"{len(frame):,} matching preview rows")


def _bar_chart(prefix: str, labels: list[str], values: list[float]) -> None:
    dpg.delete_item(f"{prefix}_y", children_only=True)
    dpg.reset_axis_ticks(f"{prefix}_x")
    if values:
        positions = [float(i) for i in range(len(values))]
        dpg.add_bar_series(positions, values, weight=0.65, parent=f"{prefix}_y")
        dpg.set_axis_ticks(f"{prefix}_x", tuple((label[:24], pos) for label, pos in zip(labels, positions)))
        dpg.fit_axis_data(f"{prefix}_x")
        dpg.fit_axis_data(f"{prefix}_y")


def _render_missing_chart() -> None:
    if _profile.empty:
        return
    subset = _profile.sort_values("missing_pct", ascending=False).head(12)
    _bar_chart("missing", subset["column"].astype(str).tolist(), subset["missing_pct"].astype(float).tolist())


def on_distribution_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    column = _text("distribution_combo")
    if column not in _preview:
        return
    series = _preview[column].dropna()
    dpg.delete_item("distribution_y", children_only=True)
    dpg.reset_axis_ticks("distribution_x")
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            counts, edges = np.histogram(values, bins=min(25, max(5, int(np.sqrt(len(values))))))
            dpg.add_bar_series(((edges[:-1] + edges[1:]) / 2).tolist(), counts.astype(float).tolist(),
                               weight=float(edges[1] - edges[0]) * 0.9, parent="distribution_y")
            dpg.fit_axis_data("distribution_x")
            dpg.fit_axis_data("distribution_y")
    else:
        counts = series.astype(str).value_counts().head(15)
        _bar_chart("distribution", counts.index.astype(str).tolist(), counts.astype(float).tolist())
    dpg.set_value("distribution_note", f"{column}: {len(series):,} non-missing values, {series.nunique():,} distinct values in preview. Up to 15 categories shown.")


def on_scatter_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    x_col, y_col = _text("scatter_x_combo"), _text("scatter_y_combo")
    dpg.delete_item("scatter_y", children_only=True)
    if x_col not in _preview or y_col not in _preview:
        return
    x = pd.to_numeric(_preview[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(_preview[y_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    dpg.add_scatter_series(x[keep].tolist(), y[keep].tolist(), parent="scatter_y")
    dpg.configure_item("scatter_x", label=x_col)
    dpg.configure_item("scatter_y", label=y_col)
    dpg.fit_axis_data("scatter_x")
    dpg.fit_axis_data("scatter_y")


# ---------------------------------------------------------------------------
# Data visualization page: several chart forms over the same bounded preview
# used elsewhere (histogram, categorical bar/pie, numeric relationships,
# median/IQR "box plot" by group, correlation heatmap, row-order trend).
# ---------------------------------------------------------------------------


def _numeric_columns() -> list[str]:
    return [str(c) for c in _preview.select_dtypes(include="number").columns]


def _categorical_like_columns() -> list[str]:
    numeric_high_card = {
        col for col in _numeric_columns() if _preview[col].nunique(dropna=True) > 15
    }
    return [str(c) for c in _preview.columns if c not in numeric_high_card]


def _active_columns() -> list[str]:
    """The columns a run will actually see: once any preparation steps are
    added, that's _pipeline_preview's columns (which may have been renamed,
    dropped, or added to) rather than the raw file's -- target/grouper/role
    selection and validation must track this, not the original preview,
    once a step could have renamed or removed the column they point at."""
    return list(_pipeline_preview.columns) if _pipeline_steps else list(_preview.columns)


def on_histogram_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    column = _text("hist_combo")
    if column not in _preview:
        _chart_placeholder("histogram", "Load a dataset to see a distribution.")
        return
    values = pd.to_numeric(_preview[column], errors="coerce").dropna()
    if values.empty:
        _chart_placeholder("histogram", f"No numeric values in '{column}'.")
        return
    fig, ax = _new_figure("histogram")
    sns.histplot(values, bins=30, kde=len(values) > 1 and values.nunique() > 1,
                color=CATEGORICAL_PALETTE[0], edgecolor="none", ax=ax)
    for line in ax.lines:  # the KDE curve
        line.set_color(_rgb01(ACCENT_LIGHT))
        line.set_linewidth(1.6)
    ax.set_title(f"Distribution of {column}")
    ax.set_ylabel("Count")
    _style_axes(ax)
    _publish_figure(fig, "histogram")


def on_categorical_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    column = _text("cat_combo")
    if column not in _preview or _preview.empty:
        _chart_placeholder("catbar", "Load a dataset to see a breakdown.")
        _chart_placeholder("catpie", "")
        return
    counts = _preview[column].astype(str).value_counts().head(10)
    if counts.empty:
        _chart_placeholder("catbar", f"No values in '{column}'.")
        _chart_placeholder("catpie", "")
        return
    colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(counts))]

    fig, ax = _new_figure("catbar")
    ax.bar(range(len(counts)), counts.to_numpy(dtype=float), color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(i)[:12] for i in counts.index], rotation=30, ha="right")
    ax.set_title(f"{column} -- value counts")
    ax.set_ylabel("Count")
    _style_axes(ax)
    _publish_figure(fig, "catbar")

    fig, ax = _new_figure("catpie")
    ax.pie(counts.to_numpy(dtype=float), labels=[str(i)[:14] for i in counts.index],
          colors=colors, autopct="%1.0f%%", textprops={"color": _rgb01(TEXT_PRIMARY), "fontsize": 8},
          wedgeprops={"edgecolor": _rgb01(BG_CARD), "linewidth": 1.2})
    ax.set_title("Share of total")
    ax.title.set_color(_rgb01(TEXT_PRIMARY))
    ax.axis("equal")
    _publish_figure(fig, "catpie")

    dpg.set_value("categorical_note", f"{column}: top {len(counts)} of {_preview[column].nunique():,} distinct values in preview.")


def on_viz_relationship_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    x_col, y_col = _text("viz_relationship_x_combo"), _text("viz_relationship_y_combo")
    if x_col not in _preview or y_col not in _preview:
        _chart_placeholder("relationship", "Choose two numeric columns to see their relationship.")
        return
    x = pd.to_numeric(_preview[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(_preview[y_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.any():
        _chart_placeholder("relationship", "No finite numeric pairs to plot.")
        return
    fig, ax = _new_figure("relationship")
    ax.scatter(x[keep], y[keep], s=16, alpha=0.55, color=CATEGORICAL_PALETTE[0], edgecolors="none")
    ax.set_title(f"{x_col} vs {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    _style_axes(ax)
    _publish_figure(fig, "relationship")


def on_box_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    value_col, group_col = _text("box_value_combo"), _text("box_group_combo")
    if value_col not in _preview or group_col not in _preview or value_col == group_col:
        _chart_placeholder("box", "Choose a numeric feature and a group to compare.")
        return
    frame = _preview[[value_col, group_col]].copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        _chart_placeholder("box", f"No numeric values in '{value_col}' for these groups.")
        return
    top_groups = frame[group_col].astype(str).value_counts().head(12).index
    frame = frame.loc[frame[group_col].astype(str).isin(top_groups)]
    fig, ax = _new_figure("box")
    sns.boxplot(data=frame, x=group_col, y=value_col, ax=ax,
               color=CATEGORICAL_PALETTE[0], fliersize=2,
               boxprops={"alpha": 0.85}, medianprops={"color": _rgb01(ACCENT_LIGHT), "linewidth": 1.6})
    ax.set_title(f"{value_col} by {group_col}")
    labels = [label.get_text()[:14] for label in ax.get_xticklabels()]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    _style_axes(ax)
    _publish_figure(fig, "box")


def on_trend_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    column = _text("trend_combo")
    if column not in _preview:
        _chart_placeholder("trend", "Choose a numeric feature to see it across preview row order.")
        return
    values = pd.to_numeric(_preview[column], errors="coerce")
    if values.notna().sum() == 0:
        _chart_placeholder("trend", f"No numeric values in '{column}'.")
        return
    fig, ax = _new_figure("trend")
    ax.plot(range(len(values)), values.to_numpy(dtype=float), color=CATEGORICAL_PALETTE[0], linewidth=1.1)
    ax.set_title(f"{column} across preview row order")
    ax.set_xlabel("Row order (file order, not necessarily time)")
    ax.set_ylabel(column)
    _style_axes(ax)
    _publish_figure(fig, "trend")


def _render_heatmap() -> None:
    numeric_cols = _numeric_columns()[:20]
    if len(numeric_cols) < 2:
        dpg.set_value("heatmap_note", "Need at least two numeric columns for a correlation heatmap.")
        _chart_placeholder("heatmap", "Need at least two numeric columns for a correlation heatmap.")
        return
    corr = _preview[numeric_cols].corr().fillna(0.0)
    fig, ax = _new_figure("heatmap")
    annotate = len(numeric_cols) <= 10
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
               annot=annotate, fmt=".1f", annot_kws={"fontsize": 7, "color": _rgb01(TEXT_PRIMARY)},
               linewidths=0.6, linecolor=_rgb01(BG_CARD),
               cbar_kws={"shrink": 0.85}, square=False)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(colors=_rgb01(TEXT_MUTED), labelsize=7)
    colorbar.outline.set_edgecolor(_rgb01(BORDER))
    ax.set_xticklabels([label.get_text()[:12] for label in ax.get_xticklabels()], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([label.get_text()[:12] for label in ax.get_yticklabels()], rotation=0, fontsize=7)
    ax.set_title("Correlation heatmap")
    _style_axes(ax, grid=False)
    _publish_figure(fig, "heatmap")
    suffix = "" if len(_numeric_columns()) <= 20 else f" (first 20 of {len(_numeric_columns())})"
    dpg.set_value("heatmap_note", f"Pearson correlation across numeric preview columns{suffix}. Red = positive, blue = negative.")


def on_pca_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    color_col = _text("pca_color_combo")
    numeric_cols = _numeric_columns()
    if len(numeric_cols) < 2:
        _chart_placeholder("pca", "Need at least two numeric columns for a PCA projection.")
        return
    frame = _preview[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all", axis=1).dropna()
    if frame.shape[0] < 3 or frame.shape[1] < 2:
        _chart_placeholder("pca", "Not enough complete numeric rows for a PCA projection.")
        return
    # sklearn's import cost (~3.5s) is paid only when this tab is actually
    # used, same lazy-import philosophy as the ML stack in runner.py.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    standardized = StandardScaler().fit_transform(frame.to_numpy(dtype=float))
    projection = PCA(n_components=2, random_state=0).fit(standardized)
    coords = projection.transform(standardized)
    explained = projection.explained_variance_ratio_ * 100

    fig, ax = _new_figure("pca")
    labels = _preview.loc[frame.index, color_col].astype(str) if color_col in _preview else None
    if labels is not None and labels.nunique() <= 8:
        for i, group in enumerate(labels.unique()):
            mask = (labels == group).to_numpy()
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.65,
                      color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
                      edgecolors="none", label=group[:16])
        legend = ax.legend(fontsize=7, loc="best", framealpha=0.15, labelcolor=_rgb01(TEXT_PRIMARY))
        legend.get_frame().set_edgecolor(_rgb01(BORDER))
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.6, color=CATEGORICAL_PALETTE[0], edgecolors="none")
    ax.set_title("PCA projection (2D)")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    _style_axes(ax)
    _publish_figure(fig, "pca")
    dpg.set_value(
        "pca_note",
        f"scikit-learn PCA on {frame.shape[1]} standardized numeric columns, {frame.shape[0]:,} complete preview rows. "
        f"First two components explain {explained.sum():.1f}% of variance.",
    )


def on_target_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    target = _text("target_combo")
    _roles.pop(target, None)
    _extra_targets.discard(target)
    if _text("grouper_combo") == target:
        dpg.set_value("grouper_combo", "")
    render_roles()
    render_multi_target_checklist()
    on_config_changed()


def on_role_column_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    dpg.set_value("role_value_combo", _roles.get(_text("role_column_combo"), "Automatic"))


def on_apply_role(sender: ItemId = 0, app_data: Any = None) -> None:
    column = _text("role_column_combo")
    if column == _text("target_combo") or column == _text("grouper_combo"):
        _set_status("Target and group columns have dedicated controls.", WARNING)
        return
    if column:
        role = _text("role_value_combo")
        if role == "Automatic":
            _roles.pop(column, None)
        else:
            _roles[column] = role
    render_roles()
    on_config_changed()


def render_roles() -> None:
    target, grouper = _text("target_combo"), _text("grouper_combo")
    if _preview.empty:
        return
    rows = [{"Column": str(col), "Role": "Target" if col == target else "Group" if col == grouper else _roles.get(str(col), "Automatic"), "Detected dtype": str(_preview[col].dtype)} for col in _preview.columns]
    _table("roles_container", pd.DataFrame(rows), limit=150)


def on_group_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    _roles.pop(_text("grouper_combo"), None)
    _extra_targets.discard(_text("grouper_combo"))
    render_roles()
    render_multi_target_checklist()
    on_config_changed()


def render_multi_target_checklist() -> None:
    """A dynamic, rebuilt-on-change checklist of extra target columns -- DPG
    has no native multiselect, so this mirrors the model-selection checkbox
    pattern, except the column list is dataset-dependent and so has to be
    (re)built at runtime rather than fixed at UI-construction time, the same
    way render_roles() rebuilds its table."""
    if not dpg.does_item_exist("multi_target_container"):
        return
    dpg.delete_item("multi_target_container", children_only=True)
    if _preview.empty:
        dpg.add_text("Load a dataset to choose extra targets.", color=TEXT_MUTED, parent="multi_target_container")
        return
    target, grouper = _text("target_combo"), _text("grouper_combo")
    candidates = [c for c in _active_columns() if c not in (target, grouper)]
    if not candidates:
        dpg.add_text("No other columns available to use as extra targets.", color=TEXT_MUTED, parent="multi_target_container")
        return
    for column in candidates:
        dpg.add_checkbox(label=column, tag=f"mt_target_{column}", default_value=column in _extra_targets,
                         callback=on_extra_target_toggled, user_data=column, parent="multi_target_container")


def on_extra_target_toggled(sender: ItemId, app_data: bool, user_data: str) -> None:
    if app_data:
        _extra_targets.add(user_data)
    else:
        _extra_targets.discard(user_data)
    on_config_changed()


def on_multi_target_toggle_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    dpg.configure_item("multi_target_group", show=_flag("multi_target_checkbox"))
    on_config_changed()


# ---------------------------------------------------------------------------
# Data preparation pipeline (Preprocessing workspace > "Data preparation"
# tab): a fixed, no-code catalog of real pandas/numpy operations (see
# gc.TRANSFORM_OPS) built up one step at a time. Every step actually runs
# against the current preview immediately, so a user can never even select a
# column a prior step removed or renamed -- there is no separate schema
# simulator that could drift from what the pipeline really does.
# ---------------------------------------------------------------------------

_PREP_LABEL_TO_KEY = {spec.label: key for key, spec in gc.TRANSFORM_OPS.items()}

# Combo tags whose items must always reflect the CURRENT pipeline preview's
# real columns -- refreshed on every add/remove/clear.
_PREP_COLUMN_COMBOS = [
    "prep_rename_column_combo", "prep_cast_column_combo", "prep_drop_column_combo",
    "prep_text_column_combo", "prep_dt_column_combo", "prep_filter_column_combo",
    "prep_fill_column_combo", "prep_flag_column_combo", "prep_clip_column_combo",
    "prep_threshold_column_combo", "prep_combine_a_combo", "prep_combine_b_combo",
]


def _refresh_pipeline_preview() -> None:
    """Re-run _pipeline_steps against a copy of _preview (bounded, in-memory,
    never touches disk) so every dependent widget -- the pipeline history,
    the sample table, every column-picker combo, and target/grouper/role
    selection -- always reflects the pipeline's REAL current output."""
    global _pipeline_preview
    summaries: list[str] = []
    if not _pipeline_steps:
        _pipeline_preview = _preview.copy()
    else:
        try:
            _pipeline_preview, summaries = gc.apply_transform_pipeline(_preview, _pipeline_steps)
        except ValueError as exc:
            dpg.set_value("prep_error_text", str(exc))
            _pipeline_preview = _preview.copy()
    if not dpg.does_item_exist("pipeline_table_container"):
        return  # UI not built yet
    render_pipeline_table(summaries)
    _table("prep_sample_container", _pipeline_preview.head(20))
    dpg.set_value(
        "prep_summary_text",
        f"{len(_pipeline_steps)} step(s) applied -> {len(_pipeline_preview):,} rows, "
        f"{_pipeline_preview.shape[1]} columns (was {len(_preview):,} rows, {_preview.shape[1]} columns).",
    )
    columns = list(_pipeline_preview.columns)
    for tag in _PREP_COLUMN_COMBOS:
        dpg.configure_item(tag, items=columns)
    dpg.configure_item("prep_dropna_column_combo", items=["(any column)", *columns])
    # Target/grouper/role selection must keep pointing at real columns once a
    # step could have renamed or removed the one currently chosen.
    active = _active_columns()
    dpg.configure_item("role_column_combo", items=active)
    dpg.configure_item("target_combo", items=active)
    dpg.configure_item("grouper_combo", items=["", *active])
    if active and _text("target_combo") not in active:
        dpg.set_value("target_combo", active[0])
        _set_status("A preparation step changed your target column. A new target was chosen -- please review it.", WARNING)
    if _text("grouper_combo") not in ("", *active):
        dpg.set_value("grouper_combo", "")
    if active and _text("role_column_combo") not in active:
        dpg.set_value("role_column_combo", active[0])
    # A step may have renamed/removed a column that was checked as an extra
    # target -- drop anything no longer real before the checklist rebuilds.
    _extra_targets.intersection_update(active)
    render_roles()
    render_multi_target_checklist()


def render_pipeline_table(summaries: list[str]) -> None:
    dpg.delete_item("pipeline_table_container", children_only=True)
    if not _pipeline_steps:
        dpg.add_text("No preparation steps added yet.", color=TEXT_MUTED, parent="pipeline_table_container")
        return
    with ui.table(parent="pipeline_table_container", header_row=True, row_background=True,
                  borders_innerH=True, borders_outerH=True, policy=dpg.mvTable_SizingFixedFit):
        dpg.add_table_column(label="#", width_fixed=True, init_width_or_weight=36)
        dpg.add_table_column(label="Step", width_fixed=True, init_width_or_weight=560)
        dpg.add_table_column(label="", width_fixed=True, init_width_or_weight=90)
        for index, step in enumerate(_pipeline_steps):
            label = gc.TRANSFORM_OPS.get(step.op, gc.TransformOpSpec(step.op, "")).label
            text = summaries[index] if index < len(summaries) else f"{label} (not yet applied)"
            with ui.table_row():
                dpg.add_text(f"#{index + 1}")
                dpg.add_text(text, wrap=540)
                dpg.add_button(label="Remove", callback=on_remove_step, user_data=index)


def _add_pipeline_step(step: gc.TransformStep) -> None:
    """Validate a candidate step by actually running the WHOLE pipeline with
    it appended against the live preview before committing to it -- the same
    "verify against real behavior" discipline used throughout this app,
    rather than a hand-written pre-check that could drift from reality."""
    global _pipeline_steps
    trial = [*_pipeline_steps, step]
    try:
        gc.apply_transform_pipeline(_preview, trial)
    except ValueError as exc:
        dpg.set_value("prep_error_text", str(exc))
        return
    _pipeline_steps = trial
    dpg.set_value("prep_error_text", "")
    _refresh_pipeline_preview()
    on_config_changed()


def on_remove_step(sender: ItemId, app_data: Any, user_data: int) -> None:
    global _pipeline_steps
    _pipeline_steps = [step for index, step in enumerate(_pipeline_steps) if index != user_data]
    dpg.set_value("prep_error_text", "")
    _refresh_pipeline_preview()
    on_config_changed()


def on_clear_pipeline(sender: ItemId = 0, app_data: Any = None) -> None:
    global _pipeline_steps
    _pipeline_steps = []
    dpg.set_value("prep_error_text", "")
    _refresh_pipeline_preview()
    on_config_changed()


def on_prep_op_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    key = _PREP_LABEL_TO_KEY.get(_text("prep_op_combo"))
    for op_key in gc.TRANSFORM_OPS:
        dpg.configure_item(f"prep_params_{op_key}", show=op_key == key)
    dpg.set_value("prep_error_text", "")


def on_prep_fill_strategy_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    dpg.configure_item("prep_fill_custom_group", show=_text("prep_fill_strategy_combo") == "Custom value")


def _collect_prep_params(key: str) -> dict[str, Any]:
    if key == "rename_column":
        return {"column": _text("prep_rename_column_combo"), "new_name": _text("prep_rename_new_name_input")}
    if key == "cast_column_dtype":
        return {"column": _text("prep_cast_column_combo"), "dtype": _text("prep_cast_dtype_combo")}
    if key == "drop_columns":
        column = _text("prep_drop_column_combo")
        if not column:
            raise ValueError("Choose a column to drop.")
        return {"columns": [column]}
    if key == "clean_text_column":
        return {"column": _text("prep_text_column_combo"), "mode": _text("prep_text_mode_combo")}
    if key == "extract_datetime_parts":
        parts = [part for part in gc.DATETIME_PARTS if _flag(f"prep_dt_part_{part}")]
        return {"column": _text("prep_dt_column_combo"), "parts": parts}
    if key == "filter_rows":
        return {
            "column": _text("prep_filter_column_combo"), "operator": _text("prep_filter_operator_combo"),
            "value": _text("prep_filter_value_input"), "keep": _text("prep_filter_keep_radio") == "Keep",
        }
    if key == "dedupe_rows":
        return {"keep": _text("prep_dedupe_keep_combo")}
    if key == "drop_missing_rows":
        column = _text("prep_dropna_column_combo")
        return {"column": None if column in ("", "(any column)") else column}
    if key == "fill_missing_column":
        return {
            "column": _text("prep_fill_column_combo"), "strategy": _text("prep_fill_strategy_combo"),
            "custom_value": _text("prep_fill_custom_input"),
        }
    if key == "flag_missing_column":
        return {"column": _text("prep_flag_column_combo")}
    if key == "clip_column_outliers":
        return {
            "column": _text("prep_clip_column_combo"), "method": _text("prep_clip_method_combo"),
            "low": _number("prep_clip_low_input"), "high": _number("prep_clip_high_input"),
        }
    if key == "threshold_flag_column":
        return {
            "column": _text("prep_threshold_column_combo"), "operator": _text("prep_threshold_operator_combo"),
            "threshold": _number("prep_threshold_value_input"), "new_column": _text("prep_threshold_new_column_input"),
        }
    if key == "combine_columns":
        return {
            "left": _text("prep_combine_a_combo"), "operator": _text("prep_combine_operator_combo"),
            "right": _text("prep_combine_b_combo"), "new_column": _text("prep_combine_new_column_input"),
        }
    raise ValueError(f"Unknown action: {key}")


def on_add_prep_step(sender: ItemId = 0, app_data: Any = None) -> None:
    key = _PREP_LABEL_TO_KEY.get(_text("prep_op_combo"))
    if key is None:
        dpg.set_value("prep_error_text", "Choose an action first.")
        return
    try:
        params = _collect_prep_params(key)
    except ValueError as exc:
        dpg.set_value("prep_error_text", str(exc))
        return
    _add_pipeline_step(gc.TransformStep(op=key, params=params))


def on_mode_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    # Both the Data explorer and Model studio pages have their own mode
    # radio (same tag can't be reused on two widgets), so whichever one
    # fired this callback sets the mode and both are kept in sync here.
    mode = app_data if app_data in ("classify", "regress") else _text("mode_radio")
    dpg.set_value("mode_radio", mode)
    dpg.set_value("mode_radio_studio", mode)
    classify = mode == "classify"
    dpg.configure_item("clf_group", show=classify)
    dpg.configure_item("reg_group", show=not classify)
    metrics = gc.CLS_METRICS if classify else gc.REG_METRICS
    dpg.configure_item("metric_combo", items=metrics)
    dpg.set_value("metric_combo", gc.default_metric(metrics) if classify else "mae")
    cont_stats = gc.CONT_CLS_STATS if classify else gc.CONT_REG_STATS
    cat_stats = gc.CAT_CLS_STATS if classify else gc.CAT_REG_STATS
    pred_scores = gc.CLS_PRED_SCORES if classify else gc.REG_PRED_SCORES
    dpg.configure_item("filter_assoc_cont_combo", items=cont_stats)
    dpg.set_value("filter_assoc_cont_combo", cont_stats[0] if "mut_info" not in cont_stats else "mut_info")
    dpg.configure_item("filter_assoc_cat_combo", items=cat_stats)
    dpg.set_value("filter_assoc_cat_combo", "mut_info" if "mut_info" in cat_stats else cat_stats[0])
    dpg.configure_item("filter_pred_combo", items=pred_scores)
    dpg.set_value("filter_pred_combo", pred_scores[0])
    on_config_changed()


def on_feat_select_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    if sender == "feat_none" and app_data:
        for method in gc.FEAT_SELECT_METHODS:
            if method != "none":
                dpg.set_value(f"feat_{method}", False)
    elif app_data:
        dpg.set_value("feat_none", False)
    selected = _active_feat_select()
    dpg.configure_item("filter_group", show="filter" in selected)
    dpg.configure_item("embed_group", show="embed" in selected)
    dpg.configure_item("wrapper_group", show="wrap" in selected)
    on_config_changed()


def on_redundant_toggle_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    dpg.configure_item("redundant_group", show=_flag("redundant_wrapper_checkbox"))
    on_config_changed()


def on_select_models(sender: ItemId, app_data: Any, user_data: bool) -> None:
    names, prefix = (gc.CLASSIFIERS, "clf") if _text("mode_radio") == "classify" else (gc.REGRESSORS, "reg")
    for name in names:
        dpg.set_value(f"{prefix}_{name}", user_data)
    on_config_changed()


def on_preset_selected(sender: ItemId = 0, app_data: Any = None) -> None:
    # Presets also work before selecting a dataset.
    baseline = gc.RunConfig(data_path=_data_path or Path("dataset.csv"), target=_text("target_combo"),
                            mode=_text("mode_radio"), models=[], feat_select=["none"])
    cfg = gc.apply_preset(baseline, _text("preset_combo"))
    prefix = "clf" if cfg.mode == "classify" else "reg"
    names = gc.CLASSIFIERS if cfg.mode == "classify" else gc.REGRESSORS
    for name in names:
        dpg.set_value(f"{prefix}_{name}", name in cfg.models)
    for method in gc.FEAT_SELECT_METHODS:
        dpg.set_value(f"feat_{method}", method in cfg.feat_select)
    dpg.set_value("htune_trials_input", cfg.htune_trials)
    dpg.set_value("no_preds_checkbox", cfg.no_preds)
    on_feat_select_changed()
    _set_status(f"Preset: {_text('preset_combo')}", SUCCESS)


def on_config_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    try:
        cfg = _build_run_config()
        issues = gc.validate_config(cfg, _active_columns())
        command = subprocess.list2cmdline(["df-analyze", *gc.build_argv(cfg)])
        dpg.set_value("command_text", command)
        dpg.set_value("validation_text", "\n".join(issues) if issues else "Ready to run. Dataset, target, models and settings passed validation.")
        dpg.configure_item("validation_text", color=WARNING if issues else SUCCESS)
        dpg.set_value("experiment_summary", f"{cfg.mode.upper()}  |  {len(cfg.models)} models  |  {cfg.htune_trials} tuning trials  |  {cfg.test_val_size:.0%} holdout")
    except Exception as exc:
        dpg.set_value("validation_text", str(exc))
        dpg.configure_item("validation_text", color=WARNING)
        dpg.set_value("command_text", "Select a dataset to generate the reproducible command.")


def on_outdir_dir_selected(sender: ItemId, app_data: Any) -> None:
    value = app_data.get("file_path_name", "") if isinstance(app_data, dict) else ""
    if value:
        dpg.set_value("outdir_input", str(value))
        on_config_changed()


def on_copy_command(sender: ItemId = 0, app_data: Any = None) -> None:
    dpg.set_clipboard_text(_text("command_text"))
    _set_status("Command copied", SUCCESS)


def on_export_config(sender: ItemId = 0, app_data: Any = None) -> None:
    try:
        cfg = _build_run_config()
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        path = cfg.outdir / f"experiment-{time.time_ns()}.json"
        content = _workflow.capture().to_json(cfg) if _workflow is not None and _workflow.active else gc.config_to_json(cfg)
        path.write_text(content, encoding="utf-8")
        _set_status(f"Saved: {path.name}", SUCCESS)
        dpg.set_value("settings_notice", str(path))
    except Exception as exc:
        _set_status(f"Cannot save experiment: {exc}", ERROR)


def on_import_config(sender: ItemId, app_data: Any) -> None:
    global _pipeline_steps, _extra_targets
    value = app_data.get("file_path_name", "") if isinstance(app_data, dict) else ""
    if not value:
        return
    try:
        content = Path(str(value)).read_text(encoding="utf-8")
        cfg = gc.config_from_json(content)
        imported_flow = import_workflow(content)
        if imported_flow is not None:
            imported_flow.compile(cfg)
        dpg.set_value("input_mode_combo", cfg.input_mode)
        dpg.set_value("delimiter_combo", next((k for k, v in gc.DELIMITERS.items() if v == cfg.separator), "Custom"))
        dpg.set_value("delimiter_custom", cfg.separator)
        dpg.set_value("test_files_input", "\n".join(cfg.test_files))
        dpg.set_value("tests_method_combo", cfg.tests_method)
        advanced_settings.restore(cfg.advanced_options)
        if not _load_data(cfg.data_path):
            return
        _pipeline_steps = list(cfg.pipeline_steps)
        _refresh_pipeline_preview()
        dpg.set_value("mode_radio", cfg.mode)
        on_mode_changed()
        for tag, item_value in {"target_combo": cfg.target, "embed_model_combo": cfg.embed_model,
                               "wrapper_method_combo": cfg.wrapper_method, "wrapper_model_combo": cfg.wrapper_model,
                               "htune_trials_input": cfg.htune_trials, "metric_combo": cfg.htune_metric,
                               "test_val_size_slider": cfg.test_val_size, "seed_input": cfg.seed,
                               "outdir_input": str(cfg.outdir), "extra_args_input": cfg.extra_args,
                               "grouper_combo": cfg.grouper or "", "no_preds_checkbox": cfg.no_preds,
                               "adaptive_error_checkbox": cfg.adaptive_error,
                               "norm_combo": cfg.norm, "nan_combo": cfg.nan,
                               "drop_duplicates_checkbox": cfg.drop_duplicates,
                               "n_feat_filter_input": cfg.n_feat_filter,
                               "n_feat_wrapper_input": cfg.n_feat_wrapper,
                               "filter_method_combo": cfg.filter_method,
                               "redundant_wrapper_checkbox": cfg.redundant_wrapper_selection,
                               "redundant_threshold_input": cfg.redundant_threshold,
                               "redundant_corr_threshold_input": cfg.redundant_corr_threshold,
                               "multi_target_checkbox": bool(cfg.extra_targets),
                               "mt_agg_strategy_combo": cfg.mt_agg_strategy,
                               "mt_top_k_input": cfg.mt_top_k}.items():
            dpg.set_value(tag, item_value)
        dpg.set_value("filter_assoc_cont_combo", cfg.filter_assoc_cont_classify if cfg.mode == "classify" else cfg.filter_assoc_cont_regress)
        dpg.set_value("filter_assoc_cat_combo", cfg.filter_assoc_cat_classify if cfg.mode == "classify" else cfg.filter_assoc_cat_regress)
        dpg.set_value("filter_pred_combo", cfg.filter_pred_classify if cfg.mode == "classify" else cfg.filter_pred_regress)
        for prefix, names in (("clf", gc.CLASSIFIERS), ("reg", gc.REGRESSORS)):
            for name in names:
                dpg.set_value(f"{prefix}_{name}", name in cfg.models)
        for method in gc.FEAT_SELECT_METHODS:
            dpg.set_value(f"feat_{method}", method in cfg.feat_select)
        for role, columns in (("Categorical", cfg.categoricals), ("Ordinal", cfg.ordinals), ("Exclude", cfg.drops)):
            _roles.update({col: role for col in columns})
        _extra_targets = set(cfg.extra_targets)
        render_roles()
        on_role_column_changed()  # sync the Role dropdown to the just-restored roles
        on_multi_target_toggle_changed()  # sync the multi-target group's visibility
        render_multi_target_checklist()
        on_feat_select_changed()
        if _workflow is not None:
            if imported_flow is not None:
                _workflow.load(imported_flow)
            else:
                dpg.set_value("workflow_enabled", False)
                _workflow.rebuild()
        _set_status("Experiment loaded", SUCCESS)
    except Exception as exc:
        _set_status(f"Cannot load experiment: {exc}", ERROR)


def _apply_data_preparation(cfg: gc.RunConfig) -> gc.RunConfig:
    """Apply GUI-side preparation to a real, kept file before the CLI ever
    sees the data, so the analysis -- and the reproducible command shown for
    it -- both refer to the exact data that was actually used, not the
    original unmodified file: first the legacy "drop duplicate rows" quick
    option (unchanged behavior), then any steps added on the Data
    preparation tab, in order."""
    if not cfg.drop_duplicates and not cfg.pipeline_steps:
        return cfg
    full = gc.load_full_dataset(cfg.data_path, cfg.separator, cfg.input_mode)
    summaries: list[str] = []
    if cfg.drop_duplicates:
        full, removed = gc.drop_duplicate_rows(full)
        if removed:
            summaries.append(f"Removed {removed:,} duplicate row(s) (quick option).")
    if cfg.pipeline_steps:
        full, step_summaries = gc.apply_transform_pipeline(full, cfg.pipeline_steps)
        summaries.extend(step_summaries)
    if not summaries:
        return cfg
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    prepared_path = cfg.outdir / f"{cfg.data_path.stem}_prepared{cfg.data_path.suffix}"
    gc.save_prepared_dataset(full, prepared_path, cfg)
    test_files = gc.prepare_external_test_files(cfg)
    for line in summaries:
        _line_buffer.feed(line + "\n")
    dpg.set_value("log_box", _line_buffer.render())
    dpg.set_value(
        "run_stage",
        f"Applied {len(summaries)} preparation step(s); analyzing {prepared_path.name} "
        f"({len(full):,} rows, {full.shape[1]} columns).",
    )
    # Prepared CSV output uses the writer's comma delimiter.
    return replace(cfg, data_path=prepared_path, separator=",",
                   test_files=test_files,
                   input_mode=cfg.input_mode)


def on_run_clicked(sender: ItemId = 0, app_data: Any = None) -> None:
    global _current_run, _line_buffer, _run_outdir, _run_started
    if _current_run is not None:
        return
    if embedding_panel.running():
        _set_status("Wait for or cancel the embedding task first.", WARNING)
        return
    # Reset the log first -- _apply_data_preparation (below) feeds real
    # preparation summaries into it, and those must survive, not be wiped by
    # a later reset once the run has already started.
    _line_buffer = LineBuffer()
    dpg.set_value("log_box", "")
    try:
        cfg = _build_run_config()
        issues = gc.validate_config(cfg, _active_columns())
        if issues:
            raise ValueError(issues[0])
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        cfg = replace(cfg, outdir=cfg.outdir / run_name)
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        content = _workflow.capture().to_json(cfg) if _workflow is not None and _workflow.active else gc.config_to_json(cfg)
        (cfg.outdir / "studio_experiment.json").write_text(content, encoding="utf-8")
        cfg = _apply_data_preparation(cfg)
        on_config_changed()
        # If preparation swapped in a different file, the command preview
        # above (rebuilt from raw widget state) would otherwise still show
        # the original path -- keep "reproducible command" honest about the
        # data actually used for this run.
        dpg.set_value("command_text", subprocess.list2cmdline(["df-analyze", *gc.build_argv(cfg)]))
        handle = start_run(cfg)
    except Exception as exc:
        _set_status(str(exc), ERROR)
        dpg.set_value("validation_text", str(exc))
        return
    _current_run = handle
    _run_outdir = cfg.outdir
    _run_started = time.monotonic()
    dpg.set_value("run_stage", "Starting analysis")
    _set_status("Analysis running", ACCENT_LIGHT)
    dpg.configure_item("run_button", enabled=False)
    dpg.configure_item("cancel_button", enabled=handle.can_cancel)
    dpg.configure_item("run_loading_indicator", show=True)
    dpg.set_value("cancel_note", "Stop cancels this analysis and its worker processes." if handle.can_cancel else "This packaged runner finishes its active analysis before accepting another run.")
    for key, _ in NAV_ITEMS:
        if key in {"data", "prepare", "models", "tuning", "workflow"}:
            dpg.configure_item(f"page_{key}", enabled=False)
    on_nav_selected(user_data="run")


def on_cancel_clicked(sender: ItemId = 0, app_data: Any = None) -> None:
    if _current_run is not None and _current_run.cancel():
        _set_status("Stopping analysis...", WARNING)
        dpg.configure_item("cancel_button", enabled=False)


def poll_run_state() -> None:
    global _current_run
    if _current_run is None:
        return
    seconds = int(time.monotonic() - _run_started)
    dpg.set_value("elapsed_text", f"Elapsed  {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
    changed = False
    # Bound work per frame so verbose models cannot freeze the interface.
    for _ in range(300):
        try:
            chunk = _current_run.log_queue.get_nowait()
        except queue.Empty:
            break
        _line_buffer.feed(chunk)
        changed = True
    if changed:
        content = _line_buffer.render()
        dpg.set_value("log_box", content)
        if _flag("follow_log_checkbox"):
            dpg.set_y_scroll("log_window", dpg.get_y_scroll_max("log_window"))
        stage = next((line.strip()[:120] for line in reversed(content.splitlines()) if line.strip()), "Running")
        dpg.set_value("run_stage", stage)
    # The runner publishes completion only after stdout and workers finish.
    try:
        status, message = _current_run.done_queue.get_nowait()
    except queue.Empty:
        return
    while True:
        try:
            _line_buffer.feed(_current_run.log_queue.get_nowait())
        except queue.Empty:
            break
    dpg.set_value("log_box", _line_buffer.render())
    dpg.configure_item("run_button", enabled=True)
    dpg.configure_item("cancel_button", enabled=False)
    dpg.configure_item("run_loading_indicator", show=False)
    for key in ("data", "prepare", "models", "tuning", "workflow"):
        dpg.configure_item(f"page_{key}", enabled=True)
    if status == "success":
        _set_status("Analysis complete", SUCCESS)
        dpg.set_value("run_stage", "Complete. Results are ready in Results library.")
        if _run_outdir is not None:
            refresh_results(_run_outdir)
    elif status == "cancelled":
        _set_status("Analysis stopped", WARNING)
        dpg.set_value("run_stage", "Stopped. Partial output may remain in the experiment folder.")
    else:
        summary = message.splitlines()[0] if message else "Unknown failure"
        _set_status("Analysis failed - check log", ERROR)
        dpg.set_value("run_stage", summary)
        _line_buffer.feed(f"\n[df-analyze] {message}\n")
        dpg.set_value("log_box", _line_buffer.render())
    _current_run = None


def on_save_log(sender: ItemId = 0, app_data: Any = None) -> None:
    try:
        outdir = _run_outdir or Path(_text("outdir_input")).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"desktop-log-{time.time_ns()}.txt"
        path.write_text(_line_buffer.render(), encoding="utf-8")
        _set_status(f"Saved: {path.name}", SUCCESS)
    except OSError as exc:
        _set_status(f"Cannot save log: {exc}", ERROR)


def on_refresh_results(sender: ItemId = 0, app_data: Any = None) -> None:
    refresh_results(Path(_text("outdir_input")).expanduser())


def refresh_results(outdir: Path) -> None:
    global _result_dirs
    try:
        paths = gc.list_runs(outdir)
        _result_dirs = {f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(path.stat().st_mtime))}  |  {path.relative_to(outdir) if path.is_relative_to(outdir) else path}": path for path in paths}
        labels = list(_result_dirs)
        dpg.configure_item("run_history_combo", items=labels)
        dpg.set_value("run_history_combo", labels[0] if labels else "")
        dpg.set_value("library_count", f"{len(paths)} completed result sets")
        if labels:
            on_history_changed()
        else:
            _clear_results("No completed results in this output folder yet.")
    except Exception as exc:
        _clear_results(f"Cannot read results: {exc}")


def _clear_results(message: str) -> None:
    global _result_dir, _result_table, _artifacts, _per_target_table
    _result_dir, _result_table, _artifacts, _per_target_table = None, pd.DataFrame(), {}, None
    dpg.configure_item("results_target_group", show=False)
    dpg.configure_item("results_target_combo", items=[])
    dpg.set_value("results_status_text", message)
    dpg.set_value("results_best_model", "--")
    _animations.pop("results_best_value", None)
    _kpi_raw.pop("results_best_value", None)
    dpg.set_value("results_best_value", "--")
    dpg.set_value("results_best_text", "Run an experiment or select an output folder to explore results.")
    dpg.set_value("results_baseline_note", "")
    dpg.configure_item("results_baseline_note", color=TEXT_MUTED)
    dpg.set_value("results_report_text", "")
    dpg.set_value("artifact_preview", "")
    dpg.configure_item("artifact_combo", items=[])
    dpg.set_value("artifact_combo", "")
    dpg.configure_item("results_metric_combo", items=[])
    dpg.set_value("results_metric_combo", "")
    dpg.delete_item("results_y_axis", children_only=True)
    dpg.reset_axis_ticks("results_x_axis")
    dpg.delete_item("results_table_container", children_only=True)


def on_history_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    global _result_dir, _result_table, _artifacts, _per_target_table
    path = _result_dirs.get(_text("run_history_combo"))
    if path is None:
        return
    try:
        table = gc.load_results_table(path)
        metrics = gc.metric_options(table)
        _result_dir, _result_table = path, table
        # performance_long_table_per_target.csv only exists for a run that
        # analyzed more than one target together -- absent otherwise, in
        # which case load_per_target_table returns None and the selector
        # stays hidden, leaving this exactly like a single-target run today.
        _per_target_table = gc.load_per_target_table(path)
        if _per_target_table is not None:
            targets = sorted(_per_target_table["target"].dropna().astype(str).unique().tolist())
            dpg.configure_item("results_target_combo", items=["All targets (aggregate)", *targets])
            dpg.set_value("results_target_combo", "All targets (aggregate)")
            dpg.configure_item("results_target_group", show=True)
        else:
            dpg.configure_item("results_target_group", show=False)
        dpg.configure_item("results_metric_combo", items=metrics)
        dpg.set_value("results_metric_combo", gc.default_metric(metrics))
        dpg.set_value("results_status_text", str(path))
        report = gc.report_artifact(path)
        dpg.set_value("results_report_text", report.read_text(encoding="utf-8", errors="replace")[:100000] if report.exists() else "No Markdown report found for this run.")
        artifact_root = gc.run_artifact_root(path).resolve()
        files = sorted(p for p in artifact_root.rglob("*") if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(artifact_root))
        _artifacts = {str(file.relative_to(artifact_root)): file for file in files}
        dpg.configure_item("artifact_combo", items=list(_artifacts))
        dpg.set_value("artifact_combo", next(iter(_artifacts), ""))
        on_artifact_changed()
        on_results_metric_changed()
    except Exception as exc:
        _clear_results(f"Cannot read selected run: {exc}")


def _current_results_source() -> pd.DataFrame:
    """The table the leaderboard/chart should actually read from: the
    per-target slice if a specific target is chosen, else the run's regular
    (aggregate, mean-across-targets) results -- identical to today's
    single-target behavior when no multi-target selector is even shown."""
    if _per_target_table is not None:
        choice = _text("results_target_combo")
        if choice and choice != "All targets (aggregate)":
            return _per_target_table.loc[_per_target_table["target"] == choice]
    return _result_table


def on_results_metric_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    source = _current_results_source()
    if not source.empty:
        render_results(source, _text("results_metric_combo"))


def on_results_target_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    if _result_dir is not None:
        target = _text("results_target_combo")
        report = gc.report_artifact(_result_dir, target if target and target != "All targets (aggregate)" else None)
        dpg.set_value("results_report_text", report.read_text(encoding="utf-8", errors="replace")[:100000] if report.exists() else "No report found for this target.")
    on_results_metric_changed()


def _render_leaderboard_table(view: pd.DataFrame, metric: str, limit: int = 100) -> None:
    """A ranked leaderboard, not a plain dump: numbered rows, the top three
    tinted like medals, the winning row's background highlighted, and a
    data bar in the score column whose length encodes relative performance
    (Excel-style conditional formatting) -- green for the winner, matching
    the bar chart above it, blue for the rest."""
    dpg.delete_item("results_table_container", children_only=True)
    if view.empty:
        dpg.add_text("No holdout scores match this selection.", color=TEXT_MUTED, parent="results_table_container")
        return
    rows = view.head(limit).reset_index(drop=True)
    scores = rows["holdout"].astype(float)
    lo, hi = float(scores.min()), float(scores.max())
    span = hi - lo
    ascending = gc.ascending_for_metric(metric)
    extra_columns = [(key, label) for key, label in (("trainset", "Train"), ("5-fold", "5-fold")) if key in rows.columns]

    with ui.table(parent="results_table_container", header_row=True, row_background=True,
                  borders_innerH=True, borders_outerH=True, scrollY=True,
                  policy=dpg.mvTable_SizingFixedFit) as table_id:
        dpg.add_table_column(label="#", width_fixed=True, init_width_or_weight=36)
        dpg.add_table_column(label="Configuration", width_fixed=True, init_width_or_weight=320)
        dpg.add_table_column(label=f"Holdout {metric}", width_fixed=True, init_width_or_weight=230)
        for _, label in extra_columns:
            dpg.add_table_column(label=label, width_fixed=True, init_width_or_weight=90)
        for i, row in rows.iterrows():
            with ui.table_row():
                dpg.add_text(f"#{i + 1}")
                dpg.add_text(str(row["combo"])[:60])
                value = float(row["holdout"])
                normalized = 0.5 if span <= 1e-12 else (
                    (hi - value) / span if ascending else (value - lo) / span
                )
                bar = dpg.add_progress_bar(default_value=max(0.0, min(1.0, normalized)),
                                           width=140, overlay=f"{value:.4f}")
                dpg.bind_item_theme(bar, _THEMES["best_progress"] if i == 0 else _THEMES["rest_progress"])
                for key, _ in extra_columns:
                    raw = row.get(key)
                    dpg.add_text("--" if pd.isna(raw) else f"{float(raw):.4f}")
        for rank, color in enumerate((RANK_GOLD, RANK_SILVER, RANK_BRONZE)):
            if rank < len(rows):
                dpg.highlight_table_cell(table_id, rank, 0, (*color, 90))
        dpg.highlight_table_row(table_id, 0, (*SUCCESS, 30))


def render_results(long_table: pd.DataFrame, metric: str) -> None:
    view = gc.filtered_sorted_view(long_table, metric)
    search = _text("result_search").strip()
    if search:
        view = view.loc[view["combo"].str.contains(search, case=False, regex=False)]
    ascending = gc.ascending_for_metric(metric)
    direction = "Lower" if ascending else "Higher"
    dpg.configure_item("results_x_axis", label="Model / feature selection")
    dpg.configure_item("results_y_axis", label=f"{metric} ({direction.lower()} is better)")
    dpg.delete_item("results_y_axis", children_only=True)
    dpg.reset_axis_ticks("results_x_axis")

    if view.empty:
        dpg.set_value("results_best_model", "--")
        _kpi_raw.pop("results_best_value", None)
        dpg.set_value("results_best_value", "--")
        dpg.set_value("results_best_text", f"No holdout {metric} scores match this selection.")
        dpg.set_value("results_baseline_note", "")
        _render_leaderboard_table(view, metric)
        return

    best = view.iloc[0]
    dpg.set_value("results_best_model", str(best["combo"]))
    _animate_number("results_best_value", float(best["holdout"]), lambda v: f"{v:.4f}")
    dpg.set_value("results_best_text", f"Holdout {metric}  --  {direction.lower()} is better")

    baseline_rows = view.loc[view["model"] == "dummy"]
    if best["model"] == "dummy":
        dpg.set_value(
            "results_baseline_note",
            "This is the naive (dummy) baseline -- no trained model beat it on this metric. "
            "The dataset may carry little signal for this target, or try more tuning trials, "
            "different features, or a different feature-selection method.",
        )
        dpg.configure_item("results_baseline_note", color=WARNING)
    elif not baseline_rows.empty and float(baseline_rows.iloc[0]["holdout"]):
        baseline = float(baseline_rows.iloc[0]["holdout"])
        improvement = (baseline - best["holdout"]) / abs(baseline) if ascending else (best["holdout"] - baseline) / abs(baseline)
        dpg.set_value("results_baseline_note", f"{improvement * 100:.1f}% better than the dummy baseline ({baseline:.4f}).")
        dpg.configure_item("results_baseline_note", color=SUCCESS if improvement > 0 else WARNING)
    else:
        dpg.set_value("results_baseline_note", "")

    top = view.head(15)
    positions = [float(i) for i in range(len(top))]
    values = top["holdout"].astype(float).tolist()
    best_series = dpg.add_bar_series(positions[:1], values[:1], weight=0.6, label="Best", parent="results_y_axis")
    dpg.bind_item_theme(best_series, _THEMES["best_bar"])
    if len(values) > 1:
        rest_series = dpg.add_bar_series(positions[1:], values[1:], weight=0.6, label="Other", parent="results_y_axis")
        dpg.bind_item_theme(rest_series, _THEMES["rest_bar"])
    dpg.set_axis_ticks("results_x_axis", tuple((str(name)[:24], pos) for name, pos in zip(top["combo"], positions)))
    dpg.fit_axis_data("results_x_axis")
    dpg.fit_axis_data("results_y_axis")
    _render_leaderboard_table(view, metric)


def on_artifact_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    path = _artifacts.get(_text("artifact_combo"))
    if path is None:
        dpg.set_value("artifact_preview", "")
        return
    try:
        if path.suffix.lower() in {".csv", ".md", ".txt", ".json", ".html"}:
            with path.open(encoding="utf-8", errors="replace") as stream:
                content = stream.read(60000)
            dpg.set_value("artifact_preview", content)
        else:
            dpg.set_value("artifact_preview", f"{path.name}\nUse Open artifact to view this file in its associated application.")
    except OSError as exc:
        dpg.set_value("artifact_preview", str(exc))


def on_open_artifact(sender: ItemId = 0, app_data: Any = None) -> None:
    path = _artifacts.get(_text("artifact_combo"))
    if path is not None and path.exists():
        webbrowser.open(path.resolve().as_uri())


def on_open_output(sender: ItemId = 0, app_data: Any = None) -> None:
    path = _result_dir or Path(_text("outdir_input")).expanduser()
    if path.exists():
        webbrowser.open(path.resolve().as_uri())
    else:
        _set_status("The output folder does not exist yet.", WARNING)


def on_export_scores(sender: ItemId = 0, app_data: Any = None) -> None:
    source = _current_results_source()
    if _result_dir is None or source.empty:
        return
    try:
        view = gc.filtered_sorted_view(source, _text("results_metric_combo"))
        term = _text("result_search").strip()
        if term:
            view = view.loc[view["combo"].str.contains(term, case=False, regex=False)]
        path = _result_dir / f"leaderboard-{time.time_ns()}.csv"
        view.to_csv(path, index=False)
        _set_status(f"Exported: {path.name}", SUCCESS)
    except (OSError, ValueError) as exc:
        _set_status(f"Cannot export: {exc}", ERROR)


BG_MAIN = (11, 17, 30)
BG_SIDEBAR = (8, 13, 24)
BG_CARD = (20, 28, 46)
BG_CARD_ALT = (28, 39, 63)
ACCENT = (61, 145, 235)
ACCENT_HOVER = (95, 173, 246)
ACCENT_ACTIVE = (43, 112, 191)
ACCENT_LIGHT = (150, 200, 250)
TEXT_PRIMARY = (226, 232, 240)
TEXT_MUTED = (138, 150, 172)
BORDER = (42, 54, 78)
SUCCESS = (58, 199, 130)
ERROR = (235, 105, 105)
WARNING = (232, 178, 78)

# Reserved solely for leaderboard rank badges (#1/#2/#3) -- a fixed,
# universally-recognized "medal" mapping, never reused as a general
# categorical or status color elsewhere in the app.
RANK_GOLD = (240, 190, 60)
RANK_SILVER = (176, 190, 205)
RANK_BRONZE = (196, 132, 78)

NAV_ITEMS = [
    ("overview", "01   Workspace"),
    ("workflow", "02   Visual workflow"),
    ("data", "03   Data explorer"),
    ("visualize", "04   Data visualization"),
    ("prepare", "05   Preprocessing"),
    ("models", "06   Model studio"),
    ("tuning", "07   Experiment settings"),
    ("run", "08   Run monitor"),
    ("results", "09   Results library"),
    ("embeddings", "10   Text / image embeddings"),
    ("notebook", "11   Python notebook"),
]

_FONTS: dict[str, ItemId] = {}
_THEMES: dict[str, ItemId] = {}

# ---------------------------------------------------------------------------
# Matplotlib/seaborn chart rendering for the Data visualization page. Charts
# are drawn as real matplotlib figures (dark-styled to match the rest of the
# app, transparent background so the card behind shows through) and
# published into a fixed-size DPG texture -- this is what gives them a
# standard data-science-plot look (anti-aliasing, real box plots, colorbars)
# instead of DPG's more basic native implot widgets.
# ---------------------------------------------------------------------------

CHART_DPI = 100
DEFAULT_CHART_SIZE = (860, 380)  # (width, height) px -- fixed per chart so
                                  # its texture can be updated in place via
                                  # dpg.set_value() on redraw instead of
                                  # being torn down and recreated.
# The categorical-breakdown tab shows two charts side by side, so those two
# get a narrower size -- otherwise they don't both fit in the content width.
CHART_SIZES: dict[str, tuple[int, int]] = {
    "catbar": (410, 380),
    "catpie": (410, 380),
}
CATEGORICAL_PALETTE = [
    "#3d91eb", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#4fae4f", "#8a7fe0", "#e34948",
]
CHART_KEYS = ["heatmap", "histogram", "catbar", "catpie", "relationship", "box", "trend", "pca"]
_TEXTURES: dict[str, ItemId] = {}


def _chart_size(key: str) -> tuple[int, int]:
    return CHART_SIZES.get(key, DEFAULT_CHART_SIZE)


def _rgb01(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255, color[1] / 255, color[2] / 255)


def _new_figure(key: str) -> tuple[Any, Any]:
    width, height = _chart_size(key)
    fig = plt.figure(figsize=(width / CHART_DPI, height / CHART_DPI), dpi=CHART_DPI)
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    return fig, ax


def _style_axes(ax: Any, grid: bool = True) -> None:
    text, muted, border = _rgb01(TEXT_PRIMARY), _rgb01(TEXT_MUTED), _rgb01(BORDER)
    ax.title.set_color(text)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_color(muted)
    ax.yaxis.label.set_color(muted)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.tick_params(colors=muted, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color=border, alpha=0.5, linewidth=0.6)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _publish_figure(fig: Any, key: str) -> None:
    fig.tight_layout()
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32) / 255.0  # (h, w, 4)
    plt.close(fig)
    target_w, target_h = _chart_size(key)
    actual_h, actual_w = buffer.shape[0], buffer.shape[1]
    if (actual_w, actual_h) != (target_w, target_h):
        # matplotlib's figsize(inches) * dpi -> pixel-size conversion can be
        # off by a pixel or two from what was requested (float rounding), but
        # the DPG texture was allocated at the *requested* size. Writing a
        # differently-sized flat buffer into it is a raw buffer overrun in
        # DPG's C++ layer -- a hard segfault, not a catchable Python
        # exception -- so always crop/pad to the exact expected size before
        # publishing, regardless of what matplotlib actually produced.
        fixed = np.zeros((target_h, target_w, 4), dtype=np.float32)
        copy_h, copy_w = min(actual_h, target_h), min(actual_w, target_w)
        fixed[:copy_h, :copy_w] = buffer[:copy_h, :copy_w]
        buffer = fixed
    dpg.set_value(_TEXTURES[key], buffer.flatten().tolist())


def _chart_placeholder(key: str, message: str) -> None:
    fig, ax = _new_figure(key)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=_rgb01(TEXT_MUTED), fontsize=10, wrap=True)
    _publish_figure(fig, key)


def _build_chart_textures() -> None:
    with ui.texture_registry():
        for key in CHART_KEYS:
            width, height = _chart_size(key)
            blank = [0.0] * (width * height * 4)
            _TEXTURES[key] = dpg.add_dynamic_texture(width, height, blank)


def _chart_image(key: str) -> None:
    width, height = _chart_size(key)
    dpg.add_image(_TEXTURES[key], width=width, height=height)


def _build_theme() -> None:
    with ui.theme() as global_theme:
        with ui.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BG_MAIN)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BG_CARD)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (*ACCENT, 90))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (*ACCENT, 140))
            dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, TEXT_MUTED)
            dpg.add_theme_color(dpg.mvThemeCol_Border, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_Button, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, ACCENT_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (*ACCENT, 80))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (*ACCENT, 120))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, ACCENT_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_TabSelected, ACCENT_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, BG_MAIN)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BG_MAIN)
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, BORDER)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 10)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 10, 6)
    dpg.bind_theme(global_theme)

    with ui.theme() as run_button_theme:
        with ui.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 26, 12)
    _THEMES["run_button"] = run_button_theme

    with ui.theme() as sidebar_theme:
        with ui.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BG_SIDEBAR)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0)
    _THEMES["sidebar"] = sidebar_theme

    with ui.theme() as highlight_theme:
        with ui.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (*ACCENT, 35))
            dpg.add_theme_color(dpg.mvThemeCol_Border, ACCENT)
    _THEMES["highlight_card"] = highlight_theme

    # Bound per-series (not globally) in render_results() so the single
    # winning configuration stands out from the rest of the leaderboard.
    with ui.theme() as best_bar_theme:
        with ui.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, SUCCESS, category=dpg.mvThemeCat_Plots)
    _THEMES["best_bar"] = best_bar_theme

    with ui.theme() as rest_bar_theme:
        with ui.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, ACCENT, category=dpg.mvThemeCat_Plots)
    _THEMES["rest_bar"] = rest_bar_theme

    # Leaderboard table "data bars" -- same best-vs-rest color convention as
    # the bar chart above it (SUCCESS for the winning row, ACCENT for the
    # rest); bar LENGTH (not color) carries the score magnitude.
    with ui.theme() as best_progress_theme:
        with ui.theme_component(dpg.mvProgressBar):
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, SUCCESS)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BG_CARD_ALT)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
    _THEMES["best_progress"] = best_progress_theme

    with ui.theme() as rest_progress_theme:
        with ui.theme_component(dpg.mvProgressBar):
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BG_CARD_ALT)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
    _THEMES["rest_progress"] = rest_progress_theme


def _build_fonts() -> dict[str, ItemId]:
    """Use Segoe UI on Windows and matplotlib's bundled DejaVu fonts elsewhere."""
    fonts: dict[str, ItemId] = {}
    regular = Path(r"C:\Windows\Fonts\segoeui.ttf")
    bold = Path(r"C:\Windows\Fonts\segoeuib.ttf")
    if not regular.is_file():
        font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        regular = font_dir / "DejaVuSans.ttf"
        bold = font_dir / "DejaVuSans-Bold.ttf"
    try:
        with ui.font_registry():
            if regular.exists():
                fonts["body"] = dpg.add_font(str(regular), 17)
            if bold.exists():
                fonts["title"] = dpg.add_font(str(bold), 30)
                fonts["subheader"] = dpg.add_font(str(bold), 19)
        if "body" in fonts:
            dpg.bind_font(fonts["body"])
    except Exception:
        return {}
    return fonts


@contextlib.contextmanager
def _card(title: str, **kwargs: Any) -> Iterator[ItemId]:
    """A bordered, rounded child-window 'card' with an accent-colored title
    and a small accent tick beside it, used throughout to group related
    controls -- the same visual language as the panelled dashboard mockup
    this app's look is modeled on. Touching this one function gives every
    card in the app the same accent treatment."""
    with ui.child_window(auto_resize_y=True, **kwargs) as card_id:
        with ui.group(horizontal=True):
            with ui.drawlist(width=5, height=20):
                dpg.draw_rectangle((0, 1), (5, 19), color=ACCENT, fill=ACCENT, rounding=2)
            dpg.add_spacer(width=4)
            title_id = dpg.add_text(title, color=ACCENT)
            if _FONTS.get("subheader"):
                dpg.bind_item_font(title_id, _FONTS["subheader"])
        dpg.add_separator()
        dpg.add_spacer(height=4)
        yield card_id


def _heading(title: str, subtitle: str) -> None:
    title_id = dpg.add_text(title)
    if "title" in _FONTS:
        dpg.bind_item_font(title_id, _FONTS["title"])
    dpg.add_text(subtitle, color=TEXT_MUTED, wrap=900)
    dpg.add_spacer(height=8)


def _tip(message: str) -> None:
    with ui.tooltip(dpg.last_item()):
        dpg.add_text(message, wrap=380)


def _plot(prefix: str, title: str, y_label: str, height: int = 270) -> None:
    with ui.plot(label=title, height=height, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, tag=f"{prefix}_x", label="Feature / value")
        dpg.add_plot_axis(dpg.mvYAxis, tag=f"{prefix}_y", label=y_label)


def _build_dialogs() -> None:
    with ui.file_dialog(directory_selector=False, show=False, callback=on_test_file_selected,
                        tag="file_dialog_tests", width=800, height=480, file_count=100):
        for suffix in sorted(gc.SUPPORTED_SUFFIXES):
            dpg.add_file_extension(suffix)
    with ui.file_dialog(directory_selector=False, show=False, callback=on_data_file_selected,
                        tag="file_dialog_data", width=800, height=480):
        for extension in (".csv", ".parquet", ".xlsx", ".xls", ".json"):
            dpg.add_file_extension(extension)
    with ui.file_dialog(directory_selector=True, show=False, callback=on_outdir_dir_selected,
                        tag="file_dialog_outdir", width=800, height=480):
        pass
    with ui.file_dialog(directory_selector=False, show=False, callback=on_import_config,
                        tag="file_dialog_config", width=800, height=480):
        dpg.add_file_extension(".json")


def _build_overview() -> None:
    with ui.group(tag="page_overview"):
        _heading("Your next discovery starts here.", "A local workspace for understanding your data, building experiments, and comparing evidence.")
        with _card("WORKSPACE / DATA SNAPSHOT"):
            dpg.add_text("Choose a dataset to populate this workspace.", tag="dataset_summary", color=ACCENT_LIGHT, wrap=900)
            with ui.group(horizontal=True):
                for label, tag, color in (("PREVIEW ROWS", "kpi_rows", TEXT_PRIMARY),
                                          ("COLUMNS", "kpi_columns", TEXT_PRIMARY),
                                          ("MISSING CELLS", "kpi_missing", WARNING),
                                          ("NUMERIC FEATURES", "kpi_numeric", SUCCESS)):
                    with ui.child_window(width=185, height=94, border=False):
                        dpg.add_text(label, color=TEXT_MUTED)
                        item = dpg.add_text("--", tag=tag, color=color)
                        if "title" in _FONTS:
                            dpg.bind_item_font(item, _FONTS["title"])
            dpg.add_text("Profile statistics describe the preview, not an estimate of the full dataset.", color=TEXT_MUTED)
        dpg.add_spacer(height=10)
        with _card("MAKE AN EXPERIMENT"):
            dpg.add_text("01   Explore your data", color=ACCENT_LIGHT)
            dpg.add_text("Inspect rows, missingness, distributions, and relationships before choosing your target.", wrap=850)
            dpg.add_text("02   Design the comparison", color=ACCENT_LIGHT)
            dpg.add_text("Assign column roles, choose a model preset, and control your holdout and tuning budget.", wrap=850)
            dpg.add_text("03   Run and review", color=ACCENT_LIGHT)
            dpg.add_text("Follow the live log, compare holdout scores, and revisit reports from previous experiments.", wrap=850)
            with ui.group(horizontal=True):
                button = dpg.add_button(label="Explore a dataset", callback=on_nav_selected, user_data="data", height=42)
                dpg.bind_item_theme(button, _THEMES["run_button"])
                dpg.add_button(label="Browse results", callback=on_nav_selected, user_data="results", height=42)
                dpg.add_button(label="Load experiment", callback=on_show_dialog, user_data="file_dialog_config", height=42)
        dpg.add_spacer(height=10)
        with _card("EXPERIMENT AT A GLANCE"):
            dpg.add_text("Select a dataset to begin.", tag="experiment_summary", color=ACCENT_LIGHT)
            dpg.add_text("Your data is processed locally. Results and reusable experiment settings stay in the output folder you choose.", color=TEXT_MUTED, wrap=850)


def _build_data_page() -> None:
    with ui.group(tag="page_data", show=False):
        _heading("Data explorer", "Understand your dataset before you train. All charts use a bounded preview.")
        with _card("DATA SOURCE & PREDICTION TARGET"):
            dpg.add_combo(tag="input_mode_combo", label="Input format", items=gc.INPUT_MODES,
                          default_value="Data table", width=280, callback=on_config_changed)
            with ui.group(horizontal=True):
                dpg.add_combo(tag="delimiter_combo", label="CSV delimiter", items=[*gc.DELIMITERS, "Custom"], default_value="Comma", width=160, callback=on_config_changed)
                dpg.add_input_text(tag="delimiter_custom", label="Custom character", default_value=",", width=70, callback=on_config_changed)
            dpg.add_text("Choose format and delimiter before loading. Annotated spreadsheets may contain CLI metadata; explicit app settings take priority.", color=TEXT_MUTED, wrap=850)
            with ui.group(horizontal=True):
                dpg.add_input_text(tag="data_path_display", hint="Paste a dataset path or browse...", width=-190)
                dpg.add_button(label="Load path", callback=on_load_path)
                dpg.add_button(label="Browse", callback=on_show_dialog, user_data="file_dialog_data")
            dpg.add_text("CSV / Parquet / Excel / JSON", color=TEXT_MUTED)
            with ui.collapsing_header(label="Separate test datasets"):
                dpg.add_text("Add one or more labeled test files with the same columns as the training data. One path per line.", wrap=850)
                dpg.add_input_text(tag="test_files_input", multiline=True, height=85, width=-1, callback=on_config_changed)
                dpg.add_button(label="Add test files", callback=on_show_dialog, user_data="file_dialog_tests")
                dpg.add_combo(tag="tests_method_combo", label="Evaluation method", items=["list", "lodo"], default_value="list", width=200, callback=on_config_changed)
                dpg.add_text("list: evaluate the supplied test sets. lodo: leave one dataset out in turn.", wrap=850, color=TEXT_MUTED)
            with ui.group(horizontal=True):
                dpg.add_combo(tag="target_combo", label="Target column", items=[], width=280, callback=on_target_changed)
                dpg.add_radio_button(["classify", "regress"], tag="mode_radio", default_value="classify",
                                     horizontal=True, callback=on_mode_changed)
            dpg.add_text("Classification predicts a category. Regression predicts a continuous value.", color=TEXT_MUTED)
            dpg.add_checkbox(tag="multi_target_checkbox", label="Analyze multiple targets together",
                             default_value=False, callback=on_multi_target_toggle_changed)
            with ui.group(tag="multi_target_group", show=False):
                dpg.add_text("Extra target columns (analyzed together with the target above):", color=TEXT_MUTED)
                with ui.child_window(tag="multi_target_container", height=140, horizontal_scrollbar=True):
                    dpg.add_text("Load a dataset to choose extra targets.", color=TEXT_MUTED)
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="mt_agg_strategy_combo", label="Combine rankings using", items=gc.MT_AGG_STRATEGIES,
                                 default_value=gc.MT_AGG_STRATEGIES[0], width=160, callback=on_config_changed)
                    dpg.add_input_text(tag="mt_top_k_input", label="Keep only top K features (optional)", width=140,
                                       callback=on_config_changed)
                _tip("Multi-target runs analyze every listed target together, combine their feature rankings "
                    "into one shared selection, and produce both an overall report and a separate breakdown "
                    "for each target in the results folder.")
        dpg.add_spacer(height=10)
        with ui.tab_bar():
            with ui.tab(label="Preview"):
                dpg.add_text("Load a dataset to inspect its rows.", tag="preview_note", color=TEXT_MUTED, wrap=850)
                with ui.group(horizontal=True):
                    dpg.add_input_text(tag="preview_search", hint="Search cells in preview...", width=360,
                                       callback=on_preview_filter, on_enter=True)
                    dpg.add_button(label="Search", callback=on_preview_filter)
                    dpg.add_text("", tag="preview_count", color=TEXT_MUTED)
                with ui.child_window(tag="preview_container", height=380, horizontal_scrollbar=True):
                    dpg.add_text("Your data preview will appear here.", color=TEXT_MUTED)
            with ui.tab(label="Column profile"):
                with ui.child_window(tag="profile_container", height=440, horizontal_scrollbar=True):
                    dpg.add_text("Missingness, uniqueness, and data types for every preview column.", color=TEXT_MUTED)
            with ui.tab(label="Distributions"):
                dpg.add_combo(tag="distribution_combo", label="Feature", items=[], width=320, callback=on_distribution_changed)
                _plot("distribution", "Feature distribution", "Count", height=340)
                dpg.add_text("", tag="distribution_note", color=TEXT_MUTED, wrap=850)
            with ui.tab(label="Relationships"):
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="scatter_x_combo", label="X", items=[], width=280, callback=on_scatter_changed)
                    dpg.add_combo(tag="scatter_y_combo", label="Y", items=[], width=280, callback=on_scatter_changed)
                _plot("scatter", "Numeric relationships", "Y", height=380)
                dpg.add_text("Only finite numeric pairs are plotted. Scroll to zoom; drag to pan; double-click to fit.", color=TEXT_MUTED)
            with ui.tab(label="Data quality"):
                dpg.add_text("Load a dataset to see quality checks.", tag="quality_notes", wrap=850)
                _plot("missing", "Most incomplete columns in preview", "Missing %", height=330)


def _build_visualize_page() -> None:
    with ui.group(tag="page_visualize", show=False):
        _heading("Data visualization", "The same bounded preview, viewed through several chart forms -- "
                                       "rendered with matplotlib/seaborn/scikit-learn, the same stack Jupyter uses.")
        with ui.tab_bar(tag="visualize_tab_bar"):
            with ui.tab(label="Correlation heatmap", tag="tab_heatmap"):
                dpg.add_text("", tag="heatmap_note", color=TEXT_MUTED, wrap=850)
                _chart_image("heatmap")
            with ui.tab(label="Histogram", tag="tab_histogram"):
                dpg.add_combo(tag="hist_combo", label="Numeric feature", items=[], width=320, callback=on_histogram_changed)
                _chart_image("histogram")
            with ui.tab(label="Categorical breakdown", tag="tab_categorical"):
                dpg.add_combo(tag="cat_combo", label="Categorical feature", items=[], width=320, callback=on_categorical_changed)
                dpg.add_text("", tag="categorical_note", color=TEXT_MUTED)
                with ui.group(horizontal=True):
                    _chart_image("catbar")
                    _chart_image("catpie")
            with ui.tab(label="Relationships", tag="tab_relationships"):
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="viz_relationship_x_combo", label="X", items=[], width=280, callback=on_viz_relationship_changed)
                    dpg.add_combo(tag="viz_relationship_y_combo", label="Y", items=[], width=280, callback=on_viz_relationship_changed)
                _chart_image("relationship")
                dpg.add_text("Only finite numeric pairs are plotted.", color=TEXT_MUTED)
            with ui.tab(label="Box plot by group", tag="tab_box"):
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="box_value_combo", label="Numeric feature", items=[], width=280, callback=on_box_changed)
                    dpg.add_combo(tag="box_group_combo", label="Group by", items=[], width=280, callback=on_box_changed)
                _chart_image("box")
            with ui.tab(label="Trend", tag="tab_trend"):
                dpg.add_combo(tag="trend_combo", label="Numeric feature", items=[], width=320, callback=on_trend_changed)
                _chart_image("trend")
                dpg.add_text("Row order reflects file order, not necessarily time.", color=TEXT_MUTED)
            with ui.tab(label="PCA projection", tag="tab_pca"):
                dpg.add_combo(tag="pca_color_combo", label="Color by", items=[], width=280, callback=on_pca_changed)
                dpg.add_text("", tag="pca_note", color=TEXT_MUTED, wrap=850)
                _chart_image("pca")
                dpg.add_text("scikit-learn's PCA reduces all numeric columns to the two directions of greatest "
                            "variance -- points that cluster together look similar across those columns combined.",
                            color=TEXT_MUTED, wrap=850)


def _build_prepare_page() -> None:
    with ui.group(tag="page_prepare", show=False):
        _heading("Preprocessing workspace", "Tell the pipeline what each column means, clean and reshape your data, and keep leakage out of your experiment.")
        with ui.tab_bar():
            with ui.tab(label="Column roles"):
                with _card("COLUMN ROLES"):
                    dpg.add_text("Automatic detection is the default. Override only when you know the column's meaning.", color=TEXT_MUTED, wrap=850)
                    with ui.group(horizontal=True):
                        dpg.add_combo(tag="role_column_combo", items=[], label="Column", width=260, callback=on_role_column_changed)
                        dpg.add_combo(tag="role_value_combo", items=ROLE_OPTIONS, default_value="Automatic", label="Role", width=180)
                        dpg.add_button(label="Apply role", callback=on_apply_role)
                    with ui.child_window(tag="roles_container", height=260, horizontal_scrollbar=True):
                        dpg.add_text("Load a dataset to assign column roles.", color=TEXT_MUTED)
                    dpg.add_combo(tag="grouper_combo", label="Group column (optional)", items=[""], default_value="", width=300, callback=on_group_changed)
                    _tip("Use a patient, subject, or other group ID so observations from the same group are kept together in splits. Empty means no group column.")
                    dpg.add_text("Categorical = labels. Ordinal = ordered numeric values. Exclude = remove from the model.", color=TEXT_MUTED, wrap=850)
                    dpg.add_text("Group and target columns use their own controls. Exclude IDs and measurements unavailable at prediction time.", color=TEXT_MUTED, wrap=850)
            with ui.tab(label="Missing values & scaling"):
                with _card("MISSING VALUES & SCALING"):
                    with ui.group(horizontal=True):
                        dpg.add_combo(tag="nan_combo", label="Missing numeric values", items=gc.NAN_METHODS,
                                     default_value=gc.NAN_METHODS[1], width=200, callback=on_config_changed)
                        dpg.add_combo(tag="norm_combo", label="Numeric scaling", items=gc.NORM_METHODS,
                                     default_value=gc.NORM_METHODS[0], width=200, callback=on_config_changed)
                    _tip("Mean/median: replace missing values with that column's mean/median. Impute: predict each "
                        "missing value from the other features (slower, often more accurate). Robust scaling clips "
                        "extreme outliers before rescaling; min-max does not.")
                    dpg.add_text("Categorical missing values are always kept as their own category -- this only "
                                "affects numeric columns.", color=TEXT_MUTED, wrap=850)
                    dpg.add_text("Column inspection, categorical encoding, and invalid-feature cleanup run automatically.",
                                color=TEXT_MUTED, wrap=850)
                    dpg.add_text("Ordinal columns must already use a meaningful numeric ordering. This workspace does not invent a category order.", color=TEXT_MUTED, wrap=850)
            with ui.tab(label="Data preparation"):
                with _card("QUICK CLEANUP"):
                    with ui.group(horizontal=True):
                        dpg.add_checkbox(tag="drop_duplicates_checkbox", label="Drop duplicate rows before analysis",
                                         default_value=False, callback=on_config_changed)
                        dpg.add_input_int(tag="preview_limit_input", label="Preview row limit", default_value=_preview_limit,
                                          min_value=100, max_value=20000, step=100, width=160, callback=on_preview_limit_changed)
                    _tip("Duplicate rows are removed from the working copy of the full dataset before splitting and "
                        "training -- not just from this preview. The preview row limit only affects exploration speed "
                        "here; the analysis itself always uses the complete dataset.")
                dpg.add_spacer(height=10)
                with _card("ADD A PREPARATION STEP"):
                    dpg.add_text(
                        "Build a reusable, step-by-step recipe: pick an action, fill in its details, and add "
                        "it. Each step runs on the result of the one before it, against your COMPLETE dataset "
                        "(not just this preview) right before the analysis runs. No code or formulas to type "
                        "anywhere here -- every action is a dropdown, checkbox, or plain value.",
                        color=TEXT_MUTED, wrap=850,
                    )
                    dpg.add_combo(
                        tag="prep_op_combo", label="Action",
                        items=[spec.label for spec in gc.TRANSFORM_OPS.values()],
                        default_value=next(iter(gc.TRANSFORM_OPS.values())).label,
                        width=320, callback=on_prep_op_changed,
                    )
                    dpg.add_spacer(height=4)

                    with ui.group(tag="prep_params_rename_column", show=True):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_rename_column_combo", items=[], label="Column", width=220)
                            dpg.add_input_text(tag="prep_rename_new_name_input", label="New name", width=220)

                    with ui.group(tag="prep_params_cast_column_dtype", show=False):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_cast_column_combo", items=[], label="Column", width=220)
                            dpg.add_combo(tag="prep_cast_dtype_combo", items=gc.CAST_DTYPES,
                                         default_value=gc.CAST_DTYPES[0], label="New data type", width=180)

                    with ui.group(tag="prep_params_drop_columns", show=False):
                        dpg.add_combo(tag="prep_drop_column_combo", items=[], label="Column to drop", width=260)

                    with ui.group(tag="prep_params_clean_text_column", show=False):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_text_column_combo", items=[], label="Column", width=220)
                            dpg.add_combo(tag="prep_text_mode_combo", items=gc.TEXT_CLEAN_MODES,
                                         default_value=gc.TEXT_CLEAN_MODES[0], label="Cleanup", width=200)

                    with ui.group(tag="prep_params_extract_datetime_parts", show=False):
                        dpg.add_combo(tag="prep_dt_column_combo", items=[], label="Date/time column", width=260)
                        dpg.add_text("Parts to extract as new columns:", color=TEXT_MUTED)
                        with ui.group(horizontal=True):
                            for part in gc.DATETIME_PARTS:
                                dpg.add_checkbox(label=part, tag=f"prep_dt_part_{part}")

                    with ui.group(tag="prep_params_filter_rows", show=False):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_filter_column_combo", items=[], label="Column", width=200)
                            dpg.add_combo(tag="prep_filter_operator_combo", items=gc.FILTER_OPERATORS,
                                         default_value=gc.FILTER_OPERATORS[0], label="Comparison", width=170)
                            dpg.add_input_text(tag="prep_filter_value_input", label="Value", width=140)
                        dpg.add_radio_button(["Keep", "Remove"], tag="prep_filter_keep_radio", default_value="Keep", horizontal=True)
                        _tip("Value is ignored for 'is missing'/'is not missing'. Text vs. number comparison is detected automatically from the column.")

                    with ui.group(tag="prep_params_dedupe_rows", show=False):
                        dpg.add_combo(tag="prep_dedupe_keep_combo", items=gc.DEDUPE_KEEP,
                                     default_value=gc.DEDUPE_KEEP[0], label="Which copy to keep", width=200)
                        dpg.add_text("Compares all columns. For a specific subset, exclude irrelevant columns via column roles first.", color=TEXT_MUTED, wrap=850)

                    with ui.group(tag="prep_params_drop_missing_rows", show=False):
                        dpg.add_combo(tag="prep_dropna_column_combo", items=["(any column)"],
                                     default_value="(any column)", label="Require this column present", width=260)

                    with ui.group(tag="prep_params_fill_missing_column", show=False):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_fill_column_combo", items=[], label="Column", width=200)
                            dpg.add_combo(tag="prep_fill_strategy_combo", items=gc.FILL_STRATEGIES,
                                         default_value=gc.FILL_STRATEGIES[0], label="Fill with", width=180,
                                         callback=on_prep_fill_strategy_changed)
                        with ui.group(tag="prep_fill_custom_group", show=False):
                            dpg.add_input_text(tag="prep_fill_custom_input", label="Custom value", width=200)

                    with ui.group(tag="prep_params_flag_missing_column", show=False):
                        dpg.add_combo(tag="prep_flag_column_combo", items=[], label="Column", width=260)
                        dpg.add_text("Adds a new 0/1 column marking which rows were missing this value.", color=TEXT_MUTED, wrap=850)

                    with ui.group(tag="prep_params_clip_column_outliers", show=False):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_clip_column_combo", items=[], label="Column", width=200)
                            dpg.add_combo(tag="prep_clip_method_combo", items=gc.CLIP_METHODS,
                                         default_value=gc.CLIP_METHODS[0], label="Method", width=180)
                        with ui.group(horizontal=True):
                            dpg.add_input_float(tag="prep_clip_low_input", label="Low", default_value=1.0, width=140)
                            dpg.add_input_float(tag="prep_clip_high_input", label="High", default_value=99.0, width=140)
                        _tip("Percentile: 0-100 (e.g. 1 and 99). Std devs from mean: how many standard deviations "
                            "below/above the mean to allow. Manual bounds: exact values.")

                    with ui.group(tag="prep_params_threshold_flag_column", show=False):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_threshold_column_combo", items=[], label="Column", width=200)
                            dpg.add_combo(tag="prep_threshold_operator_combo", items=gc.THRESHOLD_OPERATORS,
                                         default_value=gc.THRESHOLD_OPERATORS[0], label="Comparison", width=140)
                            dpg.add_input_float(tag="prep_threshold_value_input", label="Threshold", width=140)
                        dpg.add_input_text(tag="prep_threshold_new_column_input", label="New column name", width=220)

                    with ui.group(tag="prep_params_combine_columns", show=False):
                        with ui.group(horizontal=True):
                            dpg.add_combo(tag="prep_combine_a_combo", items=[], label="Column A", width=180)
                            dpg.add_combo(tag="prep_combine_operator_combo", items=gc.COMBINE_OPERATORS,
                                         default_value=gc.COMBINE_OPERATORS[0], label="Operator", width=100)
                            dpg.add_combo(tag="prep_combine_b_combo", items=[], label="Column B", width=180)
                        dpg.add_input_text(tag="prep_combine_new_column_input", label="New column name", width=220)

                    dpg.add_spacer(height=6)
                    dpg.add_button(label="Add step", callback=on_add_prep_step)
                    dpg.add_text("", tag="prep_error_text", color=WARNING, wrap=850)
                dpg.add_spacer(height=10)
                with _card("PREPARATION STEPS"):
                    dpg.add_text("", tag="prep_summary_text", color=TEXT_MUTED, wrap=850)
                    with ui.child_window(tag="pipeline_table_container", height=180, horizontal_scrollbar=True):
                        dpg.add_text("No preparation steps added yet.", color=TEXT_MUTED)
                    dpg.add_button(label="Clear all steps", callback=on_clear_pipeline)
                    dpg.add_spacer(height=8)
                    dpg.add_text("Preview after preparation:", color=TEXT_MUTED)
                    with ui.child_window(tag="prep_sample_container", height=220, horizontal_scrollbar=True):
                        dpg.add_text("Load a dataset to preview preparation results.", color=TEXT_MUTED)


def _build_models_page() -> None:
    with ui.group(tag="page_models", show=False):
        _heading("Model studio", "Build a useful comparison, from a fast baseline to a broader benchmark.")
        with _card("TASK"):
            dpg.add_radio_button(["classify", "regress"], tag="mode_radio_studio", default_value="classify",
                                 horizontal=True, callback=on_mode_changed)
            dpg.add_text("Classification predicts a category. Regression predicts a continuous value. "
                         "This is the same setting as Data explorer.", color=TEXT_MUTED, wrap=850)
        dpg.add_spacer(height=10)
        with _card("EXPERIMENT PRESET"):
            with ui.group(horizontal=True):
                dpg.add_combo(tag="preset_combo", items=list(gc.PRESET_NAMES), default_value="Balanced", width=260)
                dpg.add_button(label="Apply preset", callback=on_preset_selected)
            dpg.add_text("Quick baseline reduces the search budget. Thorough explores more models and takes longer.", color=TEXT_MUTED, wrap=850)
        dpg.add_spacer(height=10)
        with _card("MODEL LIBRARY"):
            with ui.group(horizontal=True):
                dpg.add_button(label="Select all", callback=on_select_models, user_data=True)
                dpg.add_button(label="Clear selection", callback=on_select_models, user_data=False)
            for prefix, names, defaults in (("clf", gc.CLASSIFIERS, gc.DEFAULT_CLASSIFIERS),
                                             ("reg", gc.REGRESSORS, gc.DEFAULT_REGRESSORS)):
                with ui.group(tag=f"{prefix}_group", horizontal=True, show=prefix == "clf"):
                    midpoint = (len(names) + 1) // 2
                    for start in (0, midpoint):
                        with ui.group(width=330):
                            for name in names[start:start + midpoint]:
                                dpg.add_checkbox(label=f"{MODEL_LABELS.get(name, name)}  ({name})", tag=f"{prefix}_{name}",
                                                 default_value=name in defaults, callback=on_config_changed)
            dpg.add_combo(tag="metric_combo", label="Optimize for", items=gc.CLS_METRICS,
                          default_value=gc.default_metric(gc.CLS_METRICS), width=250, callback=on_config_changed)
            _tip("This metric guides hyperparameter tuning. The results library also exposes other metrics reported by the pipeline.")
        dpg.add_spacer(height=10)
        with _card("FEATURE SELECTION"):
            with ui.group(horizontal=True):
                for method, label in (("none", "All features"), ("filter", "Statistical filter"),
                                      ("embed", "Model importance"), ("wrap", "Wrapper search")):
                    dpg.add_checkbox(label=label, tag=f"feat_{method}", default_value=method == "filter", callback=on_feat_select_changed)
            with ui.group(tag="filter_group", show=True):
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="filter_method_combo", label="Filter basis", items=gc.FILTER_METHODS,
                                 default_value=gc.FILTER_METHODS[0], width=180, callback=on_config_changed)
                    dpg.add_input_text(tag="n_feat_filter_input", label="Features to keep (count or fraction)",
                                       default_value="0.5", width=100, callback=on_config_changed)
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="filter_assoc_cont_combo", label="Continuous-feature statistic",
                                 items=gc.CONT_CLS_STATS, default_value="mut_info", width=180, callback=on_config_changed)
                    dpg.add_combo(tag="filter_assoc_cat_combo", label="Categorical-feature statistic",
                                 items=gc.CAT_CLS_STATS, default_value="mut_info", width=180, callback=on_config_changed)
                    dpg.add_combo(tag="filter_pred_combo", label="Prediction filter score",
                                 items=gc.CLS_PRED_SCORES, default_value="acc", width=140, callback=on_config_changed)
                _tip("'Association' ranks features by a univariate statistic with the target; 'prediction' "
                    "ranks by a quick per-feature predictive fit (slower, but accounts for predictive "
                    "usefulness a raw association can miss). Continuous/categorical statistic and score only "
                    "apply to their respective column types and matching filter basis.")
                dpg.add_text("Enter a whole number (e.g. 20) for an exact feature count, or a fraction "
                            "like 0.5 for 50% of all features.", color=TEXT_MUTED, wrap=850)
            with ui.group(tag="embed_group", show=False):
                dpg.add_combo(tag="embed_model_combo", label="Importance model", items=gc.EMBED_MODELS,
                              default_value=gc.EMBED_MODELS[0], width=220, callback=on_config_changed)
            with ui.group(tag="wrapper_group", show=False):
                dpg.add_combo(tag="wrapper_method_combo", label="Search direction", items=gc.WRAPPER_METHODS,
                              default_value=gc.WRAPPER_METHODS[0], width=220, callback=on_config_changed)
                dpg.add_combo(tag="wrapper_model_combo", label="Wrapper model", items=gc.WRAPPER_MODELS,
                              default_value=gc.WRAPPER_MODELS[0], width=220, callback=on_config_changed)
                dpg.add_input_text(tag="n_feat_wrapper_input", label="Features to keep (count or fraction)",
                                   default_value="20", width=150, callback=on_config_changed)
                dpg.add_checkbox(tag="redundant_wrapper_checkbox", label="Extra-greedy redundancy-aware selection",
                                 default_value=False, callback=on_redundant_toggle_changed)
                with ui.group(tag="redundant_group", show=False):
                    dpg.add_input_float(tag="redundant_threshold_input", label="Score-tie threshold",
                                        default_value=0.005, min_value=0.0, min_clamped=True,
                                        format="%.4f", width=150, callback=on_config_changed)
                    dpg.add_input_float(tag="redundant_corr_threshold_input", label="Correlation threshold",
                                        default_value=0.8, min_value=0.0, max_value=1.0,
                                        min_clamped=True, max_clamped=True, format="%.2f", width=150,
                                        callback=on_config_changed)
                    _tip("At each step, also add or remove other features whose scores are within the "
                        "tie threshold of the best AND are correlated with it above the correlation "
                        "threshold -- can find a smaller, less redundant feature set, at extra compute cost.")
                dpg.add_text("Wrapper selection may fit many additional models. Start with a small experiment.", color=WARNING, wrap=850)


def _build_tuning_page() -> None:
    with ui.group(tag="page_tuning", show=False):
        _heading("Experiment settings", "Choose your compute budget, evaluation split, and a reproducible configuration.")
        with _card("EVALUATION & TUNING"):
            dpg.add_input_int(tag="htune_trials_input", label="Tuning trials", default_value=100,
                              min_value=1, min_clamped=True, width=220, callback=on_config_changed)
            dpg.add_slider_float(tag="test_val_size_slider", label="Holdout fraction", default_value=0.4,
                                 min_value=0.05, max_value=0.5, format="%.2f", width=340, callback=on_config_changed)
            dpg.add_input_int(tag="seed_input", label="Random seed (0 = CLI default)", default_value=0,
                              min_value=0, min_clamped=True, width=220, callback=on_config_changed)
            dpg.add_checkbox(tag="no_preds_checkbox", label="Skip univariate prediction analysis to save time", default_value=False, callback=on_config_changed)
            _tip("Skips the separate per-feature prediction analysis. Your selected final models are still trained and evaluated.")
            dpg.add_checkbox(tag="adaptive_error_checkbox", label="Use adaptive error scoring", default_value=False, callback=on_config_changed)
        dpg.add_spacer(height=10)
        with _card("OUTPUT & REUSABLE EXPERIMENTS"):
            with ui.group(horizontal=True):
                dpg.add_input_text(tag="outdir_input", default_value=str(gc.DEFAULT_OUTDIR), width=-100, callback=on_config_changed, on_enter=True)
                dpg.add_button(label="Browse", callback=on_show_dialog, user_data="file_dialog_outdir")
            with ui.group(horizontal=True):
                dpg.add_button(label="Save experiment JSON", callback=on_export_config)
                dpg.add_button(label="Load experiment JSON", callback=on_show_dialog, user_data="file_dialog_config")
                dpg.add_button(label="Open output folder", callback=on_open_output)
            dpg.add_text("Configuration files contain paths and settings; they do not copy the dataset.", color=TEXT_MUTED)
            dpg.add_text("", tag="settings_notice", color=ACCENT_LIGHT, wrap=850)
        dpg.add_spacer(height=10)
        with _card("ADVANCED EXPERIMENT CONTROLS"):
            advanced_settings.build(on_config_changed)
        with _card("CLI COMPATIBILITY"):
            dpg.add_text(f"df-analyze {VERSION}", color=ACCENT_LIGHT)
            dpg.add_text("Help: load a dataset in Data explorer, choose models and selection methods, configure Experiment settings, then run. Advanced controls explain each option when enabled. Reports and downloadable artifacts are in Results library.", wrap=850)
            dpg.add_text("SVM and Relief selection are disabled by the analysis engine. AUROC tuning falls back to balanced accuracy; use the results AUROC metric for evaluation. Missing-value row dropping is available in Data preparation.", wrap=850, color=TEXT_MUTED)
            dpg.add_text("Legacy imported CLI arguments (optional). All supported analysis switches now have form controls.", wrap=850)
            dpg.add_input_text(tag="extra_args_input", hint="--n-feat-wrapper 15", width=-1, callback=on_config_changed, on_enter=True)
            dpg.add_text("For additional CLI options. Conflicting values are checked before a run starts.", color=TEXT_MUTED, wrap=850)


def _build_run_page() -> None:
    with ui.group(tag="page_run", show=False):
        _heading("Run monitor", "A clear view of your active experiment, with its exact command and live output.")
        with _card("EXPERIMENT READINESS"):
            dpg.add_text("Load a dataset to configure an experiment.", tag="validation_text", color=WARNING, wrap=850)
            with ui.group(horizontal=True):
                button = dpg.add_button(tag="run_button", label="Run analysis", callback=on_run_clicked, height=44)
                dpg.bind_item_theme(button, _THEMES["run_button"])
                dpg.add_button(tag="cancel_button", label="Stop analysis", callback=on_cancel_clicked, enabled=False, height=44)
                dpg.add_button(label="Validate settings", callback=on_config_changed, height=44)
                dpg.add_text("Elapsed  00:00:00", tag="elapsed_text", color=TEXT_MUTED)
            dpg.add_text("", tag="cancel_note", color=TEXT_MUTED, wrap=850)
            dpg.add_text("Ready when you are.", tag="run_stage", color=ACCENT_LIGHT, wrap=850)
            with ui.collapsing_header(label="Reproducible command"):
                dpg.add_input_text(tag="command_text", readonly=True, multiline=True, width=-1, height=95)
                dpg.add_button(label="Copy command", callback=on_copy_command)
        dpg.add_spacer(height=10)
        with _card("LIVE OUTPUT"):
            with ui.group(horizontal=True):
                dpg.add_checkbox(tag="follow_log_checkbox", label="Follow output", default_value=True)
                dpg.add_button(label="Save log", callback=on_save_log)
            with ui.child_window(tag="log_window", height=320, horizontal_scrollbar=True):
                dpg.add_text("", tag="log_box", wrap=0)
            dpg.add_text("Training time depends on rows, features, models and tuning trials. No estimated percentage is shown.", color=TEXT_MUTED, wrap=850)


def _build_results_page() -> None:
    with ui.group(tag="page_results", show=False):
        _heading("Results library", "Compare completed experiments and inspect the evidence behind each score.")
        with _card("EXPERIMENT HISTORY"):
            with ui.group(horizontal=True):
                dpg.add_button(label="Refresh library", callback=on_refresh_results)
                dpg.add_button(label="Open results folder", callback=on_open_output)
                dpg.add_text("0 completed result sets", tag="library_count", color=TEXT_MUTED)
            dpg.add_combo(tag="run_history_combo", items=[], width=-1, callback=on_history_changed)
            dpg.add_text("Refresh to find completed experiments in your output folder.", tag="results_status_text", color=TEXT_MUTED, wrap=850)
        dpg.add_spacer(height=10)
        with _card("MODEL LEADERBOARD"):
            with ui.child_window(height=175, border=True) as best_card:
                dpg.bind_item_theme(best_card, _THEMES["highlight_card"])
                with ui.group(horizontal=True):
                    with ui.group():
                        dpg.add_text("BEST MODEL", color=TEXT_MUTED)
                        best_model_text = dpg.add_text("--", tag="results_best_model", color=ACCENT_LIGHT)
                        if "subheader" in _FONTS:
                            dpg.bind_item_font(best_model_text, _FONTS["subheader"])
                    dpg.add_spacer(width=50)
                    with ui.group():
                        dpg.add_text("HOLDOUT SCORE", color=TEXT_MUTED)
                        best_value_text = dpg.add_text("--", tag="results_best_value", color=ACCENT_LIGHT)
                        if "subheader" in _FONTS:
                            dpg.bind_item_font(best_value_text, _FONTS["subheader"])
                dpg.add_text("Complete an experiment to see its holdout comparison.", tag="results_best_text", color=TEXT_MUTED, wrap=850)
                dpg.add_text("", tag="results_baseline_note", color=TEXT_MUTED, wrap=850)
            with ui.group(horizontal=True):
                dpg.add_combo(tag="results_metric_combo", label="Metric", items=[], width=170, callback=on_results_metric_changed)
                dpg.add_input_text(tag="result_search", hint="Filter model / selection", width=260, callback=on_results_metric_changed, on_enter=True)
                dpg.add_button(label="Export scores CSV", callback=on_export_scores)
            with ui.group(tag="results_target_group", horizontal=True, show=False):
                dpg.add_combo(tag="results_target_combo", label="Target", items=[], width=220, callback=on_results_target_changed)
                dpg.add_text("This run analyzed multiple targets together. Per-target CSV/Markdown files are in the Artifacts tab.", color=TEXT_MUTED, wrap=600)
            with ui.plot(label="Top 15 configurations by holdout score", height=290, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, tag="results_x_axis", label="Model / feature selection")
                dpg.add_plot_axis(dpg.mvYAxis, tag="results_y_axis", label="Holdout score")
            with ui.child_window(tag="results_table_container", height=235, horizontal_scrollbar=True):
                pass
        dpg.add_spacer(height=10)
        with ui.tab_bar():
            with ui.tab(label="Results report"):
                dpg.add_input_text(tag="results_report_text", readonly=True, multiline=True, width=-1, height=320)
            with ui.tab(label="Artifacts"):
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="artifact_combo", items=[], width=-160, callback=on_artifact_changed)
                    dpg.add_button(label="Open artifact", callback=on_open_artifact)
                dpg.add_text("Text previews are limited to 60,000 characters. Open the original for the complete file.", color=TEXT_MUTED)
                dpg.add_input_text(tag="artifact_preview", readonly=True, multiline=True, width=-1, height=320)


def build_ui() -> None:
    global _notebook
    """Construct widgets without opening a viewport, also used by headless tests."""
    global _FONTS, _workflow
    _build_theme()
    _FONTS = _build_fonts()
    _build_dialogs()
    _build_chart_textures()
    with ui.window(tag="main_window", label="df-analyze | Research workspace"):
        with ui.child_window(height=90, border=False):
            with ui.group(horizontal=True):
                with ui.drawlist(width=58, height=58):
                    dpg.draw_rectangle((3, 31), (14, 52), color=ACCENT, fill=ACCENT, rounding=3)
                    dpg.draw_rectangle((21, 19), (32, 52), color=ACCENT_HOVER, fill=ACCENT_HOVER, rounding=3)
                    dpg.draw_rectangle((39, 5), (50, 52), color=ACCENT_LIGHT, fill=ACCENT_LIGHT, rounding=3)
                dpg.add_spacer(width=6)
                with ui.group():
                    brand = dpg.add_text("df-analyze", color=TEXT_PRIMARY)
                    if "title" in _FONTS:
                        dpg.bind_item_font(brand, _FONTS["title"])
                    dpg.add_text("DISCOVER.  UNDERSTAND.  PREDICT.", color=ACCENT_LIGHT)
        dpg.add_separator()
        with ui.group(horizontal=True):
            with ui.child_window(width=235, height=-1) as sidebar:
                dpg.bind_item_theme(sidebar, _THEMES["sidebar"])
                dpg.add_text("RESEARCH WORKSPACE", color=TEXT_MUTED)
                dpg.add_spacer(height=12)
                for key, label in NAV_ITEMS:
                    dpg.add_selectable(label=label, tag=f"nav_{key}", default_value=key == "overview",
                                       callback=on_nav_selected, user_data=key, height=36)
                dpg.add_spacer(height=25)
                dpg.add_separator()
                dpg.add_text("SESSION", color=TEXT_MUTED)
                with ui.group(horizontal=True):
                    with ui.drawlist(width=12, height=16):
                        dpg.draw_circle((6, 8), 4, tag="status_pulse_dot", color=(0, 0, 0, 0), fill=(0, 0, 0, 0))
                    dpg.add_text("Ready", tag="status_text", color=SUCCESS, wrap=170)
                dpg.add_loading_indicator(tag="run_loading_indicator", show=False, radius=2.4,
                                          color=ACCENT, secondary_color=ACCENT_LIGHT)
                dpg.add_spacer(height=20)
                dpg.add_text("Local data.\nReproducible experiments.\nEvidence you can inspect.", color=TEXT_MUTED, wrap=190)
            with ui.child_window(tag="workspace_content", width=-1, height=-1, border=False):
                _build_overview()
                _build_data_page()
                _build_visualize_page()
                _build_prepare_page()
                _build_models_page()
                _build_tuning_page()
                _build_run_page()
                _build_results_page()
                embedding_panel.build(lambda: _current_run is not None, _load_data)
                _notebook = NotebookPanel(ROOT, _build_run_config)
                _notebook.build()
                _workflow = WorkflowCanvas(_canvas_config, on_config_changed,
                                           lambda page: on_nav_selected(user_data=page))
                _workflow.build()
    on_config_changed()


def run_gui() -> None:
    dpg.create_context()
    try:
        # Keep callbacks and rendering on one thread; only analysis runs in the worker.
        dpg.configure_app(manual_callback_management=True)
        build_ui()
        dpg.create_viewport(title="df-analyze | Research workspace", width=1440, height=980,
                            min_width=1120, min_height=760)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            dpg.run_callbacks(dpg.get_callback_queue())
            poll_run_state()
            embedding_panel.poll()
            if _notebook is not None:
                _notebook.poll()
            _tick_animations()
            _update_status_pulse()
            dpg.render_dearpygui_frame()
    finally:
        if _current_run is not None and _current_run.can_cancel:
            _current_run.cancel()
        embedding_panel.cancel()
        if _notebook is not None:
            _notebook.session.close()
        dpg.destroy_context()


def _self_test_charts(data_path: Path) -> bool:
    """Exercise the matplotlib/seaborn/sklearn -> DPG-texture chart pipeline
    through a real (briefly shown) viewport.

    This needs an actual render loop, not just a headless build_ui() call:
    the texture-publish path has a documented history of crashing hard at
    the C level (a raw buffer-size mismatch) only once a chart's texture is
    actually rendered, not when it's merely computed -- so a frozen build
    with no interactive mouse/window has no other way to catch that class of
    bug before a real user hits it.
    """
    dpg.create_context()
    try:
        build_ui()
        dpg.create_viewport(title="df-analyze self-test", width=1440, height=980)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        for _ in range(5):
            dpg.render_dearpygui_frame()
        if not _load_data(data_path):
            print("[self-test] FAILED: could not load the dataset for chart rendering")
            return False
        on_nav_selected(user_data="visualize")
        for tab in ("tab_heatmap", "tab_histogram", "tab_categorical",
                    "tab_relationships", "tab_box", "tab_trend", "tab_pca"):
            dpg.set_value("visualize_tab_bar", tab)
            for _ in range(5):
                dpg.render_dearpygui_frame()
        return True
    finally:
        dpg.destroy_context()


def run_self_test(data_path: Path, target: str, outdir: Path) -> int:
    """Exercise chart rendering and the analysis runner, without requiring
    any mouse/window interaction."""
    print("[self-test] Building UI and exercising chart rendering...")
    if not _self_test_charts(data_path):
        return 1
    print("[self-test] Chart rendering OK")

    from desktop.smoke_checks import check_data_and_models

    check_data_and_models(outdir / "library-checks")

    print("[self-test] Running a real analysis...")
    cfg = gc.RunConfig(data_path=data_path, target=target, mode="classify",
                        models=["dummy"], feat_select=["none"], htune_trials=3, outdir=outdir,
                        no_preds=True)
    handle = start_run(cfg)
    handle.thread.join()
    chunks: list[str] = []
    while True:
        try:
            chunks.append(handle.log_queue.get_nowait())
        except queue.Empty:
            break
    encoding = sys.stdout.encoding or "utf-8"
    print("".join(chunks).encode(encoding, errors="replace").decode(encoding))
    status, message = handle.done_queue.get_nowait()
    if status != "success":
        print(f"[self-test] FAILED: {message}")
        return 1
    results_dir = gc.find_latest_run(outdir)
    if results_dir is None:
        print(f"[self-test] FAILED: no completed results under {outdir}")
        return 1
    print("[self-test] Executing notebook cells with the app's Python...")
    from df_analyze.gui_notebook import NotebookSession, cell

    session = NotebookSession(ROOT)
    session.working_dir = outdir / "notebook"
    session.cells = [
        cell("import numpy as np\nimport pandas as pd\nanswer = int(np.arange(5).sum()) + 30\npd.DataFrame({'value': [answer]})"),
        cell("answer + 2"),
        cell("%matplotlib inline\nimport matplotlib.pyplot as plt\nplt.plot([1, 2, 3]); plt.show()"),
    ]
    try:
        session.execute()
        session.thread.join(timeout=120)
        if session.busy or session.status != "Execution complete.":
            print(f"[self-test] FAILED: {session.status}")
            return 1
        outputs = [o for c in session.cells for o in c.get("outputs", [])]
        if any(o.get("output_type") == "error" for o in outputs):
            print(f"[self-test] FAILED: notebook errors: {outputs}")
            return 1
        if not any(o.get("data", {}).get("text/plain") == "42" for o in outputs):
            print("[self-test] FAILED: notebook did not preserve its variables")
            return 1
        if not any("image/png" in o.get("data", {}) for o in outputs):
            print("[self-test] FAILED: notebook did not render its plot")
            return 1
        (outdir / "notebook-self-test.ipynb").write_text(session.to_ipynb(), encoding="utf-8")
    finally:
        session.close()
    print("[self-test] Notebook execution, plotting and export OK")
    print(f"[self-test] PASSED: {results_dir}")
    return 0


def main() -> None:
    multiprocessing.freeze_support()
    if "--self-test" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--self-test", action="store_true")
        parser.add_argument("--data", required=True, type=Path)
        parser.add_argument("--target", default="target")
        parser.add_argument("--outdir", required=True, type=Path)
        args = parser.parse_args()
        sys.exit(run_self_test(args.data, args.target, args.outdir))
    run_gui()


if __name__ == "__main__":
    main()
