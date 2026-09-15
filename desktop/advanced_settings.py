"""Point-and-click advanced settings, shared CLI schema, no ML imports."""
from __future__ import annotations

from typing import Any, Callable

import dearpygui.dearpygui as dpg

from desktop import dpg_context as ui
from df_analyze.gui_options import OPTIONS


def tag(flag: str) -> str:
    return "option_" + flag.removeprefix("--").replace("-", "_")


def read() -> dict[str, Any]:
    result = {}
    for spec in OPTIONS:
        key = tag(spec.flag)
        if dpg.does_item_exist(key + "_enabled") and dpg.get_value(key + "_enabled"):
            result[spec.flag] = (
                [choice for i, choice in enumerate(spec.choices) if dpg.get_value(f"{key}_{i}")]
                if spec.kind == "multi" else dpg.get_value(key)
            )
    return result


def restore(values: dict[str, Any]) -> None:
    for spec in OPTIONS:
        key = tag(spec.flag)
        active = spec.flag in values
        dpg.set_value(key + "_enabled", active)
        dpg.configure_item(key + "_controls", show=active)
        value = values.get(spec.flag, spec.default)
        if spec.kind == "multi":
            for i, choice in enumerate(spec.choices):
                dpg.set_value(f"{key}_{i}", choice in value)
        else:
            dpg.set_value(key, value)


def build(on_change: Callable) -> None:
    dpg.add_text("Enable a setting to override its default. Disabled settings do not change your experiment.", wrap=850)

    def toggle(sender, value, key):
        dpg.configure_item(key + "_controls", show=bool(value))
        on_change()

    def reset():
        restore({})
        on_change()

    dpg.add_button(label="Reset advanced settings", callback=reset)
    for group in dict.fromkeys(spec.group for spec in OPTIONS):
        with ui.collapsing_header(label=group):
            if group.startswith("Adaptive"):
                dpg.add_text("Requires adaptive error scoring. Ensemble parameters also require ensemble analysis.", wrap=850)
            for spec in (s for s in OPTIONS if s.group == group):
                key = tag(spec.flag)
                dpg.add_checkbox(label=spec.label, tag=key + "_enabled", callback=toggle, user_data=key)
                with ui.group(tag=key + "_controls", show=False, indent=20):
                    dpg.add_text(spec.help, wrap=800)
                    args = dict(tag=key, default_value=spec.default, callback=on_change, width=300)
                    if spec.kind == "multi":
                        for i, choice in enumerate(spec.choices):
                            dpg.add_checkbox(label=choice.replace("_", " "), tag=f"{key}_{i}", default_value=choice in spec.default, callback=on_change)
                    elif spec.kind == "bool":
                        args.pop("width")
                        dpg.add_checkbox(label="On", **args)
                    elif spec.kind == "choice":
                        dpg.add_combo(items=list(spec.choices), **args)
                    elif spec.kind in ("int", "float"):
                        widget = dpg.add_input_int if spec.kind == "int" else dpg.add_input_double
                        bounds = {}
                        convert = int if spec.kind == "int" else float
                        if spec.minimum is not None:
                            bounds.update(min_value=convert(spec.minimum), min_clamped=True)
                        if spec.maximum is not None:
                            bounds.update(max_value=convert(spec.maximum), max_clamped=True)
                        widget(**args, **bounds)
                    else:
                        dpg.add_input_text(**args)
