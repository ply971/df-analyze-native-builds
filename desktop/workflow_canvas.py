"""Native draggable canvas. Every connection is validated by gui_workflow."""
from __future__ import annotations

from typing import Any, Callable

import dearpygui.dearpygui as dpg

from desktop import dpg_context as ui
from df_analyze import gui_common as gc
from df_analyze.gui_workflow import STAGES, Workflow, node_details


class WorkflowCanvas:
    def __init__(self, get_config: Callable[[], gc.RunConfig], changed: Callable[[], None],
                 navigate: Callable[[str], None]) -> None:
        self.get_config, self.changed, self.navigate = get_config, changed, navigate
        self.flow = Workflow.from_config(get_config())
        self.links: dict[int | str, tuple[str, str]] = {}
        self.ports: dict[int | str, tuple[str, str]] = {}
        self.items: dict[int | str, str] = {}

    @property
    def active(self) -> bool:
        return bool(dpg.get_value("workflow_enabled"))

    def build(self) -> None:
        with ui.group(tag="page_workflow", show=False):
            dpg.add_text("Visual workflow", color=(150, 200, 250))
            dpg.add_text("Design the experiment. Connect the learners. Compare the evidence.")
            dpg.add_spacer(height=12)
            with ui.group(horizontal=True):
                dpg.add_checkbox(label="Use canvas models for the next run", tag="workflow_enabled",
                                 callback=self.on_change)
                dpg.add_button(label="Rebuild from model settings", callback=self.rebuild)
                dpg.add_button(label="Arrange nodes", callback=self.arrange)
                dpg.add_button(label="Remove selected learners", callback=self.remove_selected)
            with ui.group(horizontal=True):
                dpg.add_combo(gc.CLASSIFIERS, default_value="rf", tag="workflow_model", width=180)
                dpg.add_button(label="Add learner", callback=self.add_model)
                dpg.add_button(label="Edit preparation", callback=lambda: self.navigate("prepare"))
                dpg.add_button(label="Tuning & validation", callback=lambda: self.navigate("tuning"))
                dpg.add_button(label="Review & run", callback=lambda: self.navigate("run"))
            dpg.add_text("Drag node headers to move. Drag an output pin to an input pin to connect. Ctrl-drag a link to disconnect.", wrap=900)
            dpg.add_text("All learners share the preparation and selection settings. Each selected feature method is compared by the engine.", color=(138, 150, 172), wrap=900)
            dpg.add_text("", tag="workflow_status", wrap=900)
            with ui.node_editor(tag="workflow_editor", callback=self.connect,
                                delink_callback=self.disconnect, minimap=True, height=600):
                pass
        self.render()

    def capture(self) -> Workflow:
        for item, node in self.items.items():
            if dpg.does_item_exist(item):
                point = dpg.get_item_pos(item)
                self.flow.positions[node] = [float(point[0]), float(point[1])]
        return self.flow

    def load(self, flow: Workflow) -> None:
        self.flow = flow
        dpg.set_value("workflow_enabled", True)
        self.render()

    def notify(self) -> None:
        errors = self.flow.problems(self.get_config().mode)
        message = " ".join(errors) if errors else f"Connected workflow · {len(self.flow.models())} learners · ready for review"
        dpg.set_value("workflow_status", message)
        dpg.configure_item("workflow_status", color=(232, 178, 78) if errors else (58, 199, 130))
        self.changed()

    def on_change(self, sender: int | str = 0, app_data: Any = None) -> None:
        self.notify()

    def refresh_labels(self) -> None:
        cfg = self.get_config()
        choices = gc.CLASSIFIERS if cfg.mode == "classify" else gc.REGRESSORS
        dpg.configure_item("workflow_model", items=choices)
        if dpg.get_value("workflow_model") not in choices:
            dpg.set_value("workflow_model", choices[0])
        for item, node in self.items.items():
            title, detail = node_details(node, cfg)
            dpg.configure_item(item, label=title)
            dpg.set_value(f"wf_detail_{node}", detail)
        self.notify()

    def render(self) -> None:
        dpg.delete_item("workflow_editor", children_only=True)
        self.links.clear()
        self.ports.clear()
        self.items.clear()
        ports: dict[tuple[str, str], int | str] = {}
        cfg = self.get_config()
        for node in self.flow.nodes:
            title, detail = node_details(node, cfg)
            with ui.node(label=title, parent="workflow_editor",
                         pos=self.flow.positions.get(node, [30.0, 30.0])) as item:
                self.items[item] = node
                if node != "data":
                    with ui.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as port:
                        dpg.add_text("Scores" if node == "evaluate" else "Data", color=(138, 150, 172))
                    ports[node, "in"] = port
                    self.ports[port] = node, "in"
                with ui.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    dpg.add_text(detail, tag=f"wf_detail_{node}", wrap=170, color=(150, 200, 250))
                    if node in STAGES:
                        page = {"data": "data", "prepare": "prepare", "selection": "models", "evaluate": "results"}[node]
                        dpg.add_button(label="Open settings" if node != "evaluate" else "Open results",
                                       callback=self.open_page, user_data=page)
                if node != "evaluate":
                    with ui.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as port:
                        dpg.add_text("Scores" if node.startswith("model:") else "Data", color=(58, 199, 130))
                    ports[node, "out"] = port
                    self.ports[port] = node, "out"
        for source, target in self.flow.edges:
            if (source, "out") in ports and (target, "in") in ports:
                item = dpg.add_node_link(ports[source, "out"], ports[target, "in"], parent="workflow_editor")
                self.links[item] = source, target
        self.notify()

    def open_page(self, sender: int | str, app_data: Any, user_data: str) -> None:
        self.navigate(user_data)

    def connect(self, sender: int | str, app_data: Any) -> None:
        try:
            first, second = app_data
            source, out_direction = self.ports[first]
            target, in_direction = self.ports[second]
            if out_direction == "in":
                source, target = target, source
                first, second = second, first
                out_direction, in_direction = in_direction, out_direction
            if out_direction != "out" or in_direction != "in":
                raise ValueError("Connect an output to an input.")
            before = len(self.flow.edges)
            self.flow.connect(source, target)
            if len(self.flow.edges) > before:
                item = dpg.add_node_link(first, second, parent="workflow_editor")
                self.links[item] = source, target
            self.notify()
        except (KeyError, ValueError, TypeError) as error:
            dpg.set_value("workflow_status", str(error))

    def disconnect(self, sender: int | str, app_data: Any) -> None:
        edge = self.links.pop(app_data, None)
        if edge:
            self.flow.disconnect(*edge)
            dpg.delete_item(app_data)
            self.notify()

    def add_model(self, sender: int | str = 0, app_data: Any = None) -> None:
        try:
            self.capture()
            self.flow.add_model(str(dpg.get_value("workflow_model")))
            self.render()
        except ValueError as error:
            dpg.set_value("workflow_status", str(error))

    def remove_selected(self, sender: int | str = 0, app_data: Any = None) -> None:
        self.capture()
        for item in dpg.get_selected_nodes("workflow_editor"):
            node = self.items.get(item, "")
            if node.startswith("model:"):
                self.flow.remove_model(node)
        self.render()

    def rebuild(self, sender: int | str = 0, app_data: Any = None) -> None:
        self.flow = Workflow.from_config(self.get_config())
        self.render()

    def arrange(self, sender: int | str = 0, app_data: Any = None) -> None:
        self.flow.arrange()
        self.render()
