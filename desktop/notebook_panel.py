"""Native notebook editor backed by a persistent Jupyter kernel."""
import base64
from io import BytesIO
from pathlib import Path
import re
import time

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

from desktop import dpg_context as ui
from df_analyze.gui_notebook import SNIPPETS, NotebookSession, context_code

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def render_note(source):
    """Present common Markdown blocks as native notebook reading cards."""
    lines = source.splitlines()
    index = 0
    fenced = False
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not line:
            dpg.add_spacer(height=6, parent="nb_output")
            continue
        if not fenced and line.startswith("|") and index < len(lines) and re.fullmatch(r"[\s|:\-]+", lines[index]):
            headers = [part.strip() for part in line.strip("|").split("|")]
            index += 1
            with ui.table(parent="nb_output", header_row=True, borders_innerV=True,
                          borders_outerH=True, row_background=True):
                for heading in headers:
                    dpg.add_table_column(label=heading)
                while index < len(lines) and lines[index].strip().startswith("|"):
                    with ui.table_row():
                        for value in lines[index].strip().strip("|").split("|")[:len(headers)]:
                            dpg.add_text(value.strip(), wrap=240)
                    index += 1
            continue
        color = (211, 221, 237)
        if not fenced:
            if line.startswith("#"):
                line = line.lstrip("# ")
                color = (183, 160, 224)
                dpg.add_spacer(height=8, parent="nb_output")
            elif line.startswith(">"):
                line = line.lstrip("> ")
                color = (104, 203, 176)
            line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        dpg.add_text(line, color=color, parent="nb_output", wrap=850)


class NotebookPanel:
    def __init__(self, root, config_provider):
        self.session = NotebookSession(root)
        self.config_provider = config_provider
        self.selected = 0
        self.rendered_revision = -1
        self.last_poll = 0

    def save_editor(self):
        if self.session.cells and not self.session.busy:
            self.session.edit(self.selected, dpg.get_value("nb_source"))

    def action(self, callback):
        try:
            self.save_editor()
            callback()
            self.refresh()
        except Exception as error:
            dpg.set_value("nb_status", str(error))

    def choose(self, sender, value):
        self.save_editor()
        self.selected = int(value.split(".", 1)[0]) - 1
        self.refresh()

    def add(self, kind):
        self.session.add(kind)
        self.selected = len(self.session.cells) - 1

    def insert_context(self):
        self.session.add(source=context_code(self.config_provider()))
        self.selected = len(self.session.cells) - 1

    def insert_helper(self):
        self.session.add(source=SNIPPETS[dpg.get_value("nb_snippet")])
        self.selected = len(self.session.cells) - 1

    def duplicate(self):
        self.selected = self.session.duplicate(self.selected)

    def undo_delete(self):
        restored = self.session.undo_delete()
        if restored is not None:
            self.selected = restored

    def move(self, delta):
        destination = self.selected + delta
        if 0 <= destination < len(self.session.cells):
            self.session.move(self.selected, delta)
            self.selected = destination

    def open(self, sender, data):
        path = data.get("file_path_name", "")
        if path:
            def load():
                self.session.load_ipynb(Path(path).read_text(encoding="utf-8"))
                self.selected = 0
            self.action(load)

    def export(self, sender, data):
        path = data.get("file_path_name", "")
        if path:
            def save():
                destination = Path(path)
                if destination.suffix.lower() != ".ipynb":
                    destination = destination.with_suffix(".ipynb")
                destination.write_text(self.session.to_ipynb(), encoding="utf-8")
                self.session.status = f"Notebook saved to {destination}"
            self.action(save)

    def build(self):
        with ui.theme() as editor_theme:
            with ui.theme_component(dpg.mvInputText):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (14, 23, 39))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (185, 224, 215))
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 16, 14)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 9)
        with ui.texture_registry(tag="nb_textures"):
            pass
        for tag, callback, default_name in (("nb_open_dialog", self.open, ""), ("nb_save_dialog", self.export, "df_analyze_notebook.ipynb")):
            with ui.file_dialog(tag=tag, show=False, callback=callback, width=800, height=480,
                                default_filename=default_name):
                dpg.add_file_extension(".ipynb")
        with ui.group(tag="page_notebook", show=False):
            with ui.child_window(height=145, width=-1, border=True):
                dpg.add_text("DF / FIELD NOTES                                      PYTHON LAB", color=(183, 160, 224))
                dpg.add_spacer(height=6)
                dpg.add_text("A little curiosity. A new discovery.", color=(234, 237, 248))
                dpg.add_text("Explore a question, sketch an idea in Python, and turn the results into a story.", color=(157, 176, 201), wrap=850)
                dpg.add_spacer(height=6)
                dpg.add_text("", tag="nb_summary", color=(104, 203, 176))
            dpg.add_spacer(height=8)
            with ui.group(tag="nb_edit_controls"):
                with ui.group(horizontal=True):
                    dpg.add_button(label="+ Code", callback=lambda: self.action(lambda: self.add("code")))
                    dpg.add_button(label="+ Markdown", callback=lambda: self.action(lambda: self.add("markdown")))
                    dpg.add_button(label="Run cell", tag="nb_run_cell", callback=lambda: self.action(lambda: self.session.execute(self.selected)))
                    dpg.add_button(label="Run all", callback=lambda: self.action(self.session.execute))
                    dpg.add_button(label="Restart Python", callback=lambda: self.action(self.session.restart))
                    dpg.add_button(label="Clear outputs", callback=lambda: self.action(self.session.clear_outputs))
                with ui.group(horizontal=True):
                    dpg.add_button(label="Insert app dataset and settings", callback=lambda: self.action(self.insert_context))
                    dpg.add_button(label="Open notebook", callback=lambda: dpg.show_item("nb_open_dialog"))
                    dpg.add_button(label="Save notebook", callback=lambda: dpg.show_item("nb_save_dialog"))
                dpg.add_text("Setup keeps raw_df and applies your preparation recipe to df. Insert setup again after changing settings.", wrap=850)
                with ui.group(horizontal=True):
                    dpg.add_combo(list(SNIPPETS), default_value=next(iter(SNIPPETS)), tag="nb_snippet", width=220)
                    dpg.add_button(label="Insert helper", callback=lambda: self.action(self.insert_helper))
                    dpg.add_button(label="Undo delete", tag="nb_undo", callback=lambda: self.action(self.undo_delete))
                    dpg.add_button(label="Run through here", tag="nb_run_through", callback=lambda: self.action(lambda: self.session.execute(self.selected, scope="through")))
                    dpg.add_button(label="Run from here", tag="nb_run_from", callback=lambda: self.action(lambda: self.session.execute(self.selected, scope="from")))
                with ui.group(horizontal=True):
                    dpg.add_listbox(tag="nb_cells", items=[], num_items=6, width=300, callback=self.choose)
                    with ui.group():
                        dpg.add_button(label="Move up", callback=lambda: self.action(lambda: self.move(-1)))
                        dpg.add_button(label="Move down", callback=lambda: self.action(lambda: self.move(1)))
                        dpg.add_button(label="Duplicate cell", tag="nb_duplicate", callback=lambda: self.action(self.duplicate))
                        dpg.add_button(label="Delete cell", tag="nb_delete", callback=lambda: self.action(lambda: self.session.remove(self.selected)))
                dpg.add_text("", tag="nb_editor_heading", color=(104, 203, 176))
                dpg.add_input_text(tag="nb_source", multiline=True, height=270, width=-1, tab_input=True)
                dpg.bind_item_theme("nb_source", editor_theme)
            dpg.add_button(label="Interrupt Python", tag="nb_interrupt", enabled=False, callback=lambda: self.session.interrupt())
            dpg.add_button(label="Stop and reset Python", tag="nb_stop", enabled=False, callback=lambda: self.session.stop())
            dpg.add_text("Ready", tag="nb_status", wrap=850)
            with ui.group(tag="nb_stdin", show=False):
                dpg.add_text("", tag="nb_prompt")
                dpg.add_input_text(tag="nb_answer", width=500)
                dpg.add_button(label="Send input", callback=lambda: self.session.answer(dpg.get_value("nb_answer")))
            dpg.add_spacer(height=6)
            dpg.add_text("OUTPUT / OBSERVATIONS", color=(183, 160, 224))
            with ui.child_window(tag="nb_output", height=400, width=-1, horizontal_scrollbar=True):
                pass
            dpg.add_text("Local Python session. Variables persist until restart. Imported cells run only when you press Run.", color=(138, 150, 172), wrap=850)
        self.refresh()

    def refresh(self):
        snapshot = self.session.snapshot()
        entries = snapshot["cells"]
        self.selected = min(self.selected, max(0, len(entries) - 1))
        labels = self.cell_labels(entries)
        dpg.configure_item("nb_cells", items=labels)
        dpg.set_value("nb_cells", labels[self.selected] if labels else "")
        dpg.set_value("nb_source", entries[self.selected]["source"] if entries else "")
        dpg.set_value("nb_editor_heading", f"{'CODE / PYTHON' if entries[self.selected]['cell_type'] == 'code' else 'NOTE / MARKDOWN'}  —  CELL {self.selected + 1:02d}" if entries else "YOUR NEXT QUESTION STARTS HERE")
        self.rendered_revision = -1
        self.poll(force=True)

    @staticmethod
    def cell_labels(entries):
        return [f"{i + 1}. {entry['cell_type']}  In [{entry.get('execution_count') or ' '}]  "
                + next((line.strip()[:45] for line in entry["source"].splitlines() if line.strip()), "Empty cell")
                for i, entry in enumerate(entries)]

    def poll(self, force=False):
        if not force and (not dpg.is_item_shown("page_notebook") or time.monotonic() - self.last_poll < .25):
            return
        self.last_poll = time.monotonic()
        snapshot = self.session.snapshot()
        code_count = sum(entry["cell_type"] == "code" for entry in snapshot["cells"])
        dpg.set_value("nb_summary", f"{code_count:02d} CODE CELLS     /     {len(snapshot['cells']) - code_count:02d} NOTES     /     {'PYTHON RUNNING' if snapshot['busy'] else 'READY TO EXPLORE'}")
        dpg.set_value("nb_status", snapshot["status"])
        dpg.configure_item("nb_edit_controls", enabled=not snapshot["busy"])
        dpg.configure_item("nb_interrupt", enabled=snapshot["busy"])
        dpg.configure_item("nb_stop", enabled=snapshot["busy"])
        dpg.configure_item("nb_undo", enabled=not snapshot["busy"] and snapshot["can_undo_delete"])
        for tag in ("nb_run_cell", "nb_run_through", "nb_run_from", "nb_duplicate", "nb_delete", "nb_source"):
            dpg.configure_item(tag, enabled=not snapshot["busy"] and bool(snapshot["cells"]))
        prompt = snapshot["prompt"]
        dpg.configure_item("nb_stdin", show=bool(prompt))
        if prompt:
            dpg.set_value("nb_prompt", prompt.get("prompt", "Python input"))
            dpg.configure_item("nb_answer", password=prompt.get("password", False))
        if self.rendered_revision == snapshot["revision"]:
            return
        self.rendered_revision = snapshot["revision"]
        labels = self.cell_labels(snapshot["cells"])
        dpg.configure_item("nb_cells", items=labels)
        dpg.set_value("nb_cells", labels[self.selected] if labels else "")
        dpg.delete_item("nb_output", children_only=True)
        dpg.delete_item("nb_textures", children_only=True)
        if not snapshot["cells"]:
            return
        entry = snapshot["cells"][self.selected]
        if entry["cell_type"] == "markdown":
            render_note(entry["source"])
        elif entry["cell_type"] != "code":
            dpg.add_text(entry["source"], parent="nb_output", wrap=850)
        for output in entry.get("outputs", []):
            data = output.get("data", {})
            encoded = data.get("image/png") or data.get("image/jpeg")
            if encoded:
                with Image.open(BytesIO(base64.b64decode("".join(encoded)))) as image:
                    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.
                height, width = rgba.shape[:2]
                texture = dpg.add_static_texture(width, height, rgba.ravel(), parent="nb_textures")
                scale = min(1., 820 / width)
                dpg.add_image(texture, parent="nb_output", width=int(width * scale), height=int(height * scale))
            else:
                text = output.get("text") or data.get("text/plain") or "\n".join(output.get("traceback", []))
                text = "".join(text) if isinstance(text, list) else text
                dpg.add_text(ANSI.sub("", text or "Rich output available in the saved notebook."), parent="nb_output", wrap=850)
