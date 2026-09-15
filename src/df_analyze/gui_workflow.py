"""Executable visual workflows shared by both interfaces.

The engine compares learners on one common data/preparation/selection pipeline.
The graph deliberately models that supported topology rather than accepting
arbitrary connections whose execution the engine cannot implement.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

from df_analyze import gui_common as gc

STAGES = {"data": "Dataset", "prepare": "Preparation", "selection": "Feature selection",
          "evaluate": "Evaluate & compare"}


@dataclass
class Workflow:
    nodes: list[str] = field(default_factory=lambda: list(STAGES))
    edges: list[tuple[str, str]] = field(default_factory=list)
    positions: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: gc.RunConfig) -> Workflow:
        flow = cls(edges=[("data", "prepare"), ("prepare", "selection")])
        for model in dict.fromkeys(cfg.models):
            flow.add_model(model)
        flow.arrange()
        return flow

    def add_model(self, model: str) -> None:
        if model not in set(gc.CLASSIFIERS + gc.REGRESSORS):
            raise ValueError(f"Unknown model: {model}")
        node = "model:" + model
        if node in self.nodes:
            raise ValueError(f"{model} is already on the canvas.")
        self.nodes.append(node)
        self.edges.extend([("selection", node), (node, "evaluate")])
        self.positions[node] = [710.0, 40.0 + 125.0 * (len(self.models()) - 1)]

    def models(self) -> list[str]:
        return [node[6:] for node in self.nodes if node.startswith("model:")]

    def remove_model(self, node: str) -> None:
        if node in STAGES:
            raise ValueError("Data, preparation, selection and evaluation are required stages.")
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        self.edges = [edge for edge in self.edges if node not in edge]
        self.positions.pop(node, None)

    @staticmethod
    def allowed_edge(source: str, target: str) -> bool:
        return (source, target) in {("data", "prepare"), ("prepare", "selection")} or (
            source == "selection" and target.startswith("model:")
        ) or (source.startswith("model:") and target == "evaluate")

    def connect(self, source: str, target: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both endpoints must exist on the canvas.")
        if not self.allowed_edge(source, target):
            raise ValueError("Connect Dataset → Preparation → Selection → Model → Evaluation.")
        edge = (source, target)
        if edge not in self.edges:
            self.edges.append(edge)

    def disconnect(self, source: str, target: str) -> None:
        self.edges = [edge for edge in self.edges if edge != (source, target)]

    def problems(self, mode: str) -> list[str]:
        errors = []
        if set(STAGES).difference(self.nodes):
            errors.append("The workflow is missing a required stage.")
        if len(set(self.nodes)) != len(self.nodes):
            errors.append("Workflow nodes must be unique.")
        choices = gc.CLASSIFIERS if mode == "classify" else gc.REGRESSORS
        for node in self.nodes:
            if node not in STAGES and not (node.startswith("model:") and node[6:] in choices):
                errors.append(f"{node} is not supported for {mode}. Rebuild from experiment settings.")
        for source, target in self.edges:
            if source not in self.nodes or target not in self.nodes or not self.allowed_edge(source, target):
                errors.append("The workflow contains an invalid connection.")
        if len(set(self.edges)) != len(self.edges):
            errors.append("Workflow connections must be unique.")
        for source, target in [("data", "prepare"), ("prepare", "selection")]:
            if (source, target) not in self.edges:
                errors.append(f"Connect {STAGES[source]} to {STAGES[target]}.")
        if not self.models():
            errors.append("Add at least one learner to compare.")
        for model in self.models():
            node = "model:" + model
            if ("selection", node) not in self.edges or (node, "evaluate") not in self.edges:
                errors.append(f"Connect both ports of {model}, or remove that learner.")
        return list(dict.fromkeys(errors))

    def compile(self, cfg: gc.RunConfig) -> gc.RunConfig:
        errors = self.problems(cfg.mode)
        if errors:
            raise ValueError(" ".join(errors))
        return replace(cfg, models=self.models())

    def arrange(self) -> None:
        center = max(65.0, (len(self.models()) - 1) * 62.5 + 40.0)
        self.positions = {"data": [25.0, center], "prepare": [255.0, center],
                          "selection": [485.0, center], "evaluate": [955.0, center]}
        self.positions.update({"model:" + model: [710.0, 40.0 + 125.0 * i]
                               for i, model in enumerate(self.models())})

    def as_dict(self) -> dict[str, Any]:
        return {"nodes": list(self.nodes), "edges": [list(edge) for edge in self.edges],
                "positions": {key: list(value) for key, value in self.positions.items()}}

    @classmethod
    def from_dict(cls, value: object) -> Workflow:
        if not isinstance(value, dict):
            raise ValueError("Workflow must be a JSON object.")
        nodes, edges, positions = value.get("nodes"), value.get("edges"), value.get("positions", {})
        if not isinstance(nodes, list) or len(nodes) > 30 or not all(isinstance(n, str) for n in nodes):
            raise ValueError("Workflow must contain at most 30 named nodes.")
        if not isinstance(edges, list) or len(edges) > 100 or not all(
            isinstance(edge, (list, tuple)) and len(edge) == 2
            and all(isinstance(n, str) for n in edge) for edge in edges
        ):
            raise ValueError("Workflow connections must be pairs of node names.")
        if not isinstance(positions, dict):
            raise ValueError("Workflow positions must be an object.")
        clean_positions = {}
        for key, point in positions.items():
            if key not in nodes or not isinstance(point, list) or len(point) != 2 or not all(
                type(v) in (int, float) and math.isfinite(v) and abs(v) <= 10000 for v in point
            ):
                raise ValueError("Invalid workflow node position.")
            clean_positions[key] = [float(v) for v in point]
        return cls(list(nodes), [(edge[0], edge[1]) for edge in edges], clean_positions)

    def to_json(self, cfg: gc.RunConfig) -> str:
        document = json.loads(gc.config_to_json(cfg))
        document["workflow"] = self.as_dict()
        return json.dumps(document, indent=2, ensure_ascii=False)


def import_workflow(text: str) -> Workflow | None:
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("Experiment must be a JSON object.")
    return Workflow.from_dict(document["workflow"]) if "workflow" in document else None


def node_details(node: str, cfg: gc.RunConfig) -> tuple[str, str]:
    if node == "data":
        return "Dataset", cfg.data_path.name
    if node == "prepare":
        return "Preparation", f"{len(cfg.pipeline_steps)} steps · {len(cfg.drops)} excluded columns"
    if node == "selection":
        return "Feature selection", " + ".join(cfg.feat_select)
    if node == "evaluate":
        return "Evaluate & compare", f"{cfg.htune_metric} · {cfg.test_val_size:.0%} holdout"
    return node.removeprefix("model:").upper(), f"{cfg.htune_trials} tuning trials"
