from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import randint, uniform
from typing import TYPE_CHECKING, Optional, cast

import jsonpickle
import numpy as np
from pandas import DataFrame, Series

if TYPE_CHECKING:
    from df_analyze.cli.cli import ProgramOptions
from df_analyze.enumerables import (
    ClsScore,
    RegScore,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.selection.stepwise import RedundantFeatures, stepwise_select
from df_analyze.testing.datasets import TestDataset


@dataclass
class WrapperSelected:
    method: WrapperSelection
    model: WrapperSelectionModel
    selected: list[str]
    scores: dict[str, float]
    redundants: list[RedundantFeatures]
    early_stop: bool
    is_classification: bool

    def to_markdown(self) -> str:
        fnames, scores = zip(*self.scores.items())
        scores = DataFrame(
            data=scores, index=Series(name="feature", data=fnames), columns=["score"]
        )
        metric = ClsScore.Accuracy if self.is_classification else RegScore.MAE
        mname = metric.longname()
        higher_better = self.is_classification
        direction = "Higher" if higher_better else "Lower"
        is_redundant = len(self.redundants) > 0
        text = (
            "# Wrapper-Based Feature Selection Summary\n\n"
            f"Wrapper method:    {self.method.name}\n"
            f"Wrapper model:     {self.model.name}\n"
            f"Redundancy-aware:  {is_redundant}\n"
            "\n"
            f"## Selected Features\n\n"
            f"{self.selected}\n\n"
            f"## Selection Scores ({mname}: {direction} = More important)\n\n"
            f"{scores.to_markdown(floatfmt='0.3e')}\n\n"
        )
        lines = []
        for i, redundant in enumerate(self.redundants):
            lines.append(
                redundant.to_markdown_section(
                    iteration=i, is_cls=self.is_classification
                )
            )
        redundant_info = "".join(lines)

        return text + redundant_info

    @staticmethod
    def from_json(path: Path) -> WrapperSelected:
        return cast(WrapperSelected, jsonpickle.decode(path.read_text()))

    def to_json(self) -> str:
        return str(jsonpickle.encode(self))

    @staticmethod
    def random(ds: TestDataset) -> WrapperSelected:
        method = WrapperSelection.random()
        model = WrapperSelectionModel.random()
        df = ds.load()
        is_cls = ds.is_classification
        x = df.drop(columns="target", errors="ignore")
        cols = x.columns.to_list()
        n_feat = randint(min(10, x.shape[1]), x.shape[1])
        selected = np.random.choice(cols, size=n_feat, replace=False).tolist()
        raw_scores = [uniform(0.7, 1) if is_cls else uniform(0, 0.3) for s in selected]
        raw_scores = sorted(raw_scores, reverse=is_cls)
        scores = {s: score for s, score in zip(selected, raw_scores)}
        remaining_feats = set(cols)
        metric = "acc" if is_cls else "mae"
        redundants = []
        total_feats = 0
        for feat in selected:
            best = feat
            best_score = scores[feat]

            remaining_feats.discard(feat)
            n_max = len(remaining_feats) // 2
            if n_max > 0:
                n_redundant = np.random.randint(0, n_max)
            else:
                n_redundant = 0
            feats, feat_scores = [], []
            if n_redundant > 0:
                feats = np.random.choice(
                    np.array(list(remaining_feats)), n_redundant, replace=False
                ).tolist()
                remaining_feats.difference_update(feats)
                feat_scores = [uniform(-0.01, 0.01) + best_score for f in feats]
            redundants.append(
                RedundantFeatures(
                    best=best,
                    best_score=best_score,
                    features=feats,
                    scores=feat_scores,
                    metric=metric,
                )
            )
            total_feats += redundants[-1].n_feat()

        return WrapperSelected(
            method=method,
            model=model,
            selected=selected,
            redundants=redundants,
            scores=scores,
            early_stop=total_feats >= len(cols),
            is_classification=ds.is_classification,
        )


def wrap_select_features(
    prep_train: PreparedData, options: ProgramOptions
) -> Optional[WrapperSelected]:
    if options.wrapper_select is None:
        return None

    result = stepwise_select(prep_train=prep_train, options=options)
    if result is None:
        return None
    selected, scores, redundants, early_stop = result

    return WrapperSelected(
        selected=selected,
        scores=scores,
        method=options.wrapper_select,
        model=options.wrapper_model,
        redundants=redundants,
        early_stop=early_stop,
        is_classification=prep_train.is_classification,
    )
