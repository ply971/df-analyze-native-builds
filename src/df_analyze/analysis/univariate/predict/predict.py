from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Optional,
)
from warnings import WarningMessage, warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from df_analyze.analysis.univariate.predict.models import (
    CLS_MODELS,
    REG_MODELS,
)
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.saving import add_fold_idx


@dataclass
class PredFiles:
    conts_raw = "continuous_features.parquet"
    cats_raw = "categorical_features.parquet"
    conts_csv = "continuous_features.csv"
    cats_csv = "categorical_features.csv"
    idx = "subsample.npy"


class PredResults:
    def __init__(
        self,
        conts: Optional[DataFrame],
        cats: Optional[DataFrame],
        is_classification: bool,
        idx: Optional[ndarray] = None,
        errs: Optional[list[BaseException]] = None,
        warns: Optional[list[WarningMessage]] = None,
    ) -> None:
        self.conts = conts
        self.cats = cats
        self.is_classification = is_classification
        self.errs = errs
        self.warns = warns
        self.idx_subsample = idx
        self.files = PredFiles()

    def save_raw(self, root: Path, fold_idx: Optional[int]) -> None:
        if self.conts is not None:
            self.conts.to_parquet(
                add_fold_idx(root / self.files.conts_raw, fold_idx=fold_idx)
            )
        else:
            DataFrame().to_parquet(
                add_fold_idx(root / self.files.conts_raw, fold_idx=fold_idx)
            )
        if self.cats is not None:
            self.cats.to_parquet(
                add_fold_idx(root / self.files.cats_raw, fold_idx=fold_idx)
            )
        else:
            DataFrame().to_parquet(
                add_fold_idx(root / self.files.conts_raw, fold_idx=fold_idx)
            )
        if self.idx_subsample is not None:
            with open(root / self.files.idx, "wb") as handle:
                np.save(handle, self.idx_subsample, allow_pickle=False, fix_imports=False)

    def save_tables(self, root: Path, fold_idx: Optional[int]) -> None:
        if self.conts is not None:
            self.conts.to_csv(
                add_fold_idx(root / self.files.conts_csv, fold_idx=fold_idx)
            )
        if self.cats is not None:
            self.cats.to_csv(add_fold_idx(root / self.files.cats_csv, fold_idx=fold_idx))

    @staticmethod
    def is_saved(cachedir: Path) -> bool:
        files = PredFiles()
        conts = cachedir / files.conts_raw
        cats = cachedir / files.cats_raw
        return conts.exists() and cats.exists()

    @staticmethod
    def load(cachedir: Path, is_classification: bool) -> PredResults:
        # TODO: handle fold_idx
        preds = PredResults(conts=None, cats=None, is_classification=is_classification)
        try:
            conts = pd.read_parquet(cachedir / preds.files.conts_raw)
            cats = pd.read_parquet(cachedir / preds.files.cats_raw)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing cached predictions at {cachedir}")

        preds.conts = None if conts.empty else conts
        preds.cats = None if cats.empty else cats

        npy_file = cachedir / preds.files.idx
        if npy_file.exists():
            preds.idx_subsample = np.load(npy_file, allow_pickle=False, fix_imports=False)
        return preds

    def to_markdown(self, path: Optional[Path] = None) -> Optional[str]:
        try:
            sorter = "acc" if self.is_classification else "var-exp"
            if self.conts is not None:
                conts = self.conts.sort_values(by=sorter, ascending=False).to_markdown(
                    floatfmt="0.4g"
                )
                cont_table = f"# Continuous predictions\n\n{conts}\n\n"
            else:
                cont_table = ""
            if self.cats is not None:
                cats = self.cats.sort_values(by=sorter, ascending=False).to_markdown(
                    floatfmt="0.4g"
                )
                cats_table = f"# Categorical prediction:\n\n{cats}"
            else:
                cats_table = ""

            cont_legend = (
                "\n\n"
                "mae: Mean Absolute Error\n"
                "msqe: Mean Squared Error\n"
                "mdae: Median Absolute Error\n"
                "r2: R-squared (coefficient of determination)\n"
                "var-exp: Percent Variance Explained\n"
            )

            cat_legend = (
                "\n\n"
                "acc: accuracy\n"
                "auroc: area under the receiving operater characteristic curve\n"
                "sens: sensitivity\n"
                "spec: specificity\n"
            )
            legend = cat_legend if self.is_classification else cont_legend
            tables = cont_table + cats_table

            if tables.replace("\n", "") != "":
                tables = tables + legend
                if path is not None:
                    path.write_text(tables, encoding="utf-8")
                return tables
        except Exception as e:
            warn(
                "Got exception when attempting to make predictions report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )


def continuous_feature_target_preds(
    continuous: DataFrame,
    column: str,
    target: Series,
    is_classification: bool,
) -> tuple[DataFrame, Optional[Exception], list[WarningMessage]]:
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("once")
        try:
            x = continuous[column].to_numpy().reshape(-1, 1)
            y = target
            models = CLS_MODELS if is_classification else REG_MODELS
            scores = []
            # pbar = tqdm(
            #     models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
            # )
            for model_cls in models:
                # pbar.set_description(f"Tuning {model_cls.__name__}")
                model = model_cls()
                score, spam = model.evaluate(x, y)
                score.insert(0, "model", model.short)
                score.index = pd.Index([column], name="feature")
                scores.append(score)
                # pbar.update()
            # pbar.clear()
            return pd.concat(scores, axis=0), None, warns
        except Exception as e:
            traceback.print_exc()
            print(f"Got error generating predictions for column {column}: {e}")
            return DataFrame(), e, warns


def categorical_feature_target_preds(
    categoricals: DataFrame,
    column: str,
    target: Series,
    is_classification: bool,
) -> tuple[DataFrame, Optional[Exception], list[WarningMessage]]:
    """Must be UN-ENCODED categoricals"""
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("once")
        try:
            X = pd.get_dummies(
                categoricals[column], dummy_na=True, dtype=float
            ).to_numpy()
            y = target
            if is_classification:
                y = Series(data=LabelEncoder().fit_transform(target), name=target.name)  # type: ignore
            models = CLS_MODELS if is_classification else REG_MODELS
            scores = []
            # pbar = tqdm(
            #     models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
            # )
            for model_cls in models:
                # pbar.set_description(f"Tuning {model_cls.__name__}")
                model = model_cls()
                score, spam = model.evaluate(X, y)
                score.insert(0, "model", model.short)
                score.index = pd.Index([column], name="feature")
                scores.append(score)
                # pbar.update()
            # pbar.clear()
            return pd.concat(scores, axis=0), None, warns
        except Exception as e:
            traceback.print_exc()
            print(f"Got error generating predictions for column {column}: {e}")
            return DataFrame(), e, warns


def feature_target_predictions(
    categoricals: DataFrame | None,
    continuous: DataFrame | None,
    target: Series,
    is_classification: bool,
) -> tuple[
    Optional[DataFrame], Optional[DataFrame], list[BaseException], list[WarningMessage]
]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df_conts: list[DataFrame]
        df_cats: list[DataFrame]
        results: list[tuple[DataFrame, Optional[Exception], list[WarningMessage]]] = []

        if continuous is not None:
            results = Parallel(n_jobs=-1)(
                delayed(continuous_feature_target_preds)(
                    continuous=continuous,
                    column=col,
                    target=target,
                    is_classification=is_classification,
                )
                for col in tqdm(
                    continuous.columns,
                    desc="Predicting continuous features",
                    total=continuous.shape[1],
                    leave=True,
                    position=0,
                )
            )  # type: ignore
        if len(results) > 0:
            df_conts, cont_errors, cont_warns = list(zip(*results))  # type: ignore
        else:
            df_conts, cont_errors, cont_warns = [], [], []

        if categoricals is not None:
            results = Parallel(n_jobs=-1)(
                delayed(categorical_feature_target_preds)(
                    categoricals=categoricals,
                    column=col,
                    target=target,
                    is_classification=is_classification,
                )
                for col in tqdm(
                    categoricals.columns,
                    desc="Predicting categorical features",
                    total=categoricals.shape[1],
                    leave=True,
                    position=0,
                )
            )  # type: ignore
        if len(results) > 0:
            df_cats, cat_errors, cat_warns = list(zip(*results))  # type: ignore
        else:
            df_cats, cat_errors, cat_warns = [], [], []

        df_cont = pd.concat(df_conts, axis=0) if len(df_conts) != 0 else None
        df_cat = pd.concat(df_cats, axis=0) if len(df_cats) != 0 else None
        errs = [*cont_errors, *cat_errors]
        errs = [e for e in errs if e is not None]
        all_warns = []
        if isinstance(cont_warns, tuple):
            cont_warns = list(cont_warns)
        if isinstance(cat_warns, tuple):
            cat_warns = list(cat_warns)
        for warns in cont_warns + cat_warns:
            all_warns.extend(warns)
        all_warns = [w for w in all_warns if w is not None]

    return df_cont, df_cat, errs, all_warns


def univariate_predictions(
    prepared: PreparedData,
    is_classification: bool,
) -> PredResults:
    sub, idx = prepared.representative_subsample()
    df_cont, df_cat, errs, warns = feature_target_predictions(
        categoricals=sub.X_cat,
        continuous=sub.X_cont,
        target=sub.y,
        is_classification=is_classification,
    )
    return PredResults(
        conts=df_cont,
        cats=df_cat,
        is_classification=is_classification,
        idx=idx,
        errs=errs,
        warns=warns,
    )
