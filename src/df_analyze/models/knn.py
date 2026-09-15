from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import platform
from typing import Any, Mapping, Optional, Type, Union

import numpy as np
import optuna
from optuna import Study, Trial
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel


class KNNEstimator(DfAnalyzeModel):
    shortname = "knn"
    longname = "K-Neighbours Estimator"
    timeout_s = 30 * 60  # 30 minutes is enough given the grid

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        n_jobs = 1 if platform.system().lower() == "darwin" else -1
        self.is_classifier = False
        self.needs_calibration = False
        self.target_cols: list[str] = []
        self.fixed_args = dict(n_jobs=n_jobs)
        self.model_cls: Type[Any] = type(None)
        self.grid = {
            "n_neighbors": [1, 5, 10, 25, 50],
            "weights": ["uniform", "distance"],
            "metric": ["cosine", "l2", "correlation"],
        }

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 50, step=1),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical(
                "metric", ["cosine", "l1", "l2", "correlation"]
            ),
        )

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        if self.grid is None:
            raise ValueError("Impossible!")
        n_hp = 1
        for opts in self.grid.values():
            n_hp *= len(opts)
        n_jobs = -1 if self.fixed_args["n_jobs"] == 1 else 1
        return super().htune_optuna(
            X_train,
            y_train,
            g_train,
            metric=metric,
            n_trials=n_hp,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )

    def _target_cols_for_output(self, n_targets: int) -> list[str]:
        if len(self.target_cols) == n_targets:
            return self.target_cols
        return [f"target_{i}" for i in range(n_targets)]

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
        else:
            name = y.name if y.name is not None else "target"
            self.target_cols = [str(name)]
        super().refit_tuned(X=X, y=y, g=g, tuned_args=tuned_args)

    def tuned_predict(self, X: DataFrame) -> Union[Series, DataFrame, np.ndarray]:
        preds = super().tuned_predict(X)
        if (
            isinstance(preds, np.ndarray)
            and preds.ndim == 2
            and preds.shape[1] > 1
        ):
            cols = self._target_cols_for_output(preds.shape[1])
            return DataFrame(preds, index=X.index, columns=cols)
        return preds


class KNNClassifier(KNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Classifier"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = KNeighborsClassifier

    def predict_proba(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        probs = super().predict_proba(X)
        if isinstance(probs, (list, tuple)):
            if len(probs) == 1:
                return np.asarray(probs[0])
            cols = self._target_cols_for_output(len(probs))
            return {col: np.asarray(arr) for col, arr in zip(cols, probs)}
        if (
            isinstance(probs, np.ndarray)
            and probs.ndim == 3
            and probs.shape[1] > 1
        ):
            cols = self._target_cols_for_output(probs.shape[1])
            return {col: np.asarray(probs[:, i, :]) for i, col in enumerate(cols)}
        return probs


class KNNRegressor(KNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Regressor"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = KNeighborsRegressor
        self.shortname = "knn"
        self.longname = "K-Neighbours Regressor"
