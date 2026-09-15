# ref: Multiclass & multioutput learning in scikit-learn: https://scikit-learn.org/stable/modules/multiclass.html

from __future__ import annotations

import json
import re
import traceback
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Generator as Gen
from typing import Literal, Optional, Tuple, Union, cast
from warnings import warn

import jsonpickle
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer, RobustScaler

from df_analyze._constants import (
    N_CAT_LEVEL_MIN,
    N_MULTITARGET_LEVEL_MIN,
    N_TARG_LEVEL_MIN,
    SEED,
    UNIVARIATE_PRED_MAX_N_SAMPLES,
)
from df_analyze.analysis.univariate.describe import describe_all_features
from df_analyze.enumerables import NanHandling, ValidationMethod
from df_analyze.preprocessing.cleaning import (
    clean_regression_target,
    deflate_categoricals,
    drop_target_nans,
    drop_unusable,
    encode_categoricals,
    encode_target,
    encode_targets,
    handle_continuous_nans,
    normalize_continuous,
    reindex,
)
from df_analyze.preprocessing.inspection.inspection import (
    ClsTargetInfo,
    InspectionResults,
    RegTargetInfo,
    convert_categoricals,
    inspect_target,
    unify_nans,
)
from df_analyze.preprocessing.targets import TargetSpec, as_target_list
from df_analyze.splitting import ApproximateStratifiedGroupSplit, y_split_label
from df_analyze.timing import timed


@dataclass
class PrepFiles:
    X_raw: str = "X.parquet"
    X_cont_raw: str = "X_cont.parquet"
    X_cat_raw: str = "X_cat.parquet"
    y_raw: str = "y.parquet"
    g_raw: str = "g.parquet"
    labels: str = "labels.parquet"
    ix_train: str = "idx_train.json"
    ix_tests: str = "idx_tests.json"
    info: str = "info.json"


@dataclass
class PrepFilesTrain:
    X_raw: str = "X_train.parquet"
    X_cont_raw: str = "X_train_cont.parquet"
    X_cat_raw: str = "X_train_cat.parquet"
    y_raw: str = "y_train.parquet"
    g_raw: str = "g.parquet"
    labels: str = "labels.parquet"
    info: str = "info.json"


@dataclass
class PrepFilesTest:
    X_raw: str = "X_test.parquet"
    X_cont_raw: str = "X_test_cont.parquet"
    X_cat_raw: str = "X_test_cat.parquet"
    y_raw: str = "y_test.parquet"
    g_raw: str = "g.parquet"
    labels: str = "labels.parquet"
    info: str = "info.json"


@dataclass
class PreparationInfo:
    is_classification: bool
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    n_samples_dropped_via_target_NaNs: int
    n_cont_indicator_added: int
    target_info: Optional[Union[RegTargetInfo, ClsTargetInfo]]
    runtimes: dict[str, float]

    def to_markdown(self) -> str:
        sections = []
        og = self.original_shape
        fs = self.final_shape
        task = "Classification" if self.is_classification else "Regression"
        orig_shape = f"{og[0]} samples × {og[1]} features"
        final_shape = f"{fs[0]} samples × {fs[1]} features"
        n_drop = self.n_samples_dropped_via_target_NaNs
        n_ind = self.n_cont_indicator_added
        funcs, times = zip(*self.runtimes.items())

        sections.append("# Data Preparation Summary\n\n")
        sections.append(f"Task:                   {task}\n")
        sections.append(f"Data original shape:    {orig_shape}\n")
        sections.append(f"Data final shape:       {final_shape}\n")
        if self.target_info is not None:
            sections.append(f"Target feature:         {self.target_info.name}\n")
        sections.append("\n")
        sections.append(f"Samples dropped due to NaN target: {n_drop}\n")
        sections.append(f"Indicator variables added for continuous NaNs: {n_ind}\n\n")
        sections.append("# Processing Times\n\n")
        sections.append(
            DataFrame(
                data=(np.array(times) * 1000).round(),
                columns=["runtime (ms)"],
                index=Series(name="computation", data=funcs),
            ).to_markdown()
        )

        return "".join(sections)

    def to_json(self, path: Path) -> None:
        path.write_text(str(jsonpickle.encode(self)), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> Optional[PreparationInfo]:
        content = path.read_text()
        if content.strip().replace("\n", "") == "":
            return None
        return cast(PreparationInfo, jsonpickle.decode(content))


def viable_subsample(
    df: DataFrame,
    target: Series,
    n_sub: int = 2000,
    rng: Optional[Generator] = None,
) -> ndarray:
    rng = rng or np.random.default_rng()
    unqs, cnts = np.unique(target, return_counts=True)
    n_min = N_CAT_LEVEL_MIN
    idx_count = cnts < n_min
    idx_final = np.arange(len(target))
    drop_vals = unqs[idx_count]
    unqs = unqs[~idx_count]
    idx_keep = ~target.isin(drop_vals)
    X: ndarray
    y: ndarray
    idx_final = idx_final[idx_keep]
    X = df.copy(deep=True).values[idx_keep]
    y = target.copy(deep=True).to_numpy()[idx_keep]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Keep fail"

    # shuffle once to allow getting random first n of each class later
    idx_shuffle = rng.permutation(len(y))
    idx_final = idx_final[idx_shuffle]
    X = X[idx_shuffle]
    y = y[idx_shuffle]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Shuffle fail"
    #
    idx = np.argsort(y)
    idx_final = idx_final[idx]
    X = X[idx]
    y = y[idx]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Argsort fail"
    # this gives e.g. stops[i]:stops[i+1] are class i
    stops = np.searchsorted(y, unqs)
    cls_idxs = []
    for i in range(len(stops) - 1):
        start, stop = stops[i], stops[i + 1]
        shuf = rng.permutation(stop - start)
        cls_idxs.append(np.arange(start, stop)[shuf][:n_min])
    shuf = rng.permutation(len(y) - stops[-1])
    cls_idxs.append(np.arange(stops[-1], len(y))[shuf][:n_min])

    idx_required = np.concatenate(cls_idxs).ravel()
    idx_final_req = idx_final[idx_required]
    y_req = y[idx_required]
    assert np.bincount(y_req).min() >= N_CAT_LEVEL_MIN, "collect fail"

    n_remain = n_sub - len(y_req)
    if n_remain <= 0:
        return idx_final_req

    idx_remain = np.ones_like(y, dtype=bool)
    idx_remain[idx_required] = False
    idx_final_rem = idx_final[idx_remain]
    X_remain = X[idx_remain]
    n_remain = min(n_remain, len(y))
    if n_remain >= len(y):
        idx_full = np.concatenate([idx_final_req, idx_final_rem])
        assert np.bincount(target[idx_full]).min() >= N_CAT_LEVEL_MIN, "remain fail"
        return idx_full
    else:
        smax = 2**32 - 1
        if hasattr(rng.bit_generator, "seed_seq"):
            ent = rng.bit_generator.seed_seq.entropy  # type: ignore
        elif hasattr(rng.bit_generator, "_seed_seq"):
            ent = rng.bit_generator._seed_seq.entropy  # type: ignore
        else:
            ent = np.random.randint(1, smax)
        while ent > smax:
            ent //= 2

        ss = ShuffleSplit(
            n_splits=1,
            train_size=n_remain,
            random_state=ent,
        )
        idx = next(ss.split(X_remain))[0]
        idx_final_strat = idx_final_rem[idx]
        idx_full = np.concatenate([idx_final_req, idx_final_strat])
        assert np.bincount(target[idx_full]).min() >= N_CAT_LEVEL_MIN, "concat fail"
        return idx_full


class PreparedData:
    def __init__(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        groups: Optional[Series],
        is_classification: Optional[bool] = None,
        X_cont: Optional[DataFrame] = None,
        X_cat: Optional[DataFrame] = None,
        labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]] = None,
        ix_train: Optional[ndarray] = None,
        ix_tests: Optional[list[ndarray]] = None,
        tests_method: Optional[ValidationMethod] = ValidationMethod.List,
        inspection: Optional[InspectionResults] = None,
        info: Optional[PreparationInfo] = None,
        phase: Optional[Literal["train", "test"]] = None,
        validate: bool = True,
    ) -> None:
        # Attempt to automatically infer classification problem based on y.dtype
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        if is_classification is None:
            kind = y.dtypes.iloc[0].kind if isinstance(y, DataFrame) else y.dtype.kind
            if kind == "O":
                if isinstance(y, DataFrame):
                    raise ValueError(
                        "Got object dtype for multi-target y. Please encode targets "
                        "before constructing PreparedData."
                    )
                # run cleaning.encode_target with X, y
                warn(
                    f"Found 'object' dtype for argument y (name='{y.name}'). Attempting to "
                    "automatically label encode. If this is undesirable, ensure that your "
                    "target `y` has the proper dtype, e.g. `y = y.astype(np.float64)` if "
                    "regression, or `y = y.astype(np.int64)` if classification. "
                )
                X, y, labels, _, _ = encode_target(X, y, ix_train, ix_tests, _warn=True)
                self.is_classification = True
            else:
                kinds = {"f": False, "i": True, "b": True, "u": True}
                if kind not in kinds:
                    raise ValueError(
                        f"Got argument for parameter `y` with unsupported data type: {y.dtype}"
                    )
                tname = (
                    "floating point"
                    if kind == "f"
                    else "signed/unsigned integer or boolean type"
                )
                self.is_classification = kinds[kind]
                warn(
                    f"Argument `is_classification` left at default. Inferred "
                    f"`is_classification={self.is_classification}`, since y is {tname}. "
                    "If this is incorrect, or to silence this warning, specify the value for "
                    f"`is_classification`.\ny={y}"
                )
        else:
            self.is_classification = is_classification
        self.inspection: Optional[InspectionResults] = inspection
        self.info: Optional[PreparationInfo] = info
        self.files: PrepFiles = PrepFiles()
        self.phase = phase

        if validate:
            X, X_cont, X_cat, y = self.validate(X, X_cont, X_cat, y)

        if groups is not None:
            if len(groups) != len(X):
                raise ValueError(
                    f"Grouping data number of samples ({len(groups)}) does not "
                    f"match number of samples in processed data ({len(X)})"
                )
            groups = groups.copy()
            groups.reset_index(drop=True, inplace=True)
            groups.index = X.index.copy(deep=True)

        self.X = self.rename_cols(X)
        self.X_cont: Optional[DataFrame] = (
            None if X_cont is None else self.rename_cols(X_cont)
        )
        self.X_cat: Optional[DataFrame] = (
            None if X_cat is None else self.rename_cols(X_cat)
        )
        if isinstance(y, DataFrame):
            self.y: Union[Series, DataFrame] = y
            self.target_cols = y.columns.tolist()
            self.target = self.target_cols[0] if len(self.target_cols) > 0 else ""
        else:
            name = "target" if y.name is None else str(y.name)
            if y.name != name:
                y = y.rename(name)
            self.y = y
            self.target_cols = [name]
            self.target = name
        self.labels = labels or {}
        self.split_labels = (
            self.labels
            if isinstance(self.labels, dict)
            and all(isinstance(k, int) for k in self.labels.keys())
            else None
        )
        self.groups: Optional[Series] = groups

        self.ix_train: Optional[ndarray] = ix_train
        self.ix_tests: list[ndarray] = ix_tests or []
        self.tests_method = tests_method

    @property
    def num_classes(self) -> int:
        if not self.is_classification:
            return 1
        if isinstance(self.y, DataFrame):
            return int(max(self.y[col].nunique() for col in self.y.columns))
        return len(np.unique(self.y))

    def get_splits(
        self, test_size: Union[int, float] = 0.4, seed: int | None = SEED
    ) -> Union[
        list[tuple[PreparedData, PreparedData]],
        Gen[tuple[PreparedData, PreparedData], None, None],
    ]:
        if isinstance(test_size, int):
            test_size = test_size / len(self.y)
        train_size = 1 - test_size

        if self.ix_train is None or self.ix_tests is None:
            # Old df-analyze behaviour prior to multiple test sets
            yield (self.split(train_size=train_size, seed=seed))
            return

        prep_train = self.subsample(self.ix_train)

        if self.tests_method is ValidationMethod.List:
            for ix_test in self.ix_tests:
                yield (prep_train, self.subsample(ix_test))
            return

        if self.tests_method is not ValidationMethod.LODO:
            raise ValueError(f"Impossible! Invalid ValidationMethod: {self.tests_method}")

        # now construct the indices needed
        ix_all = [self.ix_train, *self.ix_tests]
        ix_pairs = []
        for i, ix in enumerate(ix_all):
            ix_train = ix
            ix_tests = ix_all[:i] + ix_all[i + 1 :]
            ix_test = np.concatenate(ix_tests)
            ix_pairs.append((ix_train, ix_test))

        for ix_train, ix_test in ix_pairs:
            yield (self.subsample(ix_train), self.subsample(ix_test))

    def split(
        self,
        train_size: Union[int, float] = 0.6,
        seed: int | None = SEED,
    ) -> tuple[PreparedData, PreparedData]:
        y = self.y.copy()
        if self.groups is None:
            if self.is_classification:
                ss = StratifiedShuffleSplit(
                    train_size=train_size, n_splits=1, random_state=seed
                )
                if isinstance(y, DataFrame):
                    split_y = y_split_label(y)
                    idx_train, idx_test = next(ss.split(split_y.to_frame(), split_y))
                else:
                    idx_train, idx_test = next(ss.split(y.to_frame(), y))
            else:
                ss = ShuffleSplit(train_size=train_size, n_splits=1, random_state=seed)
                idx_train, idx_test = next(ss.split(self.X))
        else:
            ss = ApproximateStratifiedGroupSplit(
                train_size=train_size,
                is_classification=self.is_classification,
                grouped=self.groups is not None,
                labels=self.split_labels,
                seed=seed,
                warn_on_fallback=True,
                warn_on_large_size_diff=True,
                df_analyze_phase="Initial holdout splitting",
            )
            if self.is_classification and isinstance(y, DataFrame):
                split_y = y_split_label(y)
                (idx_train, idx_test), group_fail = ss.split(
                    split_y.to_frame(), split_y, self.groups
                )
            else:
                split_y = y.mean(axis=1) if isinstance(y, DataFrame) else y
                (idx_train, idx_test), group_fail = ss.split(
                    split_y.to_frame(), split_y, self.groups
                )

        prep_train = self.subsample(idx_train)
        prep_train.phase = "train"

        prep_test = self.subsample(idx_test)
        prep_test.phase = "test"

        return prep_train, prep_test

    def subsample(self, idx: ndarray) -> PreparedData:
        try:
            X_sub = self.X.iloc[idx].reset_index(drop=True)
        except IndexError as e:
            raise IndexError(
                f"Couldn't subsample prepared data. Data shape: {self.X.shape}, "
                f"and subsampling indices range: [{idx.min()}, {idx.max()}]"
            ) from e
        X_cont, X_cat = self.X_cont, self.X_cat
        groups = None if self.groups is None else self.groups.iloc[idx]
        if self.info is not None:
            info_sub = deepcopy(self.info)
            info_sub.final_shape = X_sub.shape
        else:
            info_sub = None
        return PreparedData(
            X=X_sub,
            X_cont=None if X_cont is None else X_cont.iloc[idx].reset_index(drop=True),
            X_cat=None if X_cat is None else X_cat.iloc[idx].reset_index(drop=True),
            y=self.y.iloc[idx].copy().reset_index(drop=True),
            groups=groups,
            labels=self.labels,
            inspection=self.inspection,
            info=info_sub,
            is_classification=self.is_classification,
            validate=True,
        )

    def for_target(self, target: str) -> PreparedData:
        if not isinstance(self.y, DataFrame):
            if self.y.name != target:
                raise ValueError(
                    f"Single-target PreparedData has y.name={self.y.name}, "
                    f"requested={target}"
                )
            return self

        if target not in self.y.columns:
            raise ValueError(f"Target '{target}' not found in PreparedData.y.")

        y_col = self.y[target].copy()
        labels: Optional[dict[int, str]]
        if (
            isinstance(self.labels, dict)
            and target in self.labels
            and isinstance(self.labels[target], dict)
        ):
            labels = cast(dict[int, str], self.labels[target])
        else:
            labels = None

        return PreparedData(
            X=self.X,
            X_cont=self.X_cont,
            X_cat=self.X_cat,
            y=y_col,
            groups=self.groups,
            labels=labels,
            ix_train=self.ix_train,
            ix_tests=self.ix_tests,
            tests_method=self.tests_method,
            inspection=self.inspection,
            info=self.info,
            is_classification=self.is_classification,
            phase=self.phase,
            validate=True,
        )

    def representative_subsample(
        self,
        n_sub: int = UNIVARIATE_PRED_MAX_N_SAMPLES,
        rng: Optional[Generator] = None,
    ) -> tuple[PreparedData, ndarray]:
        rng = rng or np.random.default_rng()
        X, X_cont, X_cat = self.X, self.X_cont, self.X_cat
        y = self.y

        g = self.groups
        if g is not None:
            warn(
                "Grouping is currently NOT implemented for `representative_subsample`. "
                "The grouping variable will be ignored when creating a minimal viable "
                "subsample. This may introduce a significant bias in the unviariate "
                "predictions if samples from the same group end up distributed across "
                "subsequent training and test splits. However, this bias will be limited "
                "to the univariate predictive stats. Grouping is handled properly in all "
                "subsequent df-analyze splitting procedures."
            )

        if len(X) <= UNIVARIATE_PRED_MAX_N_SAMPLES:
            return self, np.arange(len(X), dtype=np.int64)

        if self.is_classification:
            if isinstance(y, DataFrame):
                split_y = y_split_label(y)
                split_codes = split_y.astype("category").cat.codes
                idx = viable_subsample(df=X, target=split_codes, n_sub=n_sub, rng=rng)
            else:
                idx = viable_subsample(df=X, target=y, n_sub=n_sub, rng=rng)
            X = X.iloc[idx]
            if X_cont is not None:
                X_cont = X_cont.iloc[idx]
            if X_cat is not None:
                X_cat = X_cat.iloc[idx]
            if g is not None:
                g = g.iloc[idx]
            y = y.iloc[idx]
        else:
            kb = KBinsDiscretizer(n_bins=5, encode="ordinal")
            if isinstance(y, DataFrame):
                y_vals = y.mean(axis=1)
            else:
                y_vals = y
            strat = kb.fit_transform(y_vals.to_numpy().reshape(-1, 1))
            n_train = UNIVARIATE_PRED_MAX_N_SAMPLES
            ss = StratifiedShuffleSplit(n_splits=1, train_size=n_train)
            idx = next(ss.split(strat, strat))[0]
            X = cast(DataFrame, self.X.iloc[idx, :].copy(deep=True))
            y = self.y.loc[idx].copy(deep=True)
            if X_cont is not None:
                X_cont = X_cont.loc[idx, :].copy(deep=True)
            if X_cat is not None:
                X_cat = X_cat.loc[idx, :].copy(deep=True)
            if g is not None:
                g = g.iloc[idx]

        return PreparedData(
            X=X,
            X_cont=X_cont,
            X_cat=X_cat,
            y=y,
            groups=g,
            labels=self.labels,
            inspection=self.inspection,
            info=self.info,
            is_classification=self.is_classification,
        ), idx

    def validate(
        self,
        X: DataFrame,
        X_cont: Optional[DataFrame],
        X_cat: Optional[DataFrame],
        y: Union[Series, DataFrame],
    ) -> tuple[
        DataFrame,
        Optional[DataFrame],
        Optional[DataFrame],
        Union[Series, DataFrame],
    ]:
        n_samples = len(X)
        if X_cont is not None and len(X_cont) != n_samples:
            raise ValueError(
                f"Continuous data number of samples ({len(X_cont)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if X_cat is not None and len(X_cat) != n_samples:
            raise ValueError(
                f"Categorical data number of samples ({len(X_cat)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if len(y) != n_samples:
            raise ValueError(
                f"Target number of samples ({len(X)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )

        if self.is_classification:
            if isinstance(y, DataFrame):
                undersampled_counts: list[str] = []
                for col in y.columns:
                    cnts = y[col].value_counts()
                    if cnts.empty:
                        raise ValueError(
                            f"Target '{col}' has no valid (non-missing) samples."
                        )
                    if cnts.min() <= N_MULTITARGET_LEVEL_MIN:
                        df = DataFrame(
                            index=pd.Index(data=cnts.index, name="Target Level"),
                            columns=["Count"],
                            data=cnts.values,
                        )
                        info = df.to_markdown(tablefmt="simple")
                        undersampled_counts.append(f"Target '{col}' counts:\n\n{info}")
                if len(undersampled_counts) > 0:
                    info = "\n\n".join(undersampled_counts)
                    warn(
                        "One or more multi-target classification targets have "
                        "undersampled levels. This means that one or more target levels "
                        f"(classes) have fewer than or equal to {N_MULTITARGET_LEVEL_MIN} samples. "
                        "Preprocessing will continue, but splitting and classification "
                        "metrics for these levels may be unstable. This may also cause "
                        "downstream CV splitting to fail (especially when class counts "
                        "are below the number of folds).\n\n"
                        "Observed target level counts:\n\n"
                        f"{info}"
                    )
            elif np.bincount(y).min() < N_TARG_LEVEL_MIN:
                unqs, cnts = np.unique(y.to_numpy(), return_counts=True)
                df = DataFrame(
                    index=pd.Index(data=unqs, name="Target Level"),
                    columns=["Count"],
                    data=cnts,
                )
                info = df.to_markdown(tablefmt="simple")
                raise ValueError(
                    f"Target '{y.name}' has undersampled levels. This means that one or "
                    f"more of the target levels (classes) has less than {N_TARG_LEVEL_MIN} "
                    "samples, either before or after splitting into a holdout set. This is "
                    "simply far too few samples for meaningful generalization or stable "
                    "performance estimates, and means your data is far too small to use for "
                    "automated machine learning via df-analyze.\n\n"
                    "Observed target level counts:\n\n"
                    f"{info}"
                )

        # Handle some BS due to stupid Pandas index behaviour
        X.reset_index(drop=True, inplace=True)
        if X_cont is not None:
            X_cont.index = X.index.copy(deep=True)
        if X_cat is not None:
            X_cat.index = X.index.copy(deep=True)
        y.index = X.index.copy(deep=True)
        return X, X_cont, X_cat, y

    def rename_cols(self, df: DataFrame) -> DataFrame:
        # see https://github.com/microsoft/LightGBM/issues/6202#issuecomment-1820286842
        # for LightGBM disallowed characters
        df = df.rename(
            columns=lambda col: re.sub(r"[\[\]\{\},:\"]+", "", str(col)),
        )
        dupe_cols = set(df.columns[df.columns.duplicated()])
        counts = {col: 0 for col in dupe_cols}
        new_cols = []
        for col in df.columns:
            if col not in dupe_cols:
                new_cols.append(col)
                continue
            if counts[col] == 0:  # leave name unchanged
                new_cols.append(col)
            else:
                new_cols.append(f"{col}_{counts[col]}")
            counts[col] += 1
        df.columns = new_cols
        return df

    def describe_features(
        self,
    ) -> tuple[Optional[DataFrame], Optional[DataFrame], DataFrame]:
        if isinstance(self.y, DataFrame):
            raise ValueError(
                "describe_features does not support multi-target data. "
                "Call for_target(...) first."
            )
        return describe_all_features(
            continuous=self.X_cont,
            categoricals=self.X_cat,
            target=self.y,
            is_classification=self.is_classification,
        )

    def to_markdown(self) -> Optional[str]:
        if self.info is not None:
            return self.info.to_markdown()
        warn("No preparation info found, no Markdown report to make")

    def save_raw(self, root: Path) -> None:
        try:
            self.X.to_parquet(root / self.files.X_raw)
            if self.X_cont is not None:
                self.X_cont.to_parquet(root / self.files.X_cont_raw)
            if self.X_cat is not None:
                self.X_cat.to_parquet(root / self.files.X_cat_raw)
            if isinstance(self.y, DataFrame):
                self.y.to_parquet(root / self.files.y_raw)
            else:
                self.y.to_frame().to_parquet(root / self.files.y_raw)
            if self.groups is not None:
                self.groups.to_frame().to_parquet(root / self.files.g_raw)
            if self.labels is not None:
                if all(isinstance(v, dict) for v in self.labels.values()):
                    DataFrame(self.labels).to_parquet(root / self.files.labels)
                else:
                    Series(self.labels).to_frame().to_parquet(root / self.files.labels)
            if self.info is not None:
                self.info.to_json(root / self.files.info)
            if self.ix_train is not None:
                js = json.dumps(self.ix_train.tolist())
                (root / self.files.ix_train).write_text(js)
            if self.ix_tests is not None:
                obj = [ix_test.tolist() for ix_test in self.ix_tests]
                js = json.dumps(obj)
                (root / self.files.ix_tests).write_text(js)
        except Exception as e:
            warn(
                f"Exception while saving {self.__class__.__name__} to {root}."
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    @staticmethod
    def from_saved(root: Path, inspection: InspectionResults) -> PreparedData:
        files = PrepFiles()
        X = pd.read_parquet(root / files.X_raw)
        X_cont = pd.read_parquet(root / files.X_cont_raw)
        X_cat = pd.read_parquet(root / files.X_cat_raw)
        y_raw = pd.read_parquet(root / files.y_raw)
        gfile = root / files.g_raw
        g_raw = pd.read_parquet(gfile) if gfile.exists() else None
        y: Union[Series, DataFrame]
        if y_raw.shape[1] == 1:
            y = Series(name=y_raw.columns[0], data=y_raw.values.ravel(), index=y_raw.index)
        else:
            y = y_raw
        g = (
            Series(name=g_raw.columns[0], data=g_raw.values.ravel(), index=g_raw.index)
            if g_raw is not None
            else None
        )
        labelpath = root / files.labels
        labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]]
        if labelpath.exists():
            labels_raw = pd.read_parquet(labelpath)
            if labels_raw.shape[1] == 1:
                labels = cast(dict[int, str], labels_raw.iloc[:, 0].dropna().to_dict())
            else:
                labels = {
                    str(col): cast(dict[int, str], labels_raw[col].dropna().to_dict())
                    for col in labels_raw.columns
                }
        else:
            labels = None
        info = PreparationInfo.from_json(root / files.info)
        if info is not None:
            is_cls = info.is_classification
        else:
            is_cls = None

        ix_train_path = root / files.ix_train
        if ix_train_path.exists():
            obj = json.loads(ix_train_path.read_text())
            ix_train = np.array(obj, dtype=np.int64)
        else:
            ix_train = None

        ix_tests_path = root / files.ix_tests
        if ix_tests_path.exists():
            obj: list[list[int]] = json.loads(ix_tests_path.read_text())
            ix_tests = [np.array(ix_test, dtype=np.int64) for ix_test in obj]
        else:
            ix_tests = None

        return PreparedData(
            X=X,
            X_cont=X_cont,
            X_cat=X_cat,
            y=y,
            groups=g,
            labels=labels,
            inspection=inspection,
            info=info,
            is_classification=is_cls,
            ix_train=ix_train,
            ix_tests=ix_tests,
        )


def prepare_target(
    df: DataFrame,
    target: str,
    is_classification: bool,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
    _warn: bool = True,
) -> tuple[
    DataFrame,
    Series,
    Optional[dict[int, str]],
    Optional[ndarray],
    Optional[list[ndarray]],
]:
    y = df[target]
    df = df.drop(columns=target)
    if is_classification:
        df, y, labels, ix_train, ix_tests = encode_target(
            df, y, ix_train, ix_tests, _warn=_warn
        )
    else:
        labels = None
        df, y, ix_train, ix_tests = clean_regression_target(df, y, ix_train, ix_tests)
    return df, y, labels, ix_train, ix_tests


def prepare_data(
    df: DataFrame,
    target: TargetSpec,
    grouper: Optional[str],
    results: InspectionResults,
    is_classification: bool,
    ix_train: Optional[ndarray] = None,
    ix_tests: Optional[list[ndarray]] = None,
    tests_method: Optional[ValidationMethod] = ValidationMethod.List,
    _warn: bool = True,
    *,
    nan_handling: NanHandling = NanHandling.Median,
    robust: bool = True,
) -> PreparedData:
    """
    Returns
    -------
    X_encoded: DataFrame
        All encoded and processed predictors.

    X_cat: DataFrame
        The categorical variables remaining after processing (no encoding,
        for univariate metrics and the like).

    X_cont: DataFrame
        The continues variables remaining after processing (no encoding,
        for univariate metrics and the like).

    y: Series
        The regression or classification target, also encoded.

    info: dict[str, str]
        Other information regarding warnings and cleaning effects.

    """
    times: dict[str, float] = {}
    timer = partial(timed, times=times)
    target_cols = as_target_list(target)
    orig_shape = (df.shape[0], df.shape[1] - len(target_cols))

    df = timer(unify_nans)(df)
    df = timer(convert_categoricals)(df=df, target=target_cols, grouper=grouper)
    info: Optional[Union[RegTargetInfo, ClsTargetInfo]] = None
    n_targ_drop = 0
    labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]]
    if is_classification:
        if len(target_cols) > 1:
            orig_n = df.shape[0]
            df_no_y, y, labels, ix_train, ix_tests = timer(encode_targets)(
                df, target_cols, ix_train, ix_tests, _warn=_warn
            )
            n_targ_drop = orig_n - df_no_y.shape[0]
            df = pd.concat([df_no_y, y], axis=1)
        else:
            df, n_targ_drop, ix_train, ix_tests = timer(drop_target_nans)(
                df, target_cols[0], ix_train, ix_tests
            )
            df, y, labels, ix_train, ix_tests = timer(encode_target)(
                df, df[target_cols[0]], ix_train, ix_tests
            )
    else:
        if len(target_cols) > 1:
            y_df_raw = df[target_cols].copy()
            keep_mask = ~y_df_raw.isna().any(axis=1)
            n_targ_drop = int((~keep_mask).sum())
            ix_train, ix_tests = reindex(keep_mask, ix_train, ix_tests)
            df = df.loc[keep_mask].reset_index(drop=True)
            y_df_raw = y_df_raw.loc[keep_mask].reset_index(drop=True)
            y = DataFrame(index=y_df_raw.index)
            for col in target_cols:
                scaled = (
                    RobustScaler(quantile_range=(2.5, 97.5))
                    .fit_transform(y_df_raw[[col]])
                    .ravel()
                )
                y[col] = scaled
            df[target_cols] = y
        else:
            df, n_targ_drop, ix_train, ix_tests = timer(drop_target_nans)(
                df, target_cols[0], ix_train, ix_tests
            )
            df, y, ix_train, ix_tests = timer(clean_regression_target)(
                df, df[target_cols[0]], ix_train, ix_tests
            )
        labels = None

    for i, target_col in enumerate(target_cols):
        target_info = timer(inspect_target)(
            df, target_col, is_classification=is_classification
        )
        if len(target_cols) == 1 and i == 0:
            info = target_info
    df = timer(drop_unusable)(df, results, target=target_cols, _warn=_warn)
    df, X_cont, n_ind_added = handle_continuous_nans(
        df=df,
        target=target_cols,
        grouper=grouper,
        results=results,
        nans=nan_handling,
    )
    X_cont = normalize_continuous(X_cont, robust=robust)
    # Keep the model matrix consistent with the continuous feature analyses.
    # Previously only X_cont was scaled, while final models received raw df.
    if not X_cont.empty:
        df.loc[:, X_cont.columns] = X_cont.to_numpy()

    df = timer(deflate_categoricals)(df, grouper, results, _warn=_warn)
    df, X_cat = timer(encode_categoricals)(
        df=df,
        target=target_cols,
        grouper=grouper,
        results=results,
        warn_explosion=_warn,
    )

    X = df.drop(columns=target_cols).reset_index(drop=True)
    if grouper is not None:
        g = X[grouper]
        X = X.drop(columns=grouper)
    else:
        g = None
    return PreparedData(
        X=X,
        X_cont=X_cont,
        X_cat=X_cat,
        y=y,
        groups=g,
        labels=labels,
        ix_train=ix_train,
        ix_tests=ix_tests,
        tests_method=tests_method,
        info=PreparationInfo(
            original_shape=orig_shape,
            final_shape=X.shape,
            n_samples_dropped_via_target_NaNs=n_targ_drop,
            n_cont_indicator_added=n_ind_added,
            target_info=info,
            runtimes=times,
            is_classification=is_classification,
        ),
        inspection=results,
        is_classification=is_classification,
    )
