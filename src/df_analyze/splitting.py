from __future__ import annotations

from typing import Any, Mapping, Optional, Type, Union
from warnings import catch_warnings, filterwarnings, warn

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Index, Series
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit as StratSplit,
)

from df_analyze._constants import (
    N_MULTITARGET_LEVEL_MIN,
    N_TARG_LEVEL_MIN_TEST_INTERNAL,
    N_TARG_LEVEL_MIN_TRAIN_INTERNAL,
    SEED,
)

AnyKFold = Union[
    Type[KFold],
    Type[GroupKFold],
    Type[StratifiedKFold],
    Type[StratifiedGroupKFold],
]

AnySplitter = Union[
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
]

APPROXIMATE_GROUP_SPLIT_DIFF_WARN_THRESHOLD = 0.1


def y_split_label(
    y_df: DataFrame,
    min_count: int = N_MULTITARGET_LEVEL_MIN,
    warn_on_fallback: bool = True,
) -> Series:
    if y_df.shape[1] == 0:
        raise ValueError("Cannot split with empty target columns.")
    cols = list(y_df.columns)
    while True:
        label = y_df[cols].astype(str).agg("__".join, axis=1)
        cnts = label.value_counts()
        n_unique = cnts.shape[0]
        max_unique = max(1, len(y_df) // min_count)
        if (cnts.min() >= min_count) and (n_unique <= max_unique):
            return label
        if len(cols) == 1:
            return label
        nunique = y_df[cols].nunique(dropna=False)
        drop_col = nunique.idxmax()
        cols.remove(drop_col)
        if warn_on_fallback:
            warn(
                "Multi-target stratification produced undersampled target combinations. "
                f"Dropping target column '{drop_col}' to improve split feasibility."
            )


class OmniKFold:
    def __init__(
        self,
        n_splits: int = 5,
        is_classification: bool = True,
        grouped: bool = False,
        labels: Optional[dict[int, str]] = None,
        shuffle: bool = False,
        seed: Optional[int] = SEED,
        warn_on_fallback: bool = True,
        df_analyze_phase: Optional[str] = None,
    ) -> None:
        self.n_splits: int = n_splits
        self.is_cls = self.is_classification = is_classification
        labels = labels or {}
        self.labels: dict[str, str] = {
            str(encoded): str(orig) for encoded, orig in labels.items()
        }
        self.grouped: bool = grouped
        self.shuffle: bool = shuffle
        self.seed: Optional[int] = seed
        self.warn: bool = warn_on_fallback
        self.kf: AnyKFold
        self.kf_fallback: Union[Type[StratifiedKFold], Type[KFold]]
        self.df_analyze_phase: Optional[str] = df_analyze_phase
        self.phase_info = (
            ""
            if self.df_analyze_phase is None
            else f"Splitting error occurred at df-analyze phase: {self.df_analyze_phase}"
        )

        if self.is_cls:
            self.kf = StratifiedGroupKFold if self.grouped else StratifiedKFold
            self.kf_fallback = StratifiedKFold
        else:
            self.kf = GroupKFold if self.grouped else KFold
            self.kf_fallback = KFold  # regression doesn't NEED stratification

        self.no_fallback = self.kf is self.kf_fallback

    def split(
        self,
        X_train: DataFrame,
        y_train: Series,
        g_train: Optional[Series] = None,
    ) -> tuple[list[tuple[ndarray, ndarray]], bool]:
        """
        Returns
        -------
        splits: list[tuple[ndarray, ndarray]]
            NumPy array of split indices (as int64)

        did_fail: bool
            Boolean indicating if the initial grouping split attempt failed, so
            that subsequent split attempts can disable the warnings.
        """
        if self.grouped and g_train is None:
            raise ValueError(
                f"Got grouping data for splitting, but {self.__class__.__name__} "
                f"was initialized with `grouped=False`. {self.phase_info}"
            )

        y_cnts = np.unique(y_train.apply(str), return_counts=True)[1]
        if len(y_cnts) <= 1:
            raise RuntimeError(
                "Split function recieved target variable which appears to be "
                "constant. This should have been caught much earlier in df-analyze "
                f"and so is likely a bug in df-analyze code. df-analyze phase: {self.phase_info}\n"
                f"Target variable:\n{y_train}"
            )
        rng = np.random.default_rng(seed=self.seed)
        y: Series
        ix_shuf: Optional[ndarray] = None
        if self.shuffle:
            n = len(X_train)
            ix_shuf = rng.permutation(n)
            X = X_train.iloc[ix_shuf]
            y = y_train.iloc[ix_shuf]
            g = None if g_train is None else g_train.iloc[ix_shuf]
        else:
            X, y, g = X_train, y_train, g_train

        def to_original(ix: ndarray) -> ndarray:
            if ix_shuf is None:
                return ix
            return ix_shuf[ix]

        # First attempt: try doing grouped stratified and check target level counts
        kf_args = self._kf_args(kf=self.kf)
        cv = self.kf(**kf_args)
        if isinstance(cv, StratifiedKFold):
            cv_args = dict(X=X, y=y)  # groups ignored, causes a warning
        else:
            cv_args = dict(X=X, y=y, groups=g)
        splits: list[tuple[ndarray, ndarray]] = []
        ix_tr: ndarray
        ix_t: ndarray
        initial_split_fail = False
        for ix_tr, ix_t in cv.split(**cv_args):  # type: ignore
            if self.is_cls and self._split_fail(y, ix_tr=ix_tr, ix_t=ix_t):
                initial_split_fail = True
                splits = []
                if self.warn:
                    warn(
                        "Could not perform a grouped, stratified split of the target. "
                        f"Attempting to fall back to {self.kf_fallback.__name__}.\n\n"
                        "**NOTE**: This means instances of the same group will be "
                        "allowed to appear in both the train and test splits of each "
                        "fold, and thus that performance estimates may appear much "
                        "more favourable, due the bias and/or contamination introduced "
                        "if samples within groups are high in similarity."
                    )
                break
            splits.append((to_original(ix_tr), to_original(ix_t)))
        if not initial_split_fail:
            return splits, initial_split_fail

        if self.no_fallback:
            raise self._informative_error(y_train)
        if self.phase_info is None:
            self.phase_info = ""
        self.phase_info = f"{self.phase_info} - Splitting fallback attempt."

        # fallback attempt
        kf_args = self._kf_args(kf=self.kf_fallback)
        cv = self.kf_fallback(**kf_args)
        if isinstance(cv, StratifiedKFold):
            cv_args = dict(X=X, y=y)  # groups ignored, causes a warning
        else:
            cv_args = dict(X=X, y=y, groups=g)
        for ix_tr, ix_t in cv.split(**cv_args):  # type: ignore
            if self.is_cls and self._split_fail(y, ix_tr=ix_tr, ix_t=ix_t):
                # not worried about the regression fallback case
                raise self._informative_error(y_train)
            splits.append((to_original(ix_tr), to_original(ix_t)))

        return splits, initial_split_fail

    def _kf_args(self, kf: AnyKFold) -> dict[str, Any]:
        """

                    KFold(n_splits=5, *, shuffle=False, random_state=None)
          StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
        StratifiedGroupKFold(n_splits=5, shuffle=False, random_state=None)
                  GroupKFold(n_splits=5)

                       KFold.split(X, y=None, groups=None)
             StratifiedKFold.split(X, y,      groups=None)
        StratifiedGroupKFold.split(X, y=None, groups=None)
                  GroupKFold.split(X, y=None, groups=None)
        """
        # NOTE: We always do shuffle=False, random_state=None since
        # shuffling is handled by us and seeded there

        kf_args = dict(n_splits=self.n_splits, shuffle=False, random_state=None)
        grp_args = dict(
            n_splits=self.n_splits,
            shuffle=False,  # VERY IMPORTANT: shuffle is bugged!
            # https://github.com/scikit-learn/scikit-learn/issues/24656
        )
        args = grp_args if kf is GroupKFold else kf_args
        return args

    def _split_fail(self, y: Series, ix_tr: ndarray, ix_t: ndarray) -> bool:
        if not self.is_cls:
            return False
        y_str = y.apply(str).to_numpy()  # .astype(str) unreliable
        y_tr = y_str[ix_tr]
        y_t = y_str[ix_t]
        tr_cnts = np.unique(y_tr, return_counts=True)[1]
        t_cnts = np.unique(y_t, return_counts=True)[1]
        if len(tr_cnts) <= 1:
            return True  # definitely cannot proceed, degen training set

        # Can proceed, technically, but will get errors for later AUROC and
        # other metrics...
        if len(t_cnts) <= 1:
            return True

        tr_cnts_n_min = tr_cnts.min()
        if tr_cnts_n_min < N_TARG_LEVEL_MIN_TRAIN_INTERNAL:
            return True

        t_cnts_n_min = t_cnts.min()
        if t_cnts_n_min < N_TARG_LEVEL_MIN_TEST_INTERNAL:
            return True

        return False

    def _informative_error(self, y: Series) -> RuntimeError:
        unqs, cnts = np.unique(y.apply(str).to_numpy(), return_counts=True)
        unqs = (
            Series(unqs)
            .apply(lambda lab: self.labels[lab] if lab in self.labels else lab)
            .values
        )

        df = DataFrame(
            index=Index(data=unqs, name="Target Level"),
            columns=["Count"],
            data=cnts,
        )
        info = df.to_markdown(tablefmt="simple")
        kf = self.kf.__name__
        return RuntimeError(
            f"Attempted to split target variable '{y.name}' with {kf}, but found "
            "insufficient samples per target level, i.e. the classification "
            "target has undersampled levels. This means that one or more of "
            f"the target levels (classes) cannot be split in internal 5-fold."
            f"That is, at least one cross-validation training split will always "
            f"result in less than {N_TARG_LEVEL_MIN_TRAIN_INTERNAL} samples per "
            "level, or that at least one cross-validation test split will always "
            f"have less than {N_TARG_LEVEL_MIN_TEST_INTERNAL} samples per target "
            f"level. df-analyze phase: {self.phase_info}\n\n"
            "The current data thus has target classes with insufficient data for "
            f"meaningful generalization or stable performance estimates via automated "
            "machine learning via df-analyze. Consider removing instances of these "
            "rare target levels, or collect more data.\n\n"
            "Observed target level counts:\n\n"
            f"{info}"
        )


class ApproximateStratifiedGroupSplit:
    def __init__(
        self,
        train_size: float = 0.6,
        is_classification: bool = True,
        grouped: bool = False,
        labels: dict[int, str] | None = None,
        shuffle: bool = False,
        seed: int | None = SEED,
        warn_on_fallback: bool = True,
        warn_on_large_size_diff: bool = True,
        df_analyze_phase: str | None = None,
    ) -> None:
        self.desired_train_size = train_size
        self.warn_on_large_size_diff: bool = warn_on_large_size_diff
        n_splits = int(1 / (1 - train_size))
        if n_splits == 1:
            n_splits = 2
        self.kf = OmniKFold(
            n_splits=n_splits,
            is_classification=is_classification,
            grouped=grouped,
            labels=labels,
            shuffle=shuffle,
            seed=seed,
            warn_on_fallback=warn_on_fallback,
            df_analyze_phase=df_analyze_phase,
        )

    def split(
        self, X_train: DataFrame, y_train: Series, g_train: Series | None = None
    ) -> tuple[tuple[ndarray, ndarray], bool]:
        with catch_warnings():
            filterwarnings(
                "ignore", message="The least populated class in y", category=UserWarning
            )
            splits, group_fail = self.kf.split(X_train, y_train, g_train)

        # TODO:
        desired = self.desired_train_size
        (ix_tr, ix_t), d_min = self._get_best_split(y_train, splits)

        if self.kf._split_fail(y=y_train, ix_tr=ix_tr, ix_t=ix_t):
            if self.kf.grouped and self.kf.warn:
                fallback = "StratifiedShuffleSplit" if self.kf.is_cls else "ShuffleSplit"
                warn(
                    "Could not perform a grouped, stratified split of the target. "
                    f"Attempting to fall back to {fallback}.\n\n"
                    "**NOTE**: This means instances of the same group will be "
                    "allowed to appear in both the train and test splits,"
                    "and thus that performance estimates may appear much "
                    "more favourable, due the bias and/or contamination introduced "
                    "if samples within groups are high in similarity."
                )

            ss_args: Mapping = dict(
                train_size=self.desired_train_size, n_splits=1, random_state=self.kf.seed
            )
            ss = StratSplit(**ss_args) if self.kf.is_cls else ShuffleSplit(**ss_args)
            ix_tr, ix_t = next(ss.split(y_train.to_frame(), y_train))
            if self.kf._split_fail(y=y_train, ix_tr=ix_tr, ix_t=ix_t):
                self.kf.phase_info += " - ApproximateStratifiedGroupSplit fallback"
                raise self.kf._informative_error(y_train)

        # d_min = min(d_train, d_test)
        if self.warn_on_large_size_diff and (
            d_min > APPROXIMATE_GROUP_SPLIT_DIFF_WARN_THRESHOLD
        ):
            tst = round(1 - desired, 2)
            warn(
                f"User requested an initial holdout test set size of approximately {tst}, "
                "But the closest approximate Could not generate initial holdout "
                "https://github.com/scikit-learn/scikit-learn/issues/12076#issuecomment-2047948563"
            )

        return (ix_tr, ix_t), group_fail

    def _get_best_split(
        self, y_train: Series, splits: list[tuple[ndarray, ndarray]]
    ) -> tuple[tuple[ndarray, ndarray], float]:
        n = len(y_train)
        desired = self.desired_train_size

        p_splits = [len(splits[i][0]) / n for i in range(len(splits))]
        ds = [abs(p - desired) for p in p_splits]
        ix_min = np.argmin(ds)
        d_min = min(ds)
        ix_tr, ix_t = splits[ix_min]
        return (ix_tr, ix_t), d_min
