"""
Sequential feature selection

NOTE: We are copying this in from sklearn because sklearn folks are too lazy to add any logging to
such a long computation...
"""

import numbers
from typing import Any, Literal, no_type_check

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection._base import SelectorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

# from sklearn.utils._tags import _safe_tags
from df_analyze.sklearn_pasta._tags import _safe_tags


@no_type_check
class SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Transformer that performs Sequential Feature Selection.

    This Sequential Feature Selector adds (forward selection) or
    removes (backward selection) features to form a feature subset in a
    greedy fashion. At each stage, this estimator chooses the best feature to
    add or remove based on the cross-validation score of an estimator.

    Read more in the :ref:`User Guide <sequential_feature_selection>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

    direction : {'forward', 'backward'}, default='forward'
        Whether to perform forward selection or backward selection.

    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        If None, the estimator's score method is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. When evaluating a new feature to
        add or remove, the cross-validation procedure is parallel over the
        folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    n_features_to_select_ : int
        The number of features that were selected.

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    See Also
    --------
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination based on importance weights, with
        automatic selection of the number of features.
    SelectFromModel : Feature selection based on thresholds of importance
        weights.

    Examples
    --------
    >>> from sklearn.feature_selection import SequentialFeatureSelector
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
    >>> sfs.fit(X, y)
    SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                              n_features_to_select=3)
    >>> sfs.get_support()
    array([ True, False,  True,  True])
    >>> sfs.transform(X).shape
    (150, 3)
    """

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select=None,
        direction="forward",
        scoring=None,
        cv=5,
        n_jobs=None,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Learn the features to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        tags = self._get_tags()  # type: ignore
        X, y = self._validate_data(  # type: ignore
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )
        n_features = X.shape[1]

        error_msg = (
            "n_features_to_select must be either None, an "
            "integer in [1, n_features - 1] "
            "representing the absolute "
            "number of features, or a float in (0, 1] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )
        if self.n_features_to_select is None:
            self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, numbers.Integral):
            if not 0 < self.n_features_to_select < n_features:  # type: ignore
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, numbers.Real):
            if not 0 < self.n_features_to_select <= 1:  # type: ignore
                raise ValueError(error_msg)
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)
        else:
            raise ValueError(error_msg)

        if self.direction not in ("forward", "backward"):
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction}."
            )

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        self.scores = np.full(shape=n_features, fill_value=np.nan)
        n_iterations = (
            self.n_features_to_select_
            if self.direction == "forward"
            else n_features - self.n_features_to_select_
        )

        ddesc = "Step-up" if self.direction == "forward" else "Step-down"
        for _ in tqdm(
            range(n_iterations),
            total=n_iterations,  # type: ignore
            desc=f"{ddesc} feature selection: ",
            leave=True,
        ):  # type: ignore
            new_feature_idx, score = self._get_best_new_feature(
                cloned_estimator, X, y, current_mask
            )
            current_mask[new_feature_idx] = True
            self.scores[new_feature_idx] = score

        if self.direction == "backward":
            current_mask = ~current_mask
        self.support_ = current_mask

        return self

    def _get_best_new_feature(self, estimator, X, y, current_mask) -> tuple[int, float]:
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_feature_indices = np.flatnonzero(~current_mask)
        # scores = {}

        scores: list[tuple[float, int]] = Parallel()(
            delayed(get_score)(
                estimator=estimator,
                X=X,
                y=y,
                cv=self.cv,
                scoring=self.scoring,
                current_mask=current_mask,
                feature_idx=feature_idx,
                direction=self.direction,  # type: ignore
            )
            for feature_idx in candidate_feature_indices
        )
        scores_dict = {idx: score for score, idx in scores}

        # for feature_idx in tqdm(candidate_feature_indices, total=len(candidate_feature_indices)):
        #     score, idx = get_score(
        #         estimator=estimator,
        #         X=X,
        #         y=y,
        #         cv=self.cv,
        #         scoring=self.scoring,
        #         current_mask=current_mask,
        #         feature_idx=feature_idx,
        #         direction=self.direction,  # type: ignore
        #     )
        #     scores[idx] = score
        feature_idx = max(scores_dict, key=lambda feature_idx: scores_dict[feature_idx])
        score = scores_dict[feature_idx]
        return feature_idx, score

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def _more_tags(self):
        return {
            "allow_nan": _safe_tags(self.estimator, key="allow_nan"),
            "requires_y": True,
        }


def get_score(
    estimator: Any,
    X: Any,
    y: Any,
    cv: Any,
    scoring: Any,
    current_mask: ndarray,
    feature_idx: int,
    direction: Literal["backward", "forward"],
) -> tuple[float, int]:
    candidate_mask = current_mask.copy()
    candidate_mask[feature_idx] = True
    if direction == "backward":
        candidate_mask = ~candidate_mask
    X_new = X[:, candidate_mask]
    return cross_val_score(
        estimator, X_new, y, cv=cv, scoring=scoring, n_jobs=1
    ).mean(), feature_idx
