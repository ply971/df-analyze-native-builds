# NOTE: feature selection is PRIOR to hypertuning, but "what features are best" is of course
# contengent on the choice of regressor / classifier. The correct way to frame this is as an overall
# derivative-free optimization problem where the classifier choice is *just another hyperparameter*.
# However, this is not possible for stepwise selection methods, which already take 5-20 hours even
# without incorporating any tuning.

import os
from typing import Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing_extensions import Literal

from df_analyze._setup import MEMOIZER
from df_analyze._types import (
    CorrMethod,
    Estimator,
)
from df_analyze.legacy.src.classifiers import get_classifier_constructor
from df_analyze.legacy.src.regressors import get_regressor_constructor
from df_analyze.sklearn_pasta._sequential import SequentialFeatureSelector

# FEATURE_CACHE = GLOBALS.JOBLIB_CACHE_DIR / "__features__"
# MEMOIZER = Memory(location=FEATURE_CACHE, backend="local", compress=9)


def cohens_d(df: DataFrame, target: str) -> Series:
    """For each feature in `df`, compute the absolute Cohen's d values.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column
        `target`

    Returns
    -------
    ds: Series
        Cohen's d values for each feature, as a Pandas Series.
    """
    X = df.drop(columns=target)
    y = df[target].copy()
    x1, x2 = X.loc[y == 0, :], X.loc[y == 1, :]
    n1, n2 = len(x1) - 1, len(x2) - 1
    sd1, sd2 = np.std(x1, ddof=1, axis=0), np.std(x2, ddof=1, axis=0)
    sd_pools = np.sqrt((n1 * sd1 + n2 * sd2) / (n1 + n2))
    m1, m2 = np.mean(x1, axis=0), np.mean(x2, axis=0)
    ds = np.abs(m1 - m2) / sd_pools
    return ds


def auroc(df: DataFrame, target: str) -> Series:
    """For each feature in `df` compute rho, the common-language effect size (see Notes)
    via the area-under-the-ROC curve (AUC), and rescale this effect size to allow sorting
    across features.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column
        `target`

    Returns
    -------
    rescaled_aucs: Series
        The rescaled AUC values (see notes) for each feature.

    Notes
    -----
    This is equivalent to calculating U / abs(y1*y2), where `y1` and `y2` are the subgroup
    sizes, and U is the Mann-Whitney U, and is also sometimes referred to as Herrnstein's
    rho [1] or "f", the "common-language effect size" [2].

    Note we also *must* rescale as rho is implicity "signed", with values above 0.5
    indicating separation in one direction, and values below 0.5 indicating separation in
    the other.

    [1] Herrnstein, R. J., Loveland, D. H., & Cable, C. (1976). Natural concepts in
        pigeons. Journal of Experimental Psychology: Animal Behavior Processes, 2, 285-302

    [2] McGraw, K.O.; Wong, J.J. (1992). "A common language effect size statistic".
        Psychological Bulletin. 111 (2): 361–365. doi:10.1037/0033-2909.111.2.361.
    """
    X = df.drop(columns=target)
    y = df[target].copy()

    aucs = Series(data=[roc_auc_score(y, X[col]) for col in X], index=X.columns)
    rescaled = (aucs - 0.5).abs()
    return rescaled


def correlations(df: DataFrame, target: str, method: CorrMethod = "pearson") -> Series:
    """For each feature in `df` compute the ABSOLUTE correlation with the target variable.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column `target`

    method: "pearson" | "spearman" | "kendall"
        How to correlate.

    Returns
    -------
    corrs: Series
        Correlations for each feature.
    """
    X = df.drop(columns=target)
    y = df[target].copy()
    return X.corrwith(y, method=method).abs()


@MEMOIZER.cache
def select_stepwise_features(
    df: DataFrame,
    target: str,
    is_classification: bool,
    estimator: Estimator,
    n_features: int,
    direction: Literal["forward", "backward"] = "forward",
) -> Union[DataFrame, ndarray]:
    """Perform stepwise feature selection (which takes HOURS) and save the selected features in a
    DataFrame so that subsequent runs do not need to do this again."""

    # outfile = DATADIR / f"mcic_{direction}-select{n_features}__{classifier}.json"
    get_model = (
        get_classifier_constructor if is_classification else get_regressor_constructor
    )
    selector = SequentialFeatureSelector(
        estimator=get_model(estimator)(),  # type: ignore
        n_features_to_select=n_features,
        direction=direction,
        scoring="accuracy" if is_classification else "neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
    )
    X_raw = df.drop(columns=target)
    y = df[target].to_numpy()
    if is_classification:
        y = y.astype(int)
    X_arr = StandardScaler().fit_transform(X_raw)
    X = DataFrame(data=X_arr, columns=X_raw.columns, index=X_raw.index)
    if estimator == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = (
            "ignore"  # can't kill ConvergenceWarning any other way
        )
        selector.fit(X, y)
        os.environ["PYTHONWARNINGS"] = before
    else:
        selector.fit(X, y)

    column_idx = selector.get_support()
    # reduced = X.loc[:, column_idx].copy()
    # reduced[target] = y
    # reduced.to_json(outfile)
    # return reduced
    return column_idx
