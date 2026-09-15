from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
import warnings
from pathlib import Path
from typing import Optional
from warnings import catch_warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import (
    chisquare,
    kurtosis,
    kurtosistest,
    skew,
    skewtest,
)
from scipy.stats import differential_entropy as dentropy


def describe_continuous(df: DataFrame, column: str) -> DataFrame:
    cols = [
        "min",
        "mean",
        "max",
        "sd",
        "p05",
        "median",
        "p95",
        "iqr",
        "skew",
        "skew_p",
        "kurt",
        "kurt_p",
        "entropy",
        "nan_freq",
    ]
    x = df[column].to_numpy()
    percs = np.nanpercentile(x, q=[0.05, 0.25, 0.5, 0.75, 0.95])
    idx = np.isnan(x)
    n_nan = idx.sum()
    x = x[~idx].ravel()
    try:
        p_skew = skewtest(x, nan_policy="omit")[1] if len(x) > 8 else np.nan
        p_kurt = kurtosistest(x, nan_policy="omit")[1] if len(x) > 20 else np.nan
    except Exception:
        traceback.print_exc()
        p_skew = p_kurt = np.nan
    try:
        with catch_warnings(category=RuntimeWarning):
            warnings.filterwarnings(
                action="ignore", category=RuntimeWarning, message=".*divide by zero.*"
            )
            entrop = dentropy(x)
    except ValueError:
        traceback.print_exc()
        entrop = np.nan
    data = {
        "min": np.nanmin(x),
        "mean": np.nanmean(x),
        "max": np.nanmax(x),
        "sd": np.nanstd(x, ddof=1),
        "p05": percs[0],
        "median": percs[2],
        "p95": percs[4],
        "iqr": percs[3] - percs[1],
        "skew": skew(x),
        "skew_p": p_skew,
        "kurt": kurtosis(x, nan_policy="omit"),
        "kurt_p": p_kurt,
        "entropy": entrop,
        "nan_freq": n_nan / len(df[column]),
    }
    return DataFrame(
        data=data,
        index=[column],
        columns=cols,
    )


def describe_categorical(df: DataFrame, column: str) -> DataFrame:
    cols = [
        "n_levels",  # number of categories
        "nans",  # NaN count
        "nan_freq",  # NaN frequency
        "med_freq",  # Median of level frequencies
        "min_freq",  # Frequency of least-frequent level
        "max_freq",  # Frequency of most-frequent level
        "min_name",  # Name of least common level
        "max_name",  # Name of most common level
        "heterog",  # chi-square
        "heterog_p",  # chi-square p-value
        "n_entropy",  # normalized entropy, entropy / log(n_levels)
    ]
    x = df[column]
    nulls = x.isnull()
    unqs, cnts = np.unique(x.map(str), return_counts=True)
    freqs = cnts / np.sum(cnts)
    ent = -np.dot(freqs, np.log(freqs))
    nent = ent / np.log(len(unqs))
    result = chisquare(f_obs=freqs)
    max_name = unqs[np.argmax(cnts)]
    min_name = unqs[np.argmin(cnts)]
    chi = result.statistic
    p = result.pvalue
    data = {
        "n_levels": len(unqs),
        "nans": nulls.sum(),
        "nan_freq": nulls.mean(),
        "med_freq": np.median(freqs),
        "min_freq": np.min(freqs),
        "max_freq": np.max(freqs),
        "min_name": min_name,
        "max_name": max_name,
        "heterog": chi,
        "heterog_p": p,
        "n_entropy": nent,
    }
    return DataFrame(
        data=data,
        index=[column],
        columns=cols,
    )


def describe_all_features(
    continuous: Optional[DataFrame],
    categoricals: Optional[DataFrame],
    target: Series,
    is_classification: bool,
) -> tuple[Optional[DataFrame], Optional[DataFrame], DataFrame]:
    """
    Parameters
    ----------
    continuous: DataFrame
        DataFrame of continuous and ordinal features

    categoricals: DataFrame
        DataFrame of categorical (un-dummified) features

    target: Series
        Variable to predict

    mode: EstimationMode
        Whether a classification or regression task (nature of target).

    Returns
    -------
    desc_cont: Optional[DataFrame]
        Continuous descriptions

    desc_cat: Optional[DataFrame]
        Categorical descriptions

    Notes
    -----
    univariate descriptive stats for each feature (robust and non-robust
    measures of scale and location—e.g. mean, median, SD, IQR, and some higher
    moments—e.g. skew, kurtosis), entropy

    For each non-categorical feature (including ordinals):
        - min, mean, max, sd
        - p05, median, p95, IQR
        - skew, kurtosis
        - differential (continuous) entropy (scipy.stats.differential_entropy)

    For each categorical feature:
        - n_cats / n_levels (number of categories)
        - med_freq, min_freq, max_freq, (i.e. median/min/max class proportions)
        - min_name, max_name: names of least/most frequent classes
        - heterogeneity / class balance (scipy.stats.chisquare)
        - n_entropy: entropy normalized by max possible entropy
    """

    with warnings.catch_warnings():
        # https://github.com/pandas-dev/pandas/issues/55928
        #
        # Pandas spams spurious FutureWarnings here:
        #
        # > The behavior of DataFrame concatenation with empty or all-NA
        # > entries is deprecated. In a future version, this will no longer
        # > exclude empty or all-NA columns when determining the result dtypes.
        # > To retain the old behavior, exclude the relevant entries before the
        # > concat operation. TODO: pandas 2.1.0 has a FutureWarning for
        # > concatenating DataFrames with Null entries
        #
        # Even though it is not possible for all NaN rows to be produced here.
        # I.e. this is actually a type inference issue where it is making false
        # assumptions about NaNs (i.e. they clearly have not considered you
        # might be concatening rows with only some entries NaN), as in:
        # https://github.com/pandas-dev/pandas/issues/55928#issuecomment-2333980979
        #
        # Since future versions will behave exactly the same in our case, we
        # silence the incompetent message.
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*concatenation with empty or all-NA entries.*",
        )

        if continuous is not None and not continuous.empty:
            descs_cont: list[DataFrame] = []
            for col in continuous.columns:
                descs_cont.append(describe_continuous(continuous, col))
            df_cont = pd.concat(descs_cont, axis=0)
        else:
            df_cont = None

        if categoricals is not None and not categoricals.empty:
            descs_cat: list[DataFrame] = []
            for col in categoricals.columns:
                descs_cat.append(describe_categorical(categoricals, col))
            df_cat = pd.concat(descs_cat, axis=0)
        else:
            df_cat = None

        targ = target.to_frame()
        tcol = targ.columns[0]

        if is_classification:
            df_targ = describe_categorical(targ, tcol)
        else:
            df_targ = describe_continuous(targ, tcol)

    return df_cont, df_cat, df_targ
