from __future__ import annotations

import re
from enum import Enum
from math import ceil
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from dateutil.parser import parse
from dateutil.parser._parser import UnknownTimezoneWarning
from pandas import Series
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn.experimental import enable_iterative_imputer  # noqa

from df_analyze._constants import N_CAT_LEVEL_MIN, NAN_STRINGS
from df_analyze.preprocessing.inspection.text import (
    BINARY_INFO,
    BINARY_PLUS_NAN_INFO,
    BINARY_VIA_NAN_INFO,
    CERTAIN_CAT_INFO,
    CERTAIN_CONT_INFO,
    CERTAIN_ID_INFO,
    CERTAIN_ORD_INFO,
    CERTAIN_TIME_INFO,
    COERCED_CAT_INFO,
    COERCED_CONT_INFO,
    COERCED_ORD_INFO,
    CONST_INFO,
    MAYBE_CAT_INFO,
    MAYBE_CONT_INFO,
    MAYBE_ID_INFO,
    MAYBE_ORD_INFO,
    MAYBE_TIME_INFO,
)

TIME_WORDS = [
    r".*time.*",
    r".*duration.*",
    r".*interval.*",
    r".*lag.*",
    r".*date.*",
    r".*elapse.*",
    r".*day[ _\.\-]?",
    r".*month[ _\.\-]?",
    r".*year[ _\.\-]?",
    r".*yr[ _\.\-]?",
    r".*hour[ _\.\-]?",
]
ORD_WORDS = [
    r".*[_ \-]n[_ \-].*",
    r".*[_ \-]num[_ \-]?.*",
    r".*age.*",
    r".*days.*",
    r".*grade.*",
    r".*hours.*",
    r".*hrs.*",
    r".*months.*",
    r".*rank.*",
    r".*seconds.*",
    r".*secs.*",
    r".*size.*",
    r".*years.*",
    r".*yrs.*",
]
ID_WORDS = [r".id$"]
CAT_WORDS = [
    r".*type.*",
    r".*kind.*",
    r".*is[_ \-\.].*",
    r".*has[_ \-\.].*",
    r".*not[_ \-\.].*",
    "city",
    "race",
    "ancestry",
    "town",
    "country",
    "occupation",
    r".*relig.*",
    "state",
    "zip",
    "code",
    "name",
    "gender",
    "sex",
    "status",
]
CONT_WORDS = [
    r".*average.*",
    r".*avg.*",
    r".*pct.*",
    r".*percent.*",
    r".*rate.*",
    r".*ratio.*",
    r".*total.*",
]


class InferredKind(Enum):
    MaybeOrd = "ord?"
    MaybeCat = "cat?"
    MaybeCont = "cont?"
    MaybeId = "id?"
    MaybeTime = "time?"
    CertainOrd = "ord"
    CertainCat = "cat"
    CertainCont = "cont"
    CertainId = "id"
    CertainTime = "time"
    CertainNyan = "nyan"
    Binary = "bin"  # encode to {0, 1}
    BinaryViaNan = "const+nan"  # encode to {0, 1}
    BinaryPlusNan = "bin+nan"  # encode to two {0, 1} features
    Const = "const"
    BigCat = "big-cat"
    CoercedCat = "cat-coerce"
    CoercedOrd = "ord-coerce"
    CoercedCont = "cont-coerce"
    UserCategorical = "user-cat"
    UserOrdinal = "user-ord"
    NoInference = "none"

    def fmt(self) -> str:
        fmts = {
            InferredKind.MaybeOrd: MAYBE_ORD_INFO,
            InferredKind.MaybeCat: MAYBE_CAT_INFO,
            InferredKind.MaybeCont: MAYBE_CONT_INFO,
            InferredKind.MaybeId: MAYBE_ID_INFO,
            InferredKind.MaybeTime: MAYBE_TIME_INFO,
            InferredKind.CertainOrd: CERTAIN_ORD_INFO,
            InferredKind.CertainCat: CERTAIN_CAT_INFO,
            InferredKind.CertainCont: CERTAIN_CONT_INFO,
            InferredKind.CertainId: CERTAIN_ID_INFO,
            InferredKind.CertainTime: CERTAIN_TIME_INFO,
            InferredKind.CertainNyan: "nyan",
            InferredKind.Binary: BINARY_INFO,
            InferredKind.BinaryViaNan: BINARY_VIA_NAN_INFO,
            InferredKind.BinaryPlusNan: BINARY_PLUS_NAN_INFO,
            InferredKind.Const: CONST_INFO,
            InferredKind.BigCat: "big-cat",
            InferredKind.CoercedCat: COERCED_CAT_INFO,
            InferredKind.CoercedOrd: COERCED_ORD_INFO,
            InferredKind.CoercedCont: COERCED_CONT_INFO,
            InferredKind.UserCategorical: "User-specified categorical",
            InferredKind.UserOrdinal: "User-specified ordinal",
            InferredKind.NoInference: "none",
        }
        return fmts[self]

    def is_certain(self) -> bool:
        return self in [
            InferredKind.Binary,
            InferredKind.BinaryViaNan,
            InferredKind.BinaryPlusNan,
            InferredKind.CertainCat,
            InferredKind.CertainCont,
            InferredKind.CertainOrd,
            InferredKind.CertainId,
            InferredKind.CertainTime,
            InferredKind.Const,
        ]

    def is_coerced(self) -> bool:
        return self in [
            InferredKind.CoercedCat,
            InferredKind.CoercedCont,
            InferredKind.CoercedOrd,
        ]

    def is_bin(self) -> bool:
        return self in [
            InferredKind.Binary,
            InferredKind.BinaryViaNan,
            InferredKind.BinaryPlusNan,
        ]

    def is_cat(self) -> bool:
        return self in [
            InferredKind.MaybeCat,
            InferredKind.CertainCat,
            InferredKind.CoercedCat,
            InferredKind.UserCategorical,
            InferredKind.BigCat,
        ]

    def is_cont(self) -> bool:
        return self in [
            InferredKind.MaybeCont,
            InferredKind.CertainCont,
            InferredKind.CoercedCont,
        ]

    def is_ord(self) -> bool:
        return self in [
            InferredKind.MaybeOrd,
            InferredKind.CertainOrd,
            InferredKind.CoercedOrd,
            InferredKind.UserOrdinal,
        ]

    def is_id(self) -> bool:
        return self in [
            InferredKind.MaybeId,
            InferredKind.CertainId,
        ]

    def is_time(self) -> bool:
        return self in [
            InferredKind.MaybeTime,
            InferredKind.CertainTime,
        ]

    def is_const(self) -> bool:
        return self in [
            InferredKind.Const,
        ]

    def should_drop(self) -> bool:
        return self.is_time() or self.is_const() or self.is_id()

    def overrides_user(self) -> bool:
        return self.is_certain() or self in [
            InferredKind.MaybeId,
            InferredKind.MaybeTime,
        ]

    def __bool__(self) -> bool:
        if self is InferredKind.NoInference:
            return False
        return True


class Inference:
    def __init__(
        self, kind: InferredKind = InferredKind.NoInference, reason: str = ""
    ) -> None:
        self.kind = kind
        self.reason = reason

    def should_drop(self) -> bool:
        return self.is_time() or self.is_const() or self.is_id()

    def is_certain(self) -> bool:
        return self.kind.is_certain()

    def is_coerced(self) -> bool:
        return self.kind.is_coerced()

    def overrides_user(self) -> bool:
        return self.is_certain() or self.kind.overrides_user()

    def is_bin(self) -> bool:
        return self.kind.is_bin()

    def is_cat(self) -> bool:
        return self.kind.is_cat()

    def is_cont(self) -> bool:
        return self.kind.is_cont()

    def is_ord(self) -> bool:
        return self.kind.is_ord()

    def is_id(self) -> bool:
        return self.kind.is_id()

    def is_time(self) -> bool:
        return self.kind.is_time()

    def is_const(self) -> bool:
        return self.kind.is_const()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.kind.name}: {self.reason})"

    __repr__ = __str__

    def __bool__(self) -> bool:
        return bool(self.kind)

    def __eq__(self, other: object | Inference) -> bool:
        if not isinstance(other, Inference):
            return False
        return self.kind == other.kind and self.reason == other.reason


def has_cat_name(series: Series) -> tuple[bool, str]:
    for pattern in CAT_WORDS:
        if re.search(pattern, str(series.name).lower()) is not None:
            return True, str(pattern)
    return False, ""


def is_timelike(s: str) -> bool:
    # https://stackoverflow.com/a/25341965 for this...
    try:
        int(s)
        return False
    except Exception:
        ...
    try:
        float(s)
        return False
    except Exception:
        ...
    try:
        with catch_warnings():
            filterwarnings("ignore", category=UnknownTimezoneWarning)
            parse(s, fuzzy=False)
        return True
    except ValueError:
        return False


def infer_timelike(series: Series) -> Inference:
    # we don't want to interpret integer-like data as times, even though
    # they could be e.g. Unix timestamps or something like that
    if converts_to_int(series):
        return Inference()

    # This seems to be another false positive from dateutil.parse
    if converts_to_float(series):
        return Inference()

    series = series.astype(str)

    # We are mostly worried about timestamps we do NOT want to convert to
    # categoricals. Thus before checking that most data parses as datetime,
    # we check that we are below out inflation threshold and do not flag
    # these, even if they are timelike.
    level_counts = np.unique(series, return_counts=True)[1]
    if np.all(level_counts > N_CAT_LEVEL_MIN):
        return Inference()

    N = len(series)
    n_subsamp = max(ceil(0.5 * N), 500)
    n_subsamp = min(n_subsamp, N)
    idx = Series(np.random.permutation(N)[:n_subsamp])

    percent = series.iloc[idx].apply(is_timelike).sum() / n_subsamp
    if percent >= 1.0:
        return Inference(InferredKind.CertainTime, "100% of data parses as datetime")
    if percent > (1.0 / 3.0):
        p = series.loc[idx].apply(is_timelike).mean()
        if p > 0.5:
            return Inference(
                InferredKind.CertainTime,
                f"{p * 100:< 2.2f}% of data appears parseable as datetime data",
            )
        if p > 0.3:
            return Inference(
                InferredKind.MaybeTime,
                f"{p * 100:< 2.2f}% of data appears parseable as datetime data",
            )
    return Inference()


def infer_identifier(series: Series) -> Inference:
    """
    Returns
    -------
    id_like: bool
        If the variable has a good chance of being an identifier

    desc: str
        A string describing why the variable looks like an identifier
    """
    if infer_floatlike(series).is_certain():
        return Inference()

    if isinstance(series.dtype, CategoricalDtype):
        if len(series.dtype.categories) > (len(series) / 2):
            return Inference(
                InferredKind.CertainId,
                "More unique values than one half of number of non-NaN samples",
            )
        return Inference()

    cnts = np.unique(series.apply(str), return_counts=True)[1]
    if np.all(cnts == 1):  # obvious case
        return Inference(
            InferredKind.CertainId, "All values including possible NaNs are unique"
        )

    dropped = series.dropna().apply(str)
    if len(dropped) < 0.5 * len(series):
        # seems unlikely only half of data would have identifier info?
        return Inference()

    unqs, cnts = np.unique(dropped, return_counts=True)
    if np.all(cnts == 1):  # also obvious case
        return Inference(InferredKind.CertainId, "All non-NaN values are unique")

    if len(unqs) >= (len(dropped) / 2):
        if converts_to_int(dropped):
            return Inference()
        return Inference(
            InferredKind.MaybeId,
            "More unique values than one half of number of non-NaN samples",
        )

    return Inference()


def prob_unq_ordinals(n_samples: int, ordinal_max: int) -> float:
    """
    Return the probability of all `n_samples` samples having a unique value
    when drawn uniformly from the values [0, ordinal_max].
    """
    if n_samples > ordinal_max:
        return 0.0
    p = 1.0
    for i in range(1, n_samples - 1):
        # case i you have M - i choices out of M total
        prob = (ordinal_max - i) / ordinal_max
        p *= prob
    return p


def maybe_large_ordinal(series: Series) -> bool:
    prob = prob_unq_ordinals(n_samples=len(series), ordinal_max=int(np.max(series)))
    return prob >= 0.5


def converts_to_int(series: Series) -> bool:
    try:
        x = series.astype(int)
        if np.all(x.astype(str) == series):
            return True
    except Exception:
        ...
    with catch_warnings():
        filterwarnings(
            "ignore",
            message=".*invalid value encountered in cast.*",
            category=RuntimeWarning,
        )
        converted = series.convert_dtypes(
            infer_objects=True,
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=False,  # type: ignore
        )
    return converted.dtype.kind in ["i", "b"]


def converts_to_float(series: Series) -> bool:
    try:
        converted = series.astype(float).astype(str)
        if np.all(converted == series.to_numpy()):
            return True

    except Exception:
        ...
    converted = series.convert_dtypes(
        infer_objects=True,
        convert_string=True,
        convert_integer=False,
        convert_boolean=False,
        convert_floating=True,  # type: ignore
    )
    return converted.dtype.kind == "f"


def infer_binary(series: Series) -> Inference:
    """To be run AFTER infer_constant, infer_timelike, infer_id"""
    unqs = series.astype(str).apply(lambda x: np.nan if x in NAN_STRINGS else x).unique()
    try:
        str_nan = "nan" in map(str.lower, unqs.astype(str))
    except Exception:
        str_nan = False
    try:
        numpy_nan = np.isnan(unqs).any()
    except TypeError:
        numpy_nan = False
    try:
        pandas_nan = pd.isna(unqs).any()
    except TypeError:
        pandas_nan = False

    if len(unqs) == 2:
        if numpy_nan or pandas_nan or str_nan:
            return Inference(InferredKind.BinaryViaNan, "Only one unique non-NaN value")
        return Inference(InferredKind.Binary, "Two unique non-Nan values")
    elif len(unqs) == 3 and (numpy_nan or pandas_nan or str_nan):
        return Inference(InferredKind.BinaryPlusNan, "Two unique non-Nan values plus NaN")

    return Inference()


def infer_ordinal(series: Series) -> Inference:
    """
    Returns
    -------
    ordinal: bool
        If the variable has a good chance of being ordinal

    desc: str
        A string describing the apparent ordinality when `ordinal` is True
    """
    forced = pd.to_numeric(series, errors="coerce")
    idx = forced.isna()

    if np.all(idx):  # columns definitely all not numerical
        return Inference()

    dropped = series.dropna()
    if not converts_to_int(dropped):
        return Inference()

    try:
        ints = dropped.astype(int)
    except Exception:
        # this `dropped` contains things that look like floats
        return Inference()

    unq_ints, cnts = np.unique(ints, return_counts=True)
    vmin, vmax = ints.min(), ints.max()
    if len(unq_ints) == 1:
        return Inference(InferredKind.CertainOrd, "Constant integer after dropping NaNs")

    # no categorical variable should ever be missing the first category
    if int(vmin) not in [0, 1]:
        return Inference(InferredKind.CertainOrd, "Integers not starting at 0 or 1")

    # typical case might be ordinal or ids with some NaNs
    # Two obvious cases here:
    #
    # (1) every non-NaN value is unique because it is an identifier
    # (2) we have a very large ordinal relative to the number of samples
    if np.all(cnts == 1):
        if maybe_large_ordinal(dropped):
            return Inference(
                InferredKind.MaybeOrd,
                f"All unique values in large range [{vmin}, {vmax}] relative to number of samples",
            )
        return Inference()

    # Now we have something int-like with few unique values. If diffs
    # on sorted unique values are all 1, we have something extremely
    # likely to be ordinal again.
    diffs = np.diff(np.sort(unq_ints))
    unq_diffs = np.unique(diffs)
    if len(unq_diffs) == 1:
        if vmin == 0 and vmax == 1:
            return Inference(
                InferredKind.CertainOrd, f"Binary {{{vmin}, {vmax}}} indicator"
            )
        return Inference(
            InferredKind.MaybeOrd, f"Increasing integers in [{vmin}, {vmax}]"
        )

    # Small chance remains that we are missing some level(s) of an ordinal, so
    # that we have all values in [0, ..., N] except for a couple, making some
    # diffs on the sorted unique values greater than 1.
    if np.mean(diffs == 1) >= 0.8:
        return Inference(
            InferredKind.MaybeOrd,
            r"80% or more of unique integer values differ only by 1",
        )
    if len(unq_ints) / (vmax - vmin) >= 0.8:
        return Inference(
            InferredKind.MaybeOrd,
            f"80% or more of values in [{vmin}, {vmax}] are sampled",
        )

    # Here, we just heuristically probably want to warn the user for some
    # likely common cases In most cases, this would be some rating
    imax = np.max(unq_ints)
    if imax in [4, 6]:
        return Inference(
            InferredKind.MaybeOrd,
            f"Largest int is a common 0-indexed Likert-type scale value: {imax}",
        )
    if imax in [5, 7]:
        return Inference(
            InferredKind.MaybeOrd,
            f"Largest int is a common Likert-type scale value: {imax}",
        )

    # The 9, 10 cases below are extremely unlikely
    if imax in [10, 100]:
        return Inference(
            InferredKind.MaybeOrd, f"Largest int is a common scale max: {imax}"
        )

    if imax in [9, 99]:
        return Inference(
            InferredKind.MaybeOrd,
            f"Largest int is a common 0-indexed scale max: {imax}",
        )

    return Inference()


def infer_floatlike(series: Series) -> Inference:
    """
    Returns
    -------
    floaty: bool
        If the variable has a good chance of being continuous

    desc: str
        A string describing the apparent continuousness

    """
    # for some reasons Python None converts inconsistently to NaN...
    if converts_to_int(series):
        if series.astype(str).str.contains(".", regex=False).any():
            return Inference(
                InferredKind.MaybeCont,
                "Values convert to integers but contain decimals",
            )
        return Inference()
    ord = infer_ordinal(series)
    if ord is InferredKind.CertainOrd:
        return Inference()

    try:
        series.astype(float)
        return Inference(
            InferredKind.CertainCont, "All values are not integers and convert to float"
        )
    except Exception:
        pass

    forced = pd.to_numeric(series, errors="coerce")
    idx = forced.isna()
    if np.all(idx):  # columns definitely all not float
        return Inference()

    if np.mean(idx) > 0.2:
        odd_vals = np.unique(series.astype(str)[idx]).tolist()
        if len(odd_vals) > 5:
            desc = f"{str(odd_vals[:5])[:-1]} ...]"
        else:
            desc = str(odd_vals)
        return Inference(
            InferredKind.MaybeCont,
            rf"More than 20% of values parse as floats, but couldn't parse values: {desc}",
        )
    return Inference()


def infer_categorical(series: Series) -> Inference:
    if infer_floatlike(series).is_certain():
        return Inference()

    if infer_timelike(series).is_certain():
        return Inference()
    if infer_identifier(series).is_certain():
        return Inference()

    if not converts_to_int(series):  # already checked not float
        return Inference(InferredKind.CertainCat, "Data not interpretable as numeric")

    # Now we have something that converts to int, and doesn't look like an
    # identifier. It could be a small ordinal or a categorical, but generally
    # there is no way to be sure. All we can say is if it does NOT look ordinal
    # then we definitely want to label it as categorical. Otherwise, we should
    # still flag it. Note however this relies on `infer_ordinal` being certain.
    ord = infer_ordinal(series)
    if ord.kind is InferredKind.CertainOrd:
        return Inference()
    elif ord.kind is InferredKind.MaybeOrd:
        return Inference(InferredKind.MaybeCat, "Looks both categorical and ordinal")
    else:
        return Inference(InferredKind.MaybeCat, "Integer data but not ordinal-like")


def infer_constant(series: Series) -> Inference:
    if len(np.unique(series.astype(str))) == 1:
        return Inference(InferredKind.Const, "Single value even if including NaNs")
    return Inference()
