from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from df_analyze._constants import N_CAT_LEVEL_MIN

FLOAT_INFO = (
    "Found features that are mostly or entirely continuous values. This means "
    "either that these columns convert to a floating point representation "
    "without any loss of information, or that most (e.g. over 80%) of the values "
    "do. df-analyze will automatically treat these columns as continuous to "
    "prevent wasted compute resources that would be incurred with encoding them "
    "as categorical, however, this might be an error.\n\n"
    "Columns inferred to be continuous:\n\n{info}"
)

ORD_INFO = (
    "Found features that could contain ordinal values (i.e. non-categorical "
    "integer values). Check if these features are ordinal or categorical, and "
    "then explicitly pass them to either the `--categoricals` or `--ordinals` "
    "options when configuring df-analyze.\n\n"
    "Columns inferred to be ordinal:\n\n{info}"
)

MAYBE_ORD_INFO = (
    "Found features that are possibly ordinal. These will generally be "
    "features with integer values in some typical range [min, max]. However, "
    "when all values in a small range (e.g. [0, 5]) are well-sampled, these "
    "features are indistinguishable from a label-encoded categorical variable. "
    "Make sure the features below do not in fact include features that should "
    "be included in the `--categoricals` option. "
    "\n\n"
    "Columns inferred to be likely ordinal:\n\n{info}"
)

CERTAIN_ORD_INFO = (
    "Found features that almost certainly should be interpreted as ordinal. "
    "This should mean the feature contains only integer or NaN values, but "
    "does not contain 0 or 1 as values (categorical variables will almost "
    "always be encoded starting from 0 or 1). df-analyze will NOT allow theses "
    "features to be treated as categorical even if passed to the "
    "`--categoricals` option: to treat them as categorical correct the feature "
    "values so that they are proper categorical lables (strings or integers "
    "starting at 0 or 1)."
    "\n\n"
    "Features that will be treated as certainly ordinal:\n\n{info}"
)

COERCED_ORD_INFO = (
    "Coerced some features to ordinal. In order to avoid excessive compute "
    "costs, when a feature is neither clearly categorical nor ordinal, "
    "df-analyze will assume such a feature is ordinal unless the feature is "
    "well-sampled as a categorical, AND if the feature has a typical "
    "categorical name."
    "\n\n"
    "Features coerced to ordinal:\n\n{info}"
)

BINARY_INFO = (
    "Found features with only two unique values. These will be treated as "  #
    "binary categorical."
)

BINARY_VIA_NAN_INFO = (
    "Found features with only one unique value and NaNs. These will be treated "
    "as binary categorical.\n\n"
    "Features that are binary only due to NaNs:\n\n{info}"
)

BINARY_PLUS_NAN_INFO = (
    "Found features with only two unique values and NaNs. These will be treated "
    "as binary categoricals.\n\n"
    "Features that are binary and also have NaNs:\n\n{info}"
)

ID_INFO = (
    "Found features likely containing identifiers (i.e. unique string or integer "
    "values that are assigned arbitrarily), or which have more levels (unique "
    "values) than one half of the number of (non-NaN) samples in the data. This "
    "is most likely an identifier or junk feature which has no predictive value, "
    "and thus should be removed from the data. Even if the feature is not an "
    "identifer, with such a large number of levels, then a test set (either in "
    "k-fold, or holdout) will, on average, mostly contain values that were never "
    "seen during training. Thus, these features are essentially undersampled, "
    "and too sparse to be useful given the amount of data. Encoding this many "
    "values also massively increases compute costs for little gain. We thus "
    "REMOVE these features. However, but you should inspect these features "
    "yourself and ensure these features are not better described as either "
    "ordinal or continuous. If so, specify them using the `--ordinals` "
    "`--continous` options to df-analyze.\n\n"
    "Features inferred to be identifiers:\n\n{info}"
)

MAYBE_ID_INFO = """{info}
    Found features that do not appear to be either continuous or categorical,
    but have a large number of unique non-NaN values. These may be identifiers.
    If so, consider adding them to `--drops` to silence this message.
    "\n\n"
    "Features inferred to be likely identifiers:\n\n{info}"
"""

CERTAIN_ID_INFO = (
    "Found features that do not appear continuous, but have as many unique "
    "values as non-NaN samples. These are treated as certain to be identifiers "
    "and are REMOVED from all subsequeny analyses by df-analyze. This removal "
    "cannot be prevented by specifying these features as `--ordinals` or "
    "`--categoricals`, and can be mitigated only by editing the data (or "
    "removing these features or adding them to the `--drops` option to silence "
    "this message). "
    "\n\n"
    "Features inferred to be certainly identifiers:\n\n{info}"
)

TIME_INFO = (
    "Found features that appear to be datetime data or time differences. "
    "Datetime data cannot currently be handled by `df-analyze` (or most AutoML "
    "or or most automated predictive approaches) due to special data "
    "preprocessing needs (e.g. Fourier features), splitting (e.g. time-based "
    "cross-validation, forecasting, hindcasting) and in the models used (ARIMA, "
    "VAR, etc.). Thus these data will be REMOVED from analysis, regardless of "
    "whether or not they were specified in `--ordinals` or `--categoricals` "
    "options.\n\n"
    "Columns inferred to be timestamps:\n{info}\n\n"
    ""
    "To remove this warning, DELETE these columns from your data, or manually "
    "edit or convert the column values so they are interpretable as a "
    "categorical or continuous variable reflecting some kind of cyclicality that "
    "may be relevant to the predictive task. E.g. a variable that stores the "
    "time a sample was recorded might be converted to categorical variable "
    "like:\n\n "
    ""
    "  - morning, afternoon, evening, night (for daily data)\n"
    "  - day of the week (for monthly data)\n"
    "  - month (for yearly data)\n"
    "  - season (for multi-year data)\n\n"
    ""
    "Or to continuous / ordinal cyclical versions of these, like:\n\n"
    ""
    "  - values from 0 to 23 for ordinal representation of day hour\n"
    "  - values from 0.0 to 23.99 for continuous version of above\n"
    "  - values from 0 to 7 for day of the week, 0 to 365 for year\n\n"
    ""
    "It is possible to convert a single time feature into all of the above, i.e. "
    "to expand the feature into multiple cyclical features. This would be a "
    "variant of Fourier features (see e.g.\n\n"
    ""
    "\tTian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong "
    'Jin:\n\t"FEDformer: Frequency Enhanced Decomposed Transformer\n\tfor '
    'Long-term Series Forecasting", 2022;\n\t'
    "[http://arxiv.org/abs/2201.12740].\n\n"
    ""
    "`df-analyze` may in the future attempt to automate this via `sktime` "
    "(https://www.sktime.net)."
)

MAYBE_TIME_INFO = """
    Found features with a large number of values that parse as dates or
    times, and which have undersampled levels when interpreted as
    categoricals. These data will be REMOVED from analysis, regardless of
    whether or not they were specified in `--ordinals` or `--categoricals`.
    To silence this message, either pass these features to the `--drops`
    option, or edit them as mentioned above.
    "\n\n"
    "Columns inferred likely to be datetime data:\n\n{info}"
"""

CERTAIN_TIME_INFO = MAYBE_TIME_INFO

CAT_INFO = (
    "Found features that look categorical. These will be one-hot encoded to "
    "allow use in subsequent analyses. To silence this warning, specify them as "
    "categoricals or ordinals manually either via the CLI or in the spreadsheet "
    "file header, using the `--categoricals` and/or `--ordinals` option."
    "\n\n"
    "Features that may be categorical:\n\n{info}"
)

# Probably shouldn't ever be printed?
MAYBE_CAT_INFO = """
    "Found features that were interpreted as categorical due to having both "
    "well-sampled classes AND a name matching some common (English-language) "
    "categorical variable names. "
    "\n\n"
    "Features that may be categorical:\n\n{info}"
"""

CERTAIN_CAT_INFO = """
    "Found features that can only be interpreted as categorical. These will be "
    "one-hot encoded and possibly deflated. "
    "\n\n"
    "Features that are certainly categorical:\n\n{info}"
"""

COERCED_CAT_INFO = """
    "Found features that were interpreted as categorical due to having both "
    "well-sampled classes AND a name matching some common (English-language) "
    "categorical variable names. "
    "\n\n"
    "Features coerced to categorical:\n\n{info}"
"""

CONST_INFO = (
    "Found features that are constant (i.e. all values) are NaN or all values "
    "are the same non-NaN value). These contain no information and will be "
    "removed automatically. To silence this message, add these feature names "
    "to the `--drops` option. "
    "\n\n "
    "Features that are constant:\n{info}"
)

BIG_INFO = (
    "Found features that, when interpreted as categorical, have more than 50 "
    "unique levels / classes. Unless you have a large number of samples, or if "
    "these features have a highly imbalanced / skewed distribution, then they "
    "will cause sparseness after one-hot encoding. This is generally not "
    "beneficial to most algorithms. You should inspect these features and think "
    "if it makes sense if they would be predictively useful for the given "
    "target. If they are unlikely to be useful, consider removing them from the "
    "data. This will also likely considerably improve `df-analyze` predictive "
    "performance and reduce compute "
    "times. However, we do NOT remove these features automatically.\n\n"
    "Features with over 50 levels when interpreted as categorical: {info}"
)
NYAN_INFO = (
    "Found features that are constant if dropping NaN values. Whether or not "
    "these features have predictive value will depend on if data is missing "
    "completely at random (MCAR - no information), are correlated with other "
    "data features that have predictive value, or if missingness is related to "
    "the prediction target. If you have a lot of features, these would be good "
    "candidates to consider for removal."
    "\n\n"
    "Features that are constant when dropping NaNs: {info}"
)

CONT_INFO = (
    "Found features that convert mostly or entirely to a floating point "
    "representation without loss of information, and which do not appear to be "
    "ordinal. "
    "\n\n"
    "Features that may be continuous:\n\n{info}"
)

MAYBE_CONT_INFO = (
    "Found features that convert mostly or entirely to a floating point "
    "representation without loss of information, and which do not appear to be "
    "ordinal. "
    "\n\n"
    "Features inferred likely to be continuous:\n\n{info}"
)

CERTAIN_CONT_INFO = (
    "Found features that convert to floating point representation without loss "
    "of information, and which do not appear to be ordinal. "
    "\n\n"
    "Features inferred to be continuous:\n\n{info}"
)

COERCED_CONT_INFO = (
    "Found ambiguous features that contain decimals ('.') and otherwise "
    "mostly convert to floating point. These will be treated as continuous. "
    "\n\n"
    "Features coerced to be continuous:\n\n{info}"
)

INFLATION_HEADER = (
    "Found and 'deflated' a number of categorical variables with levels that "
    "are unlikely to be reliable or useful in prediction. "
)

INFLATION_INFO = (
    f"{INFLATION_HEADER}"
    "\n\n"
    "For each level of categorical variable to be predictively useful, there "
    "must be enough samples to be statistically meaningful (or to allow some "
    "reasonable generalization) in each fold used for fitting or analyses. "
    "When a categorical variable is full of undersampled levels, these levels "
    "are essentially noise, and we say the categorical variable is 'inflated'. "
    "\n\n"
    "In practice, we would likely rarely feel confident about generalizations "
    "based on only 10-20 samples of something. Assuming k-fold is used for "
    "validation, and a default training split of 0.6, then this means about "
    f"{N_CAT_LEVEL_MIN} samples per categorical level would a reasonable "
    "default minimum sample requirement for a level to be predictively useful. "
)
