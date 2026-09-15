from typing import Literal, Union

from numpy import ndarray
from pandas import DataFrame, Series

# fmt: off
# Data types
FlatArray = Union[DataFrame, Series, ndarray]

# feature selection and cleaning types
CorrMethod = Literal["pearson", "spearman"]
UnivariateMetric = Literal["d", "auc", CorrMethod]
FeatureSelection = Literal["minimal", "step-down", "step-up", "pca", "kpca", UnivariateMetric]  # noqa
FeatureCleaning = Literal["constant", "correlated", "lowinfo"]

# Model-related types
Classifier = Literal["rf", "svm", "dtree", "mlp", "bag"]
Regressor = Literal["linear", "rf", "svm", "adaboost", "gboost", "mlp", "knn"]
Estimator = Union[Classifier, Regressor]
EstimationMode = Literal["classify", "regress"]
Kernel = Literal["rbf", "linear", "sigmoid"]

# validation- and testing-related types
ValMethod = Literal["holdout", "kfold", "k-fold", "loocv", "mc", "none"]
CVMethod = Union[int, float, Literal["loocv", "mc"]]
MultiTestCVMethod = Union[int, Literal["mc"]]
