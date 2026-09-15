from typing import Any, Callable, Dict

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTreeClassifier

from df_analyze._types import Classifier

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5
TEST_SCORES = ["accuracy", "roc_auc"]


def bagger(**kwargs: Any) -> Callable:
    """Helper for uniform interface only"""
    return BaggingClassifier(base_estimator=LR(solver=LR_SOLVER), **kwargs)  # type: ignore


def get_classifier_constructor(name: Classifier) -> Callable:
    CLASSIFIERS: Dict[str, Callable] = {
        "rf": RF,
        "svm": SVC,
        "dtree": DTreeClassifier,
        "mlp": MLP,
        "bag": bagger,
    }
    constructor = CLASSIFIERS[name]
    return constructor
