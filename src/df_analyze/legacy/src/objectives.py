import os
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from warnings import filterwarnings

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy import ndarray
from optuna import Trial
from pandas import DataFrame
from sklearn.ensemble import AdaBoostRegressor as AdaReg
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import (
    BaseCrossValidator,
    LeaveOneOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.model_selection import cross_validate as cv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier as DTreeClassifier

from df_analyze._constants import SEED
from df_analyze._types import CVMethod
from df_analyze.scoring import (
    accuracy_scorer,
    auc_scorer,
    sensitivity_scorer,
    specificity_scorer,
)

Splits = Iterable[Tuple[ndarray, ndarray]]

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_DEPTH_MIN = 1
MLP_DEPTH_MAX = 8
MLP_WIDTH_MIN = 4
MLP_WIDTH_MAX = 128

N_SPLITS = 5
TEST_SCORES = dict(
    accuracy=accuracy_scorer,
    roc_auc=auc_scorer,
    sensitivity=sensitivity_scorer,
    specificity=specificity_scorer,
)
NEG_MAE = "neg_mean_absolute_error"


def get_cv(
    y_train: DataFrame, cv_method: CVMethod, n_folds: Optional[int] = None
) -> Union[int, Splits, BaseCrossValidator]:
    """Helper to construct an object that `sklearn.model_selection.cross_validate` will accept in
    its `cv` argument

    Parameters
    ----------
    y_train: DataFrame
        Needed for stratification.

    cv_method: CVMethod
        Method to create object for.

    Returns
    -------
    cv: Union[int, Splits, BaseCrossValidator]
        The object that can be passed into the `cross_validate` function

    Notes
    -----
    If cv_method is an `int`, then it specifies k-fold with the `int` for `k`.

    If cv_method is a `float`, then it specifies holdout where the float is
    the percentage of samples heldout for testing.

    """
    if n_folds is None:
        n_folds = 5
    if isinstance(cv_method, int):
        return int(cv_method)
    if isinstance(cv_method, float):  # stratified holdout
        if cv_method <= 0 or cv_method >= 1:
            raise ValueError("Holdout test_size must be in (0, 1)")
        test_size = cv_method
        y = np.array(y_train).ravel()
        idx = np.arange(y.shape[0])
        return [
            train_test_split(
                idx, test_size=test_size, random_state=SEED, shuffle=True, stratify=y
            )  # type: ignore
        ]
    cv_method = str(cv_method).lower()  # type: ignore
    if cv_method == "loocv":
        return LeaveOneOut()
    if cv_method in ["kfold", "k-fold"]:
        return StratifiedKFold(n_splits=n_folds, random_state=SEED, shuffle=True)
    if cv_method == "mc":
        raise NotImplementedError()
        return StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=SEED)
    raise ValueError(f"Invalid `cv_method`: {cv_method}")


def mlp_args(trial: Trial) -> Dict[str, Any]:
    depth = trial.suggest_int(
        "depth", low=MLP_DEPTH_MIN, high=MLP_DEPTH_MAX + 1, step=1
    )
    width = trial.suggest_int("breadth", low=MLP_WIDTH_MIN, high=MLP_WIDTH_MAX, step=4)
    layers = mlp_layers_from_sizes(depth, width)
    args: Dict = dict(
        hidden_layer_sizes=layers,
        activation=trial.suggest_categorical("activation", ["relu"]),
        solver=trial.suggest_categorical("solver", ["adam"]),
        alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        batch_size=trial.suggest_categorical("batch_size", choices=[16, 32, 64, 128]),
        learning_rate=trial.suggest_categorical("learning_rate", choices=["constant"]),
        learning_rate_init=trial.suggest_float(
            "learning_rate_init", 5e-5, 1e-1, log=True
        ),
        max_iter=trial.suggest_categorical("max_iter", [200]),
        early_stopping=trial.suggest_categorical("early_stopping", [False]),
        validation_fraction=trial.suggest_categorical("validation_fraction", [0.1]),
    )
    return args


def mlp_layers_from_sizes(depth: int, breadth: int) -> tuple[int, ...]:
    """Convert the params returned from trial.best_params into a form that can be used by
    MLPClassifier"""
    return tuple([int(breadth) for _ in range(depth)])


def mlp_args_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    # these args below must match names in `mlp_args` above
    params = deepcopy(params)
    depth = params.pop("depth")
    width = params.pop("breadth")
    layer_args = dict(hidden_layer_sizes=mlp_layers_from_sizes(depth, width))
    return {**layer_args, **params}


"""See Optuna docs (https://optuna.org/#code_ScikitLearn) for the motivation behond the closures
below. Currently I am using closures, but this might be a BAD IDEA in parallel contexts. In any
case, they do seem to suggest this is OK https://optuna.readthedocs.io/en/stable/faq.html
#how-to-define-objective-functions-that-have-own-arguments, albeit by using classes or lambdas. """


def svm_classifier_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            C=trial.suggest_float("C", 1e-10, 1e10, log=True),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = SVC(cache_size=500, **args)
        scores = cv(
            estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1
        )
        return float(np.mean(scores["test_score"]))

    return objective


# https://neptune.ai/blog/lightgbm-parameters-guide
# https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
def lgbm_classifier_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    # TODO

    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            num_leaves=trial.suggest_int("num_leaves", 2, 256, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = LGBMClassifier(**args)
        scores = cv(
            estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1
        )
        return float(np.mean(scores["test_score"]))

    return objective


def rf_classifier_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 500),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = RF(n_jobs=2, **args)
        scores = cv(
            estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=4
        )
        return float(np.mean(scores["test_score"]))

    return objective


def dtree_classifier_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = DTreeClassifier(**args)
        scores = cv(
            estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1
        )
        return float(np.mean(scores["test_score"]))

    return objective


def bagging_classifier_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 23, 2),
            max_features=trial.suggest_uniform("max_features", 0, 1),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = BaggingClassifier(
            base_estimator=LR(solver=LR_SOLVER), random_state=SEED, n_jobs=2, **args
        )
        scores = cv(
            estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=4
        )
        return float(np.mean(scores["test_score"]))

    return objective


def mlp_classifier_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        mlp = MLP(**mlp_args(trial))
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = (
            "ignore"  # can't kill ConvergenceWarning any other way
        )
        filterwarnings("ignore", category=ConvergenceWarning)
        _cv = get_cv(y_train, cv_method)
        scores = cv(mlp, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1)
        os.environ["PYTHONWARNINGS"] = before
        acc = float(np.mean(scores["test_score"]))
        return acc

    return objective


# REGRESSORS


def linear_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            alpha=trial.suggest_uniform("alpha", 0.01, 10),
            l1_ratio=trial.suggest_uniform("l1_ratio", 0, 1),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = ElasticNet(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


def svm_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            C=trial.suggest_float("C", 1e-10, 1e10, log=True),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = SVR(cache_size=500, **args)
        scores = cv(estimator, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


# https://neptune.ai/blog/lightgbm-parameters-guide
# https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
def lgbm_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    # TODO

    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            num_leaves=trial.suggest_int("num_leaves", 2, 256, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = LGBMRegressor(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=1)
        return float(np.mean(scores["test_score"]))

    return objective


def rf_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 500),
            # NOTE: when updating SKLEAN these need to become
            # "squared_error" and "absolute_error" instead of "mse" and "mae"
            criterion=trial.suggest_categorical("criterion", ["mse", "mae", "poisson"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = RFR(n_jobs=2, **args)
        scores = cv(estimator, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=4)
        return float(np.mean(scores["test_score"]))

    return objective


def adaboost_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 550, step=50),
            learning_rate=trial.suggest_uniform("learning_rate", 0.05, 3),
            loss=trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = AdaReg(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


def gradboost_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 550, step=50),
            learning_rate=trial.suggest_uniform("learning_rate", 0.005, 3),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            alpha=trial.suggest_uniform("alpha", 0, 1),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = GBR(loss="huber", **args)
        scores = cv(estimator, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


def knn_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 20, 1),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 20),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = KNeighborsRegressor(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


def mlp_regressor_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        mlp = MLPR(**mlp_args(trial))
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = (
            "ignore"  # can't kill ConvergenceWarning any other way
        )
        filterwarnings("ignore", category=ConvergenceWarning)
        _cv = get_cv(y_train, cv_method)
        scores = cv(mlp, X=X_train, y=y_train, scoring=NEG_MAE, cv=_cv, n_jobs=-1)
        os.environ["PYTHONWARNINGS"] = before
        acc = float(np.mean(scores["test_score"]))
        return acc

    return objective
