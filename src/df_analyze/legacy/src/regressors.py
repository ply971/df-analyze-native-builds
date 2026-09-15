from typing import Callable, Dict

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.svm import SVR

from df_analyze._types import Regressor

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 16, 64, 128]
N_SPLITS = 5
TEST_SCORES = ["accuracy", "roc_auc"]


def get_regressor_constructor(name: Regressor) -> Callable:
    REGRESSORS: Dict[str, Callable] = {
        "linear": ElasticNet,
        "rf": RF,
        "svm": SVR,
        "adaboost": AdaBoostRegressor,
        "gboost": GradientBoostingRegressor,
        "mlp": MLP,
        "knn": KNN,
    }
    constructor = REGRESSORS[name]
    return constructor
