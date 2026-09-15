from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on
import sys
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    cast,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from matplotlib.axes import Axes
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import linregress
from scipy.stats._stats_mstats_common import LinregressResult
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import (
    train_test_split,
)

if TYPE_CHECKING:
    pass


def calibration_scores(
    y_train: Series,
    y_test: Series,
    preds_train: Series,
    preds_test: Series,
    probs_train: ndarray,
    probs_test: ndarray,
) -> DataFrame:
    ce_train = log_loss(y_true=y_train, y_pred=probs_train)
    ce_test = log_loss(y_true=y_test, y_pred=probs_test)

    n_cls = len(np.unique(y_train))
    if n_cls == 2:  # probs are always shape (n_samples, n_cls)
        probs_train = probs_train.copy()[:, 0].reshape(-1, 1)
        probs_test = probs_test.copy()[:, 0].reshape(-1, 1)

    unqs = np.unique(y_test)
    if n_cls == 2:
        unqs = unqs[:-1]

    brier_trains = [
        brier_score_loss(y_true=y_train == unqs[i], y_prob=probs_train[:, i])
        for i in range(probs_train.shape[1])
    ]
    brier_tests = [
        brier_score_loss(y_true=y_test == unqs[i], y_prob=probs_test[:, i])
        for i in range(probs_test.shape[1])
    ]
    ce_trains = [
        log_loss(y_true=y_train == unqs[i], y_pred=probs_train[:, i])
        for i in range(probs_train.shape[1])
    ]
    ce_tests = [
        log_loss(y_true=y_test == unqs[i], y_pred=probs_test[:, i])
        for i in range(probs_test.shape[1])
    ]

    correct_train = preds_train == y_train
    correct_test = preds_test == y_test
    acc_trains = [
        np.sum((y_train == unqs[i]) & (correct_train)) / np.sum(y_train == unqs[i])
        for i in range(len(unqs))
    ]
    acc_tests = [
        np.sum((y_test == unqs[i]) & (correct_test)) / np.sum(y_test == unqs[i])
        for i in range(len(unqs))
    ]
    acc_train_total = accuracy_score(y_train, preds_train)
    acc_test_total = accuracy_score(y_test, preds_test)
    print(f"Training total acc: {acc_train_total}")
    print(f"Testing total acc:  {acc_test_total}")
    print(f"Training accs: {acc_trains}")
    print(f"Testing accs:  {acc_tests}")

    print("Brier training scores (lower is better):", brier_trains)
    print("Brier testing scores (lower is better): ", brier_tests)

    print(f"Training CE: {ce_train}")
    print(f"Testing CE:  {ce_test}")

    totals = [
        acc_train_total,
        acc_test_total,
        np.mean(np.array(brier_trains)),
        np.mean(np.array(brier_tests)),
        ce_train,
        ce_test,
    ]
    rows = []
    for acc_train, acc_test, brier_train, brier_test, ce_train, ce_test in zip(
        acc_trains, acc_tests, brier_trains, brier_tests, ce_trains, ce_tests
    ):
        rows.append([acc_train, acc_test, brier_train, brier_test, ce_train, ce_test])

    return DataFrame(
        columns=["acc_tr", "acc", "brier_tr", "brier", "ce_tr", "ce"],
        index=Series(["all", *unqs], name="class"),
        data=[totals, *rows],
    )


def sklearn_calibration_plot(
    info: DataFrame,
    y_train: Series,
    y_test: Series,
    probs_train: ndarray,
    probs_test: ndarray,
) -> None:
    df = info
    n_cls = len(np.unique(y_train))
    if n_cls == 2:
        n_cls = 1

    fig, axes = plt.subplots(nrows=2, ncols=n_cls, squeeze=False)
    ax: Axes
    for i, ax in enumerate(axes[0]):
        yt = y_train == i
        yp = probs_train[:, i]
        CalibrationDisplay.from_predictions(y_true=yt, y_prob=yp, ax=ax, n_bins=10)
        ax.set_title(f"Class {i}: Brier={df.loc[i, 'brier_tr']:0.5f} (lower=better)")
        if i == 0:
            lab = ax.get_ylabel().replace(" ()", "")
            ax.set_ylabel(f"Training {lab}")
        else:
            ax.set_ylabel("")

    for i, ax in enumerate(axes[1]):
        yt = y_test == i
        yp = probs_test[:, i]
        CalibrationDisplay.from_predictions(y_true=yt, y_prob=yp, ax=ax, n_bins=10)
        ax.set_title(f"Class {i}: Brier={df.loc[i, 'brier']:0.5f} (lower=better)")
        if i == 0:
            lab = ax.get_ylabel().replace(" (Positive Class: 1)", "")
            ax.set_ylabel(f"Testing {lab}")
        else:
            ax.set_ylabel("")
    for ax in axes.flat:
        ax.set_xlabel(ax.get_xlabel().replace(" (Positive Class: 1)", ""))
    fig.set_size_inches(w=20, h=12)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()


def confidence_accuracies(
    confidences: ndarray, preds: Series, y_true: Series
) -> tuple[list[float], ndarray]:
    accs = []
    ps = np.linspace(0, 1, 50)
    for p in ps:
        idx = confidences >= p
        preds_train_sub = preds[idx]
        y_true_sub = y_true[idx]
        accs.append(accuracy_score(y_true_sub, preds_train_sub))
    return accs, ps


def bootstrap_means(
    confidences: ndarray, preds: Series, y_true: Series
) -> tuple[list[float], list[float]]:
    confs, accs = [], []
    count = 0
    while count < 5000:
        min_samples = 10
        start = np.random.uniform(0, 1)
        length = np.random.uniform(0, 1 - start)
        stop = start + length
        # idx = np.random.choice(train_idx, replace=True, size=100)
        idx = (confidences < stop) & (confidences > start)
        preds_sub = preds[idx]
        y_true_sub = y_true[idx]
        if len(y_true_sub) >= min_samples:
            accs.append(accuracy_score(y_true_sub, preds_sub))
            confs.append(np.mean(confidences[idx]))
            count += 1
    return confs, accs


def calibration_plot(
    y_train: Series,
    y_test: Series,
    preds_train: Series,
    preds_test: Series,
    probs_train: ndarray,
    probs_test: ndarray,
) -> None:
    """
    Notes
    -----
    Given a set of predicted probabilities, and
    """
    df = calibration_scores(
        y_train=y_train,
        y_test=y_test,
        preds_train=preds_train,
        preds_test=preds_test,
        probs_train=probs_train,
        probs_test=probs_test,
    )
    print(df.to_markdown(tablefmt="simple", floatfmt="0.4f"))
    # sklearn_calibration_plot(
    #     info=df,
    #     y_train=y_train,
    #     y_test=y_test,
    #     probs_train=probs_train,
    #     probs_test=probs_test,
    # )
    # confidence is the difference between two largest probs
    confs_train = np.diff(np.sort(probs_train, axis=1)[:, -2:]).ravel()
    confs_test = np.diff(np.sort(probs_test, axis=1)[:, -2:]).ravel()

    # confidence is the largest probability
    confs_train_max = np.max(probs_train, axis=1).ravel()
    confs_test_max = np.max(probs_test, axis=1).ravel()

    # confidence is the entropy
    confs_train_ent = (probs_train * np.log(probs_train)).sum(axis=1).ravel()
    confs_test_ent = (probs_test * np.log(probs_test)).sum(axis=1).ravel()

    accs_train, ps = confidence_accuracies(confs_train, preds_train, y_train)
    accs_test = confidence_accuracies(confs_test, preds_test, y_test)[0]

    accs_train_max = confidence_accuracies(confs_train_max, preds_train, y_train)[0]
    accs_test_max = confidence_accuracies(confs_test_max, preds_test, y_test)[0]

    res_trains = bootstrap_means(confs_train, preds_train, y_train)
    res_tests = bootstrap_means(confs_test, preds_test, y_test)
    res_trains_max = bootstrap_means(confs_train_max, preds_train, y_train)
    res_tests_max = bootstrap_means(confs_test_max, preds_test, y_test)

    conf_boot_trains, acc_boot_trains = res_trains
    conf_boot_tests, acc_boot_tests = res_tests
    conf_boot_trains_max, acc_boot_trains_max = res_trains_max
    conf_boot_tests_max, acc_boot_tests_max = res_tests_max

    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=False, sharey=False)
    axes[0][0].plot(ps, accs_train, color="black")
    axes[0][1].plot(ps, accs_test, color="black", label="diff")
    axes[0][0].plot(ps, accs_train_max, color="orange")
    axes[0][1].plot(ps, accs_test_max, color="orange", label="max")
    axes[0][0].set_xlabel("Confidence Threshold")
    axes[0][1].set_xlabel("Confidence Threshold")
    axes[0][0].set_ylabel("Accuracy on Samples with Confidence >= Threshold")
    axes[0][0].set_title("Training")
    axes[0][1].set_title("Testing")

    args = dict(s=1.0, alpha=0.5)
    axes[1][0].scatter(conf_boot_trains, acc_boot_trains, color="black", **args)
    axes[1][1].scatter(conf_boot_tests, acc_boot_tests, color="black", **args)
    axes[1][0].scatter(conf_boot_trains_max, acc_boot_trains_max, color="orange", **args)
    axes[1][1].scatter(conf_boot_tests_max, acc_boot_tests_max, color="orange", **args)

    res = cast(LinregressResult, linregress(conf_boot_trains, acc_boot_trains))
    m, b, r_train, p_train = res.slope, res.intercept, res.rvalue, res.pvalue  # type: ignore
    x = np.linspace(np.min(conf_boot_trains), np.max(conf_boot_trains), 1000)
    y = m * x + b
    axes[1][0].plot(x, y, color="red", alpha=0.7, label="diff")

    res = cast(LinregressResult, linregress(conf_boot_trains_max, acc_boot_trains_max))
    m, b, r_train_max, p_train_max = res.slope, res.intercept, res.rvalue, res.pvalue  # type: ignore
    x = np.linspace(np.min(conf_boot_trains_max), np.max(conf_boot_trains_max), 1000)
    y = m * x + b
    axes[1][0].plot(x, y, color="blue", alpha=0.7, label="max")

    res = cast(LinregressResult, linregress(conf_boot_tests, acc_boot_tests))
    m, b, r_test, p_test = res.slope, res.intercept, res.rvalue, res.pvalue  # type: ignore
    x = np.linspace(np.min(conf_boot_tests), np.max(conf_boot_tests), 1000)
    y = m * x + b
    axes[1][1].plot(x, y, color="red", alpha=0.7, label="diff")

    res = cast(LinregressResult, linregress(conf_boot_tests_max, acc_boot_tests_max))
    m, b, r_test_max, p_test_max = res.slope, res.intercept, res.rvalue, res.pvalue  # type: ignore
    x = np.linspace(np.min(conf_boot_tests_max), np.max(conf_boot_tests_max), 1000)
    y = m * x + b
    axes[1][1].plot(x, y, color="blue", alpha=0.7, label="max")

    axes[1][0].set_xlabel("Subsample Mean Confidence")
    axes[1][1].set_xlabel("Subsample Mean Confidence")
    axes[1][0].set_ylabel("Accuracy")
    axes[1][0].set_title(
        rf"Training ($r_{{diff}}={r_train:0.3f}, p_{{diff}}={p_train:0.3f}; "
        rf"r_{{max}}={r_train_max:0.3f}, p_{{max}}={p_train_max:0.3f}$)"
    )
    axes[1][1].set_title(
        rf"Testing ($r_{{diff}}={r_test:0.3f}, p_{{diff}}={p_test:0.3f}; "
        rf"r_{{max}}={r_test_max:0.3f}, p_{{max}}={p_test_max:0.3f}$)"
    )

    axes[1][1].legend().set_visible(True)
    axes[0][1].legend().set_visible(True)
    fig.suptitle(
        f"Accuracy When Limiting to Samples where Confidence is Above Threshold (Total Acc: {accuracy_score(y_test, preds_test):0.4f})"
    )
    fig.set_size_inches(w=20, h=12)
    fig.tight_layout()
    plt.show()


def main() -> None:
    n_cls = np.random.randint(2, 6)
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=n_cls, n_informative=n_cls * 2
    )
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, stratify=y)
    models = [("LGBM", LGBMClassifier), ("LR", LogisticRegressionCV)]
    for mname, model_cls in models:
        args: dict = {}
        if mname == "LGBM":
            args["verbosity"] = -1
            args["force_col_wise"] = True
        else:
            args["max_iter"] = 2000
        model = model_cls(**args)
        model.fit(X_tr, y_tr)
        preds_tr = model.predict(X_tr)
        preds_test = model.predict(X_ts)

        probs_tr = model.predict_proba(X_tr)
        probs_test = model.predict_proba(X_ts)

        print("=" * 81)
        print(mname)
        calibration_plot(
            y_train=y_tr,
            y_test=y_ts,
            preds_train=preds_tr,
            preds_test=preds_test,
            probs_train=probs_tr,
            probs_test=probs_test,
        )


if __name__ == "__main__":
    matplotlib.use("QtAgg")
    plt.rcParams["text.usetex"] = True
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
