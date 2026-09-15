"""Lightweight, typed controls for advanced CLI options in both front ends.

No ML imports at GUI startup. test_gui_option_parity checks this catalog against
the real argument parser so new CLI switches cannot silently miss the GUI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Option:
    flag: str
    label: str
    group: str
    kind: str
    default: Any
    help: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


STRATEGIES = (
    "strategy1_min_aer", "strategy2_topn", "strategy3_acc_minus_aer",
    "strategy4_exp_weighted", "strategy5_dynamic_threshold",
    "strategy6_overconfidence_penalty", "strategy7_calibration_aware",
    "strategy8_trimmed_weighted", "strategy9_borda_rank", "strategy10_switching_hybrid",
)

OPTIONS = (
    Option("--n-filter-cont", "Continuous feature budget", "Feature selection", "count", "0.5", "An integer count or fraction between 0 and 1."),
    Option("--n-filter-cat", "Categorical feature budget", "Feature selection", "count", "0.5", "An integer count or fraction between 0 and 1."),
    Option("--embed-select", "Embedded selectors", "Feature selection", "multi", ["linear"], "Run either or both embedded selectors. Enable embedded feature selection first.", ("linear", "lgbm")),
    Option("--test-val-size", "Exact holdout size", "Evaluation and runtime", "count", "0.4", "Overrides the holdout slider: enter a row count or a fraction."),
    Option("--seed", "Explicit seed", "Evaluation and runtime", "seed", "default", "Use default, random, or an integer (including 0). Overrides the basic seed input."),
    Option("--verbosity", "Log detail", "Evaluation and runtime", "choice", "1", "0: errors only; 1: normal progress; 2: CLI debug level (currently equivalent to normal).", ("0", "1", "2")),
    Option("--no-warn-explosion", "Suppress encoding expansion warnings", "Evaluation and runtime", "bool", False, "Silence warnings about the number of one-hot encoded features."),
    Option("--aer-oof-folds", "Out-of-fold splits", "Adaptive error", "int", 5, "Cross-validation folds for estimating error on out-of-fold predictions.", minimum=2),
    Option("--aer-bins", "Confidence bins", "Adaptive error", "int", 20, "Number of bins used to estimate error by confidence.", minimum=2),
    Option("--aer-target-error", "Target error rate", "Adaptive error", "float", .05, "Desired maximum error among accepted predictions.", minimum=0, maximum=1),
    Option("--aer-alpha", "Significance level", "Adaptive error", "float", .05, "Significance level for error bounds; must be strictly between 0 and 1.", minimum=0, maximum=1),
    Option("--aer-min-bin-count", "Minimum samples per bin", "Adaptive error", "int", 10, "Minimum sample support for confidence bins.", minimum=1),
    Option("--aer-prior-strength", "Prior strength", "Adaptive error", "float", 2., "Smoothing prior strength.", minimum=0),
    Option("--no-aer-smooth", "Disable smoothing", "Adaptive error", "bool", False, "Use unsmoothed error estimates."),
    Option("--aer-monotonic", "Monotonic error curve", "Adaptive error", "bool", False, "Enforce monotonic expected error versus confidence."),
    Option("--aer-adaptive-binning", "Quantile confidence bins", "Adaptive error", "bool", False, "Use quantile bins instead of fixed-width bins."),
    Option("--aer-confidence-metric", "Confidence measure", "Adaptive error", "choice", "auto", "Model-specific measures fall back to automatic selection when unavailable.", ("auto", "proba_margin", "tree_vote_agreement", "tree_leaf_support", "knn_vote", "knn_dist_weighted", "knn_min_dist", "p_max", "p_pred", "p_margin", "p_pred_margin")),
    Option("--aer-nmin", "Minimum accepted predictions", "Adaptive error", "int", 1, "Minimum accepted sample count when choosing a threshold.", minimum=1),
    Option("--aer-top-k", "Limit analyzed models", "Adaptive error", "int", 0, "0 analyzes all models; positive values analyze only the top K.", minimum=0),
    Option("--aer-ensemble", "Enable ensemble analysis", "Adaptive ensembles", "bool", False, "Combine tuned classifiers using adaptive-error ensemble strategies."),
    Option("--aer-ensemble-strategies", "Ensemble strategies", "Adaptive ensembles", "multi", list(STRATEGIES), "Choose one or more strategies. Default runs all strategies.", STRATEGIES),
    Option("--aer-ens-top-n", "Models per ensemble", "Adaptive ensembles", "int", 3, "Number of top models available to ensemble strategies.", minimum=1),
    Option("--aer-ens-beta", "Confidence weight", "Adaptive ensembles", "float", 5., "Beta for exponential confidence weighting.", minimum=0),
    Option("--aer-ens-tau0", "Base confidence threshold", "Adaptive ensembles", "float", .3, "Initial confidence threshold.", minimum=0, maximum=1),
    Option("--aer-ens-lambda", "Regularization mixture", "Adaptive ensembles", "float", .5, "Mixing factor for regularization.", minimum=0, maximum=1),
    Option("--aer-ens-alpha", "Support exponent", "Adaptive ensembles", "float", .7, "Exponent controlling the influence of sample support.", minimum=0),
    Option("--aer-ens-trim-q", "Trim quantile", "Adaptive ensembles", "float", .6, "Quantile used to suppress low-confidence members.", minimum=0, maximum=1),
    Option("--aer-ens-tau-low", "Lower switching threshold", "Adaptive ensembles", "float", .15, "Lower cutoff for the switching ensemble.", minimum=0, maximum=1),
    Option("--aer-ens-tau-high", "Upper switching threshold", "Adaptive ensembles", "float", .35, "Upper cutoff for the switching ensemble.", minimum=0, maximum=1),
)
OPTION_MAP = {option.flag: option for option in OPTIONS}


def validate_options(values: object) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError("Advanced settings must be an object.")
    for flag, value in values.items():
        spec = OPTION_MAP.get(flag)
        if spec is None:
            raise ValueError(f"Unknown advanced setting: {flag}")
        problem = False
        if spec.kind == "bool":
            problem = type(value) is not bool
        elif spec.kind in ("int", "float"):
            problem = type(value) not in ((int,) if spec.kind == "int" else (int, float))
            if not problem:
                problem = not math.isfinite(value) or (spec.minimum is not None and value < spec.minimum) or (spec.maximum is not None and value > spec.maximum)
        elif spec.kind == "multi":
            problem = not isinstance(value, list) or not value or any(not isinstance(v, str) or v not in spec.choices for v in value)
            if not problem:
                problem = len(value) != len(set(value))
        elif spec.kind == "choice":
            problem = not isinstance(value, str) or value not in spec.choices
        elif spec.kind == "seed":
            problem = not isinstance(value, str) or not (value in ("default", "random") or (value.isdecimal() and 0 <= int(value) <= 2**32 - 1))
        elif spec.kind == "count":
            try:
                problem = not isinstance(value, str) or not ((value.isdecimal() and int(value) >= 0) or 0 <= float(value) <= 1)
                if not problem and flag == "--test-val-size":
                    problem = float(value) <= 0 or float(value) == 1
            except (ValueError, TypeError):
                problem = True
        if flag == "--aer-alpha" and not problem:
            problem = not 0 < value < 1
        if problem:
            raise ValueError(f"Invalid {spec.label.lower()} ({flag}). {spec.help}")
    low = values.get("--aer-ens-tau-low", .15)
    high = values.get("--aer-ens-tau-high", .35)
    if low > high:
        raise ValueError("Lower switching threshold must not exceed the upper threshold.")
    return dict(values)


def option_argv(values: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for flag, value in validate_options(values).items():
        kind = OPTION_MAP[flag].kind
        if kind == "bool":
            if value:
                result.append(flag)
        else:
            result.extend([flag, *value] if kind == "multi" else [flag, str(value)])
    return result
