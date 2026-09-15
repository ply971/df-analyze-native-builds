from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
)

import numpy as np

from df_analyze.analysis.adaptive_error.ensemble_strategies import (
    _apply_strategy_acc_minus_aer,
    _apply_strategy_borda_rank,
    _apply_strategy_calibration_aware,
    _apply_strategy_dynamic_threshold,
    _apply_strategy_exp_weighted,
    _apply_strategy_min_aer,
    _apply_strategy_overconfidence_penalty,
    _apply_strategy_switching_hybrid,
    _apply_strategy_topn,
    _apply_strategy_trimmed_weighted,
)


@dataclass(frozen=True)
class EnsembleStrategyContext:
    proba_oof_stack: np.ndarray
    aer_oof_stack: np.ndarray
    oof_acc: np.ndarray
    proba_test_stack: np.ndarray
    aer_test_stack: np.ndarray
    slugs_ens: list[str]


@dataclass(frozen=True)
class EnsembleStrategyResult:
    p_ens_oof: np.ndarray
    r_star_oof: np.ndarray
    p_ens_test: np.ndarray
    r_star_test: np.ndarray
    params: dict[str, Any]


StrategyHandler = Callable[[EnsembleStrategyContext, Any], EnsembleStrategyResult]


def _strategy_min_aer(
    context: EnsembleStrategyContext, _options: Any
) -> EnsembleStrategyResult:
    params = {
        "strategy": 1,
        "rule": "choose model with minimum aER(x)",
    }
    p_ens_oof, r_star_oof = _apply_strategy_min_aer(
        context.proba_oof_stack,
        context.aer_oof_stack,
        context.oof_acc,
        context.slugs_ens,
    )
    p_ens_test, r_star_test = _apply_strategy_min_aer(
        context.proba_test_stack,
        context.aer_test_stack,
        context.oof_acc,
        context.slugs_ens,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_topn(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    top_n = int(options.aer_ens_top_n)
    params = {
        "strategy": 2,
        "rule": "mean probabilities over top-N models with lowest aER(x)",
        "top_n": int(top_n),
    }
    p_ens_oof, r_star_oof = _apply_strategy_topn(
        context.proba_oof_stack,
        context.aer_oof_stack,
        context.oof_acc,
        context.slugs_ens,
        top_n,
    )
    p_ens_test, r_star_test = _apply_strategy_topn(
        context.proba_test_stack,
        context.aer_test_stack,
        context.oof_acc,
        context.slugs_ens,
        top_n,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_acc_minus_aer(
    context: EnsembleStrategyContext, _options: Any
) -> EnsembleStrategyResult:
    params = {
        "strategy": 3,
        "rule": "choose model maximizing Acc_m - aER_m(x)",
        "acc_source": "oof_acc",
    }
    p_ens_oof, r_star_oof = _apply_strategy_acc_minus_aer(
        context.proba_oof_stack,
        context.aer_oof_stack,
        context.oof_acc,
        context.slugs_ens,
    )
    p_ens_test, r_star_test = _apply_strategy_acc_minus_aer(
        context.proba_test_stack,
        context.aer_test_stack,
        context.oof_acc,
        context.slugs_ens,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_exp_weighted(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    beta = float(options.aer_ens_beta)
    params = {
        "strategy": 4,
        "rule": "exp(-beta * aER(x)) weighted mean of probabilities",
        "beta": float(beta),
    }
    p_ens_oof, r_star_oof = _apply_strategy_exp_weighted(
        context.proba_oof_stack, context.aer_oof_stack, beta
    )
    p_ens_test, r_star_test = _apply_strategy_exp_weighted(
        context.proba_test_stack, context.aer_test_stack, beta
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_dynamic_threshold(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    mean_err_oof = np.mean(context.aer_oof_stack, axis=0)
    error_percentiles = (
        float(np.percentile(mean_err_oof, 25.0)),
        float(np.percentile(mean_err_oof, 50.0)),
        float(np.percentile(mean_err_oof, 75.0)),
    )
    tau0_opt = float(getattr(options, "aer_ens_tau0", 0.3))
    tau0_used = float(np.median(mean_err_oof)) if tau0_opt <= 0 else float(tau0_opt)
    params = {
        "strategy": 5,
        "rule": "select models with aER(x) <= t(x); mean probabilities",
        "tau0": float(tau0_used),
        "tau0_source": "median(mean_aer_oof)" if tau0_opt <= 0 else "cli",
        "difficulty_percentiles_oof": {
            "p25": float(error_percentiles[0]),
            "p50": float(error_percentiles[1]),
            "p75": float(error_percentiles[2]),
        },
        "easy_factor": 0.8,
        "hard_factor": 1.5,
    }
    p_ens_oof, r_star_oof = _apply_strategy_dynamic_threshold(
        context.proba_oof_stack,
        context.aer_oof_stack,
        base_threshold=tau0_used,
        error_percentiles=error_percentiles,
    )
    p_ens_test, r_star_test = _apply_strategy_dynamic_threshold(
        context.proba_test_stack,
        context.aer_test_stack,
        base_threshold=tau0_used,
        error_percentiles=error_percentiles,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_overconfidence_penalty(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    lam = float(getattr(options, "aer_ens_lambda", 0.5))
    params = {
        "strategy": 6,
        "rule": "choose model maximizing acc - lambda*aER(x)",
        "lambda": float(lam),
    }
    p_ens_oof, r_star_oof = _apply_strategy_overconfidence_penalty(
        context.proba_oof_stack,
        context.aer_oof_stack,
        context.oof_acc,
        context.slugs_ens,
        lam,
    )
    p_ens_test, r_star_test = _apply_strategy_overconfidence_penalty(
        context.proba_test_stack,
        context.aer_test_stack,
        context.oof_acc,
        context.slugs_ens,
        lam,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_calibration_aware(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    alpha = float(getattr(options, "aer_ens_alpha", 0.7))
    params = {
        "strategy": 7,
        "rule": "choose model maximizing alpha*acc - (1-alpha)*aER(x)",
        "alpha": float(alpha),
    }
    p_ens_oof, r_star_oof = _apply_strategy_calibration_aware(
        context.proba_oof_stack,
        context.aer_oof_stack,
        context.oof_acc,
        context.slugs_ens,
        alpha,
    )
    p_ens_test, r_star_test = _apply_strategy_calibration_aware(
        context.proba_test_stack,
        context.aer_test_stack,
        context.oof_acc,
        context.slugs_ens,
        alpha,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_trimmed_weighted(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    trim_q = float(getattr(options, "aer_ens_trim_q", 0.6))
    beta = float(options.aer_ens_beta)
    params = {
        "strategy": 8,
        "rule": "keep models with aER <= quantile_q; exp(-beta*aER) weighted mean",
        "trim_q": float(trim_q),
        "beta": float(beta),
    }
    p_ens_oof, r_star_oof = _apply_strategy_trimmed_weighted(
        context.proba_oof_stack,
        context.aer_oof_stack,
        float(beta),
        trim_q,
    )
    p_ens_test, r_star_test = _apply_strategy_trimmed_weighted(
        context.proba_test_stack,
        context.aer_test_stack,
        float(beta),
        trim_q,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_borda_rank(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    beta = float(options.aer_ens_beta)
    params = {
        "strategy": 9,
        "rule": "adaptive-error weighted Borda rank aggregation",
        "beta": float(beta),
    }
    p_ens_oof, r_star_oof = _apply_strategy_borda_rank(
        context.proba_oof_stack,
        context.aer_oof_stack,
        beta=float(beta),
    )
    p_ens_test, r_star_test = _apply_strategy_borda_rank(
        context.proba_test_stack,
        context.aer_test_stack,
        beta=float(beta),
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


def _strategy_switching_hybrid(
    context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    tau_low = float(getattr(options, "aer_ens_tau_low", 0.15))
    tau_high = float(getattr(options, "aer_ens_tau_high", 0.35))
    top_n = int(options.aer_ens_top_n)
    beta = float(options.aer_ens_beta)
    params = {
        "strategy": 10,
        "rule": (
            "s = min_i aER_i (per sample); if s <= tau_low: Strategy1; "
            "elif s <= tau_high: Top-N, w=exp(-beta*aER); "
            "else: w=acc*exp(-beta*aER)."
        ),
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "top_n": int(top_n),
        "beta": float(beta),
        "acc_source": "oof_acc",
    }
    p_ens_oof, r_star_oof = _apply_strategy_switching_hybrid(
        context.proba_oof_stack,
        context.aer_oof_stack,
        context.oof_acc,
        context.slugs_ens,
        beta=float(beta),
        top_n=int(top_n),
        tau_low=tau_low,
        tau_high=tau_high,
    )
    p_ens_test, r_star_test = _apply_strategy_switching_hybrid(
        context.proba_test_stack,
        context.aer_test_stack,
        context.oof_acc,
        context.slugs_ens,
        beta=float(beta),
        top_n=int(top_n),
        tau_low=tau_low,
        tau_high=tau_high,
    )
    return EnsembleStrategyResult(
        p_ens_oof=p_ens_oof,
        r_star_oof=r_star_oof,
        p_ens_test=p_ens_test,
        r_star_test=r_star_test,
        params=params,
    )


ENSEMBLE_STRATEGY_REGISTRY: dict[str, StrategyHandler] = {
    "strategy1_min_aer": _strategy_min_aer,
    "strategy2_topn": _strategy_topn,
    "strategy3_acc_minus_aer": _strategy_acc_minus_aer,
    "strategy4_exp_weighted": _strategy_exp_weighted,
    "strategy5_dynamic_threshold": _strategy_dynamic_threshold,
    "strategy6_overconfidence_penalty": _strategy_overconfidence_penalty,
    "strategy7_calibration_aware": _strategy_calibration_aware,
    "strategy8_trimmed_weighted": _strategy_trimmed_weighted,
    "strategy9_borda_rank": _strategy_borda_rank,
    "strategy10_switching_hybrid": _strategy_switching_hybrid,
}


def run_ensemble_strategy(
    strategy_key: str, context: EnsembleStrategyContext, options: Any
) -> EnsembleStrategyResult:
    handler = ENSEMBLE_STRATEGY_REGISTRY.get(strategy_key)
    if handler is None:
        raise ValueError(f"Unknown strategy: {strategy_key}")
    return handler(context, options)
