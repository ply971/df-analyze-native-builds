from __future__ import annotations

from typing import Any
from warnings import warn


def _split_strategy_tokens(value: str) -> list[str]:
    cleaned = " ".join(value.split())
    if not cleaned:
        return []
    return [token.strip() for token in cleaned.split(",") if token.strip()]


def _normalize_strategy_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = _split_strategy_tokens(value)
    else:
        try:
            raw = list(value)
        except TypeError:
            raw = [value]
    tokens: list[str] = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, str):
            parts = _split_strategy_tokens(item)
        else:
            parts = [str(item).strip()]
        tokens.extend([p for p in parts if p])
    return [t.lower() for t in tokens if t]


def _build_ensemble_strategy_params(options: Any) -> dict[str, Any]:
    tau0_opt = float(getattr(options, "aer_ens_tau0", 0.3))
    return {
        "top_n": int(options.aer_ens_top_n),
        "beta": float(options.aer_ens_beta),
        "tau0": "auto" if tau0_opt <= 0 else float(tau0_opt),
        "lambda": float(getattr(options, "aer_ens_lambda", 0.5)),
        "alpha": float(getattr(options, "aer_ens_alpha", 0.7)),
        "trim_q": float(getattr(options, "aer_ens_trim_q", 0.6)),
        "tau_low": float(getattr(options, "aer_ens_tau_low", 0.15)),
        "tau_high": float(getattr(options, "aer_ens_tau_high", 0.35)),
    }


def _build_strategy_labels(options: Any) -> dict[str, str]:
    params = _build_ensemble_strategy_params(options)
    tau0 = params["tau0"]
    if isinstance(tau0, str):
        dynamic_threshold_label = "dynamic_threshold_tau_auto"
    else:
        dynamic_threshold_label = f"dynamic_threshold_tau{float(tau0):g}"
    top_n = int(params["top_n"])
    beta = float(params["beta"])
    lam = float(params["lambda"])
    alpha = float(params["alpha"])
    trim_q = float(params["trim_q"])
    tau_low = float(params["tau_low"])
    tau_high = float(params["tau_high"])
    all_labels = {
        "strategy1_min_aer": "per_sample_min_expected_error",
        "strategy2_topn": f"top_n_min_expected_error_mean_n{top_n}",
        "strategy3_acc_minus_aer": "accuracy_minus_expected_error",
        "strategy4_exp_weighted": f"risk_weighted_mean_beta{beta:g}",
        "strategy5_dynamic_threshold": dynamic_threshold_label,
        "strategy6_overconfidence_penalty": f"overconfidence_penalty_lambda{lam:g}",
        "strategy7_calibration_aware": f"accuracy_risk_tradeoff_alpha{alpha:g}",
        "strategy8_trimmed_weighted": f"trimmed_weighted_q{trim_q:g}_beta{beta:g}",
        "strategy9_borda_rank": f"borda_rank_beta{beta:g}",
        "strategy10_switching_hybrid": (
            f"switching_hybrid_tau{tau_low:g}_{tau_high:g}_n{top_n}_beta{beta:g}"
        ),
    }
    requested = _normalize_strategy_tokens(
        getattr(options, "aer_ensemble_strategies", None)
    )
    if not requested or "all" in requested or "*" in requested:
        return all_labels

    ordered_keys = list(all_labels.keys())
    label_to_key = {v.lower(): k for k, v in all_labels.items()}
    legacy_calibration_label = f"calibration_aware_alpha{alpha:g}".lower()
    label_to_key.setdefault(legacy_calibration_label, "strategy7_calibration_aware")
    alias_to_key = {
        "min_aer": "strategy1_min_aer",
        "min_expected_error": "strategy1_min_aer",
        "topn": "strategy2_topn",
        "top_n": "strategy2_topn",
        "acc_minus_aer": "strategy3_acc_minus_aer",
        "accuracy_minus_expected_error": "strategy3_acc_minus_aer",
        "exp_weighted": "strategy4_exp_weighted",
        "risk_weighted": "strategy4_exp_weighted",
        "dynamic_threshold": "strategy5_dynamic_threshold",
        "overconfidence_penalty": "strategy6_overconfidence_penalty",
        "calibration_aware": "strategy7_calibration_aware",
        "trimmed_weighted": "strategy8_trimmed_weighted",
        "borda": "strategy9_borda_rank",
        "borda_rank": "strategy9_borda_rank",
        "switching_hybrid": "strategy10_switching_hybrid",
        "hybrid": "strategy10_switching_hybrid",
    }
    selected: list[str] = []
    unknown: list[str] = []
    for token in requested:
        canonical = alias_to_key.get(token, token)
        if canonical in all_labels:
            key = canonical
        elif canonical in label_to_key:
            key = label_to_key[canonical]
        elif canonical.startswith("strategy") and canonical[8:].isdigit():
            idx = int(canonical[8:]) - 1
            if 0 <= idx < len(ordered_keys):
                key = ordered_keys[idx]
            else:
                unknown.append(token)
                continue
        elif canonical.isdigit():
            idx = int(canonical) - 1
            if 0 <= idx < len(ordered_keys):
                key = ordered_keys[idx]
            else:
                unknown.append(token)
                continue
        else:
            unknown.append(token)
            continue
        if key not in selected:
            selected.append(key)

    if not selected:
        warn(
            "No valid aer_ensemble_strategies matched for requested tokens: "
            + ", ".join(sorted(set(requested)))
            + ". No ensemble strategies will run."
        )
        return {}
    if unknown:
        warn(
            "Ignoring unknown aer_ensemble_strategies: "
            + ", ".join(sorted(set(unknown)))
            + ". Running only matched strategies."
        )

    return {k: all_labels[k] for k in ordered_keys if k in selected}
