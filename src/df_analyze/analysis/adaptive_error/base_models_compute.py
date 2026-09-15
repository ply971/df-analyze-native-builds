from __future__ import annotations

from typing import (
    Any,
    Optional,
)

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.confidence_selection import (
    crossfit_predicted_error,
)


def _crossfit_oof_risk(
    *,
    conf_oof: np.ndarray,
    incorrect_oof: np.ndarray,
    groups_arr,
    row_id,
    n_splits: int,
    seed: int,
    aer_kwargs: dict[str, Any],
) -> tuple[Optional[np.ndarray], str, dict[str, Any]]:
    oof_risk_source = "naive"
    e_hat_oof_cv = None
    diag: dict[str, Any] = {
        "status": "DONE",
        "reason": "",
        "condition": "",
        "source": oof_risk_source,
        "n_total": int(np.asarray(conf_oof).shape[0]),
        "n_valid": 0,
        "n_splits": int(n_splits),
        "unique_incorrect": None,
        "grouped": False,
    }
    groups_for_risk = None
    if groups_arr is not None:
        if row_id is not None and hasattr(groups_arr, "reindex"):
            groups_for_risk = groups_arr.reindex(row_id)
        else:
            groups_for_risk = groups_arr
        try:
            groups_for_risk = np.asarray(groups_for_risk)
        except (TypeError, ValueError):
            groups_for_risk = None
    if (
        groups_for_risk is not None
        and groups_for_risk.shape[0] != conf_oof.shape[0]
    ):
        groups_for_risk = None
    diag["grouped"] = groups_for_risk is not None

    mask = np.isfinite(conf_oof) & np.isfinite(incorrect_oof)
    if groups_for_risk is not None:
        mask = mask & (~pd.isna(groups_for_risk))
    diag["n_valid"] = int(mask.sum())
    uniq = int(np.unique(incorrect_oof[mask]).size)
    diag["unique_incorrect"] = uniq
    can_crossfit = (
        n_splits >= 2
        and int(mask.sum()) >= n_splits
        and uniq >= 2
    )
    if can_crossfit:
        e_hat_oof_cv = crossfit_predicted_error(
            conf_oof,
            incorrect_oof,
            groups=groups_for_risk,
            n_splits=n_splits,
            seed=seed,
            aer_kwargs=aer_kwargs,
        )
        if np.isfinite(e_hat_oof_cv).sum() > 0:
            oof_risk_source = "crossfit"
        else:
            e_hat_oof_cv = None
            diag["status"] = "FALLBACK"
            diag["condition"] = "crossfit_nonfinite"
            diag["reason"] = "Cross-fit risk returned all non-finite values."
    else:
        diag["status"] = "FALLBACK"
        diag["condition"] = "insufficient_samples_or_variance"
        diag["reason"] = (
            "Cross-fit risk disabled due to insufficient samples or "
            "non-variant incorrect labels."
        )

    diag["source"] = oof_risk_source
    return e_hat_oof_cv, oof_risk_source, diag
