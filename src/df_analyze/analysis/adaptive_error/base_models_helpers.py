from __future__ import annotations

from typing import (
    Any,
    Optional,
)

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.confidence_metrics import (
    ConfidenceMetricParams,
    apply_exp_gamma,
    apply_minmax,
)


def _apply_confidence_transform(
    raw: Optional[np.ndarray],
    params: dict[str, Any],
) -> Optional[np.ndarray]:
    if raw is None:
        return None
    params_obj = ConfidenceMetricParams(
        kind=str(params.get("kind", "identity")),
        params=dict(params.get("params", {})),
    )
    if params_obj.kind == "minmax":
        return apply_minmax(raw, params_obj)
    if params_obj.kind == "exp_gamma":
        return apply_exp_gamma(raw, params_obj)
    conf = np.asarray(raw, dtype=float).ravel()
    conf = np.clip(conf, 0.0, 1.0)
    conf[~np.isfinite(conf)] = np.nan
    return conf


def _decode_labels(y: np.ndarray, labels: Optional[dict[int, str]]) -> np.ndarray:
    if not labels:
        return y
    s = pd.Series(y)
    if pd.api.types.is_numeric_dtype(s):
        mapped = s.astype(int).map(labels)
    else:
        mapped = s.map(labels)
    out = mapped.fillna(s).to_numpy()
    return out


def _proba_to_list(proba: np.ndarray) -> list[list[float]]:
    return [list(map(float, row)) for row in np.asarray(proba)]
