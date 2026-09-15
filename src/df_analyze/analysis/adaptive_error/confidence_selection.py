"""select confidence metric by minimizing cross-fitted Brier score on OOF"""
# Ref: Brier score loss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional
from warnings import warn

import numpy as np
import pandas as pd

from df_analyze._constants import SEED
from df_analyze.splitting import OmniKFold

from df_analyze.analysis.adaptive_error.aer import AdaptiveErrorCalculator


Array = np.ndarray


def brier_score(prob: Array, y_true01: Array) -> float:
    p = np.asarray(prob, dtype=float).ravel()
    y = np.asarray(y_true01, dtype=float).ravel()
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    p = np.clip(p[mask], 0.0, 1.0)
    y = np.clip(y[mask], 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def _crossfit_predicted_error(
    conf: Array,
    incorrect: Array,
    *,
    groups: Optional[Array] = None,
    n_splits: int,
    seed: int,
    aer_kwargs: dict[str, Any],
) -> Array:
    c = np.asarray(conf, dtype=float).ravel()
    inc = np.asarray(incorrect, dtype=int).ravel()
    g = None
    if groups is not None:
        g = np.asarray(groups).ravel()
        if g.shape[0] != c.shape[0]:
            raise ValueError("groups must have same length as conf")

    if c.shape[0] != inc.shape[0]:
        raise ValueError("conf and incorrect must have same length")

    mask = np.isfinite(c) & np.isfinite(inc)
    if g is not None:
        mask = mask & (~pd.isna(g))
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.full_like(c, np.nan, dtype=float)

    c_valid = c[idx]
    inc_valid = inc[idx]
    g_valid = g[idx] if g is not None else None

    if np.unique(inc_valid).size < 2 or n_splits < 2 or idx.size < n_splits:
        aer = AdaptiveErrorCalculator(**aer_kwargs)
        aer.fit(confidences=c_valid, incorrect=inc_valid)
        pred = np.full_like(c, np.nan, dtype=float)
        pred[idx] = aer.get_expected_error(c_valid)
        return pred

    pred_valid = np.full_like(c_valid, np.nan, dtype=float)
    X_df = pd.DataFrame({"conf": c_valid})
    y_ser = pd.Series(inc_valid)
    g_ser = pd.Series(g_valid) if g_valid is not None else None

    splitter = OmniKFold(
        n_splits=int(n_splits),
        is_classification=True,
        grouped=g_valid is not None,
        shuffle=True,
        seed=int(seed),
        warn_on_fallback=True,
        df_analyze_phase="adaptive_error_confidence_selection",
    )
    splits, did_fail = splitter.split(X_df, y_ser, g_ser)
    if did_fail and g_valid is not None:
        warn(
            "Grouped split failed during confidence-metric selection; "
            "falling back to a non-grouped split may introduce optimistic bias."
        )

    for trn, val in splits:
        aer = AdaptiveErrorCalculator(**aer_kwargs)
        aer.fit(confidences=c_valid[trn], incorrect=inc_valid[trn])
        pred_valid[val] = aer.get_expected_error(c_valid[val])

    pred = np.full_like(c, np.nan, dtype=float)
    pred[idx] = pred_valid
    return pred


def crossfit_predicted_error(
    conf: Array,
    incorrect: Array,
    *,
    groups: Optional[Array] = None,
    n_splits: int,
    seed: int,
    aer_kwargs: dict[str, Any],
) -> Array:
    return _crossfit_predicted_error(
        conf,
        incorrect,
        groups=groups,
        n_splits=n_splits,
        seed=seed,
        aer_kwargs=aer_kwargs,
    )


@dataclass(frozen=True)
class ConfidenceMetricScore:
    name: str
    brier: float
    n_valid: int


@dataclass(frozen=True)
class ConfidenceMetricSelection:
    selected: str
    scores: list[ConfidenceMetricScore]
    requested: Optional[str] = None
    requested_valid: Optional[bool] = None
    selection_mode: str = "auto"

    def to_json(self) -> dict[str, Any]:
        payload = {
            "selected": self.selected,
            "scores": [
                {"name": s.name, "brier": s.brier, "n_valid": s.n_valid}
                for s in self.scores
            ],
        }
        if self.requested is not None:
            payload["requested"] = self.requested
            payload["requested_valid"] = bool(self.requested_valid)
            payload["selection_mode"] = self.selection_mode
        return payload


def _normalize_metric_name(name: str) -> str:
    s = str(name).strip().lower().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    return s


def _resolve_requested_metric(
    requested: str, candidates: dict[str, Array]
) -> Optional[str]:
    norm = _normalize_metric_name(requested)
    if norm in ("", "auto", "best", "select", "default", "none"):
        return None
    if norm in candidates:
        return norm
    alias_map = {
        "margin": "proba_margin",
        "p_margin": "proba_margin",
        "prob_margin": "proba_margin",
        "tree_vote": "tree_vote_agreement",
        "vote_agreement": "tree_vote_agreement",
        "tree_leaf": "tree_leaf_support",
        "leaf_support": "tree_leaf_support",
        "knn_dist": "knn_dist_weighted",
        "knn_weighted": "knn_dist_weighted",
        "knn_min": "knn_min_dist",
        "knn_min_distance": "knn_min_dist",
    }
    alias = alias_map.get(norm)
    if alias in candidates:
        return alias
    return None


def select_confidence_metric(
    candidates: dict[str, Array],
    incorrect: Array,
    *,
    groups: Optional[Array] = None,
    aer_kwargs: dict[str, Any],
    n_splits: int = 5,
    seed: int = SEED,
    metric: Optional[str] = None,
) -> ConfidenceMetricSelection:
    inc = np.asarray(incorrect, dtype=int).ravel()

    scores: list[ConfidenceMetricScore] = []
    for name, conf in candidates.items():
        c = np.asarray(conf, dtype=float).ravel()
        if c.shape[0] != inc.shape[0]:
            continue
        mask = np.isfinite(c)
        n_valid = int(mask.sum())
        if n_valid == 0:
            scores.append(ConfidenceMetricScore(name=name, brier=float("nan"), n_valid=0))
            continue
        pred_err = _crossfit_predicted_error(
            c,
            inc,
            groups=groups,
            n_splits=int(max(2, n_splits)),
            seed=int(seed),
            aer_kwargs=aer_kwargs,
        )
        b = brier_score(pred_err, inc)
        scores.append(ConfidenceMetricScore(name=name, brier=b, n_valid=n_valid))

    finite = [s for s in scores if np.isfinite(s.brier)]
    if finite:
        best = min(finite, key=lambda s: float(s.brier))
        selected = best.name
    else:
        selected = next(iter(candidates.keys()))

    requested_raw = None
    requested_valid = None
    selection_mode = "auto"
    if metric is not None:
        requested_raw = str(metric)
        resolved = _resolve_requested_metric(requested_raw, candidates)
        if resolved is not None:
            selected = resolved
            requested_valid = True
            selection_mode = "fixed"
        else:
            norm = _normalize_metric_name(requested_raw)
            if norm in (
                "",
                "auto",
                "best",
                "select",
                "default",
                "none",
            ):
                requested_valid = True
            else:
                requested_valid = False
                warn(
                    f"Requested confidence metric '{requested_raw}' not available; "
                    f"using '{selected}'."
                )

    scores_sorted = sorted(
        scores,
        key=lambda s: (not np.isfinite(s.brier), s.brier, s.name),
    )
    return ConfidenceMetricSelection(
        selected=selected,
        scores=scores_sorted,
        requested=requested_raw,
        requested_valid=requested_valid,
        selection_mode=selection_mode,
    )
