"""probability calibration helpers for adaptive error analysis, uses out-of-fold predictions as calibration data
"""
# ref: Probability calibration (sigmoid/Platt, isotonic, CV calibration): https://scikit-learn.org/stable/modules/calibration.html
# ref: Temperature scaling: https://arxiv.org/abs/1706.04599

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from df_analyze._constants import SEED
from df_analyze.splitting import OmniKFold

CalibratorMethod = Literal[
    "none",
    "temperature",
    "platt",
    "isotonic",
    "isotonic_ovr",
]


@dataclass
class ProbaCalibrator:
    method: CalibratorMethod
    n_classes: int
    temperature: Optional[float] = None
    platt_coef: Optional[tuple[float, float]] = None
    iso_model: Optional[Any] = None
    models: Optional[list[Any]] = None

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = normalize_proba(proba)
        if self.method == "none":
            return p

        if self.method == "temperature":
            if self.temperature is None:
                raise ValueError("Temperature calibrator missing temperature.")
            return _apply_temperature(p, self.temperature)

        if self.method == "platt":
            if self.platt_coef is None:
                raise ValueError("Platt calibrator missing coefficients.")
            if p.shape[1] != 2:
                raise ValueError("Platt scaling requires binary probabilities.")
            p1 = np.clip(p[:, 1], 1e-12, 1.0 - 1e-12)
            logit = _logit(p1)
            a, b = self.platt_coef
            q1 = _sigmoid(a * logit + b)
            return np.column_stack([1.0 - q1, q1])

        if self.method == "isotonic":
            if self.iso_model is None:
                raise ValueError("Isotonic calibrator missing fitted model.")
            if p.shape[1] != 2:
                raise ValueError("Isotonic scaling requires binary probabilities.")
            p1 = np.clip(p[:, 1], 0.0, 1.0)
            q1 = np.asarray(self.iso_model.predict(p1), dtype=float)
            q1 = np.clip(q1, 0.0, 1.0)
            return np.column_stack([1.0 - q1, q1])

        if self.method == "isotonic_ovr":
            if self.models is None:
                raise ValueError("Isotonic OVR calibrator missing models.")
            q = np.empty_like(p, dtype=float)
            for k in range(self.n_classes):
                q[:, k] = np.asarray(self.models[k].predict(p[:, k]), dtype=float)
            return normalize_proba(q)

        raise ValueError(f"Invalid calibrator method: {self.method}")

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "n_classes": int(self.n_classes),
        }
        if self.method == "temperature":
            payload["temperature"] = (
                None if self.temperature is None else float(self.temperature)
            )
        elif self.method == "platt":
            if self.platt_coef is not None:
                payload["platt_a"] = float(self.platt_coef[0])
                payload["platt_b"] = float(self.platt_coef[1])
        elif self.method == "isotonic":
            if self.iso_model is not None:
                payload["x_min"] = float(getattr(self.iso_model, "X_min_", np.nan))
                payload["x_max"] = float(getattr(self.iso_model, "X_max_", np.nan))
        elif self.method == "isotonic_ovr" and self.models is not None:
            payload["per_class"] = [
                {
                    "x_min": float(getattr(m, "X_min_", np.nan)),
                    "x_max": float(getattr(m, "X_max_", np.nan)),
                }
                for m in self.models
            ]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2)


def predict_proba_or_scores(
    estimator, X
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        return np.asarray(estimator.predict_proba(X)), None
    except (AttributeError, TypeError, ValueError):
        try:
            return None, np.asarray(estimator.decision_function(X))
        except (AttributeError, TypeError, ValueError):
            return None, None


def predict_scores(estimator, X) -> Optional[np.ndarray]:
    try:
        scores = estimator.decision_function(X)
    except (AttributeError, TypeError, ValueError):
        return None
    return np.asarray(scores)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = normalize_proba(proba)
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        return p
    logp = np.log(np.clip(p, 1e-12, 1.0))
    scaled = np.exp(logp / temp)
    return normalize_proba(scaled)


def scores_to_proba(scores: np.ndarray) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.ndim == 1:
        p1 = _sigmoid(scores_arr)
        return np.column_stack([1.0 - p1, p1])
    if scores_arr.ndim != 2:
        raise ValueError(f"Expected scores to be 1D or 2D, got shape {scores_arr.shape}")
    if scores_arr.shape[1] == 1:
        p1 = _sigmoid(scores_arr[:, 0])
        return np.column_stack([1.0 - p1, p1])
    return _softmax(scores_arr)


def _proba_drift_rate(proba: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    p = normalize_proba(proba)
    y = np.asarray(y_pred).ravel()
    if p.ndim != 2 or p.shape[0] != y.shape[0]:
        return None
    return float(np.mean(np.argmax(p, axis=1) != y))


def align_proba_with_predictions(
    proba: np.ndarray,
    y_pred: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    *,
    drift_threshold: float = 0.02,
) -> np.ndarray:
    p = normalize_proba(proba)
    if y_pred is None or scores is None:
        return p
    drift = _proba_drift_rate(p, y_pred)
    if drift is None or drift <= drift_threshold:
        return p
    p_scores = normalize_proba(scores_to_proba(scores))
    drift_scores = _proba_drift_rate(p_scores, y_pred)
    if drift_scores is not None and drift_scores < drift:
        return p_scores
    return p


def ensure_proba_2d(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p1 = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])
    if p.ndim != 2:
        raise ValueError(f"Expected proba to be 1D or 2D, got shape {p.shape}")
    if p.shape[1] == 1:
        p1 = np.clip(p[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])
    return p


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    p = ensure_proba_2d(proba)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    row_sum = p.sum(axis=1, keepdims=True)
    n_classes = p.shape[1]
    safe = np.isfinite(row_sum) & (row_sum > 0)
    safe_rows = safe[:, 0]
    out = np.empty_like(p)
    out[safe_rows] = p[safe_rows] / row_sum[safe_rows]
    out[~safe_rows] = 1.0 / n_classes
    return out


def fit_temperature_calibrator(
    proba_oof: np.ndarray, y_true: np.ndarray, *, temps: Optional[Iterable[float]] = None
) -> ProbaCalibrator:
    p = normalize_proba(proba_oof)
    y = np.asarray(y_true).ravel()
    n_classes = int(p.shape[1])

    if temps is None:
        temps_arr = np.logspace(-2, 2, 25)
    else:
        temps_arr = np.asarray(list(temps), dtype=float)
    best_t = 1.0
    best_ll = float("inf")

    for t in temps_arr:
        if not np.isfinite(t) or t <= 0.0:
            continue
        p_t = _apply_temperature(p, float(t))
        ll = float(log_loss(y, p_t, labels=list(range(n_classes))))
        if ll < best_ll:
            best_ll = ll
            best_t = float(t)

    if not np.isfinite(best_ll):
        raise RuntimeError("Temperature scaling failed to compute log-loss.")

    return ProbaCalibrator(
        method="temperature", n_classes=n_classes, temperature=float(best_t)
    )


def fit_platt_calibrator(proba_oof: np.ndarray, y_true: np.ndarray) -> ProbaCalibrator:

    from sklearn.linear_model import LogisticRegression
    p = normalize_proba(proba_oof)
    if p.shape[1] != 2:
        raise ValueError("Platt scaling requires binary probabilities.")

    y = np.asarray(y_true).ravel()
    p1 = np.clip(p[:, 1], 1e-12, 1.0 - 1e-12)
    logit = _logit(p1).reshape(-1, 1)

    classes = np.unique(y)
    if classes.size < 2:
        prevalence = float(np.mean(y))
        prevalence = float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))
        return ProbaCalibrator(
            method="platt",
            n_classes=2,
            platt_coef=(0.0, float(_logit(prevalence))),
        )

    lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    lr.fit(logit, y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return ProbaCalibrator(method="platt", n_classes=2, platt_coef=(a, b))

def fit_isotonic_calibrator(proba_oof: np.ndarray, y_true: np.ndarray) -> ProbaCalibrator:
    from sklearn.isotonic import IsotonicRegression

    p = normalize_proba(proba_oof)
    if p.shape[1] != 2:
        raise ValueError("Isotonic calibration requires binary probabilities.")

    y = np.asarray(y_true).ravel()
    p1 = np.clip(p[:, 1], 0.0, 1.0)
    iso = IsotonicRegression(out_of_bounds="clip")
    if np.all(y == 0) or np.all(y == 1):
        mean = float(np.mean(y))
        iso.fit([0.0, 1.0], [mean, mean])
    else:
        iso.fit(p1, y)
    return ProbaCalibrator(method="isotonic", n_classes=2, iso_model=iso)

def fit_isotonic_ovr_calibrator(
    proba_oof: np.ndarray, y_true: np.ndarray
) -> ProbaCalibrator:
    from sklearn.isotonic import IsotonicRegression

    p = normalize_proba(proba_oof)
    y = np.asarray(y_true).ravel()
    n_classes = int(p.shape[1])

    models: list[Any] = []
    for k in range(n_classes):
        yk = (y == k).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip")
        if np.all(yk == 0) or np.all(yk == 1):
            mean = float(yk.mean())
            iso.fit([0.0, 1.0], [mean, mean])
        else:
            iso.fit(p[:, k], yk)
        models.append(iso)

    return ProbaCalibrator(method="isotonic_ovr", n_classes=n_classes, models=models)


def _fit_calibrator(
    method: CalibratorMethod, proba: np.ndarray, y_true: np.ndarray
) -> ProbaCalibrator:
    if method == "none":
        p = normalize_proba(proba)
        return ProbaCalibrator(method="none", n_classes=int(p.shape[1]))
    if method == "temperature":
        return fit_temperature_calibrator(proba, y_true)
    if method == "platt":
        return fit_platt_calibrator(proba, y_true)
    if method == "isotonic":
        return fit_isotonic_calibrator(proba, y_true)
    if method == "isotonic_ovr":
        return fit_isotonic_ovr_calibrator(proba, y_true)
    raise ValueError(f"Unknown calibrator method: {method}")


def select_best_proba_calibrator(
    proba_oof: np.ndarray,
    y_true: np.ndarray,
    *,
    methods: Optional[tuple[CalibratorMethod, ...]] = None,
    n_splits: int = 3,
    seed: int = SEED,
    groups: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    drift_threshold: Optional[float] = 0.02,
) -> tuple[ProbaCalibrator, dict[str, Any]]:
    p = normalize_proba(proba_oof)
    y = np.asarray(y_true).ravel()
    n_classes = int(p.shape[1])

    if methods is None:
        if n_classes > 2:
            requested = ["none", "temperature", "isotonic_ovr"]
        else:
            requested = ["none", "temperature", "platt", "isotonic"]
    else:
        requested = list(methods)
    candidate_methods: list[CalibratorMethod] = []
    skipped: list[dict[str, str]] = []
    for m in requested:
        if m in ("platt", "isotonic") and n_classes != 2:
            skipped.append({"method": m, "reason": "binary_only"})
            continue
        if m == "isotonic_ovr" and n_classes <= 2:
            skipped.append({"method": m, "reason": "multiclass_only"})
            continue
        candidate_methods.append(m)
    if not candidate_methods:
        candidate_methods = ["none"]

    best: Optional[ProbaCalibrator] = None
    best_ll = float("inf")
    diag: dict[str, Any] = {"candidates": [], "skipped": skipped}

    n_samples = int(y.size)
    n_splits_eff = max(2, min(int(n_splits), n_samples)) if n_samples > 1 else 0
    splits: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
    grouped = False

    if n_splits_eff >= 2 and n_samples >= 2:
        y_series = pd.Series(y)
        X_dummy = pd.DataFrame({"row_id": np.arange(n_samples)})
        g_series = None
        if groups is not None:
            g_arr = np.asarray(groups).ravel()
            if g_arr.shape[0] == n_samples:
                g_series = pd.Series(g_arr)
        grouped = g_series is not None
        kf = OmniKFold(
            n_splits=n_splits_eff,
            is_classification=True,
            grouped=grouped,
            shuffle=True,
            seed=seed,
            warn_on_fallback=True,
            df_analyze_phase="Proba calibrator selection",
        )
        splits, _ = kf.split(X_train=X_dummy, y_train=y_series, g_train=g_series)

    diag["cv"] = {
        "n_splits": int(n_splits_eff),
        "seed": int(seed),
        "grouped": grouped,
    }

    candidates: list[dict[str, Any]] = []
    for m in candidate_methods:
        fold_losses: list[float] = []

        if splits:
            for tr_idx, va_idx in splits:
                p_tr, y_tr = p[tr_idx], y[tr_idx]
                p_va, y_va = p[va_idx], y[va_idx]
                cal = _fit_calibrator(m, p_tr, y_tr)
                p_cal = cal.transform(p_va)
                ll = float(log_loss(y_va, p_cal, labels=list(range(n_classes))))
                fold_losses.append(ll)
        else:
            cal = _fit_calibrator(m, p, y)
            p_cal = cal.transform(p)
            fold_losses.append(
                float(log_loss(y, p_cal, labels=list(range(n_classes))))
            )

        ll_mean = float(np.mean(fold_losses)) if fold_losses else float("inf")

        full_cal: Optional[ProbaCalibrator] = None
        ll_full: Optional[float] = None
        drift_rate: Optional[float] = None
        full_cal = _fit_calibrator(m, p, y)
        p_full = full_cal.transform(p)
        ll_full = float(log_loss(y, p_full, labels=list(range(n_classes))))
        if y_pred is not None:
            y_pred_arr = np.asarray(y_pred).ravel()
            if y_pred_arr.shape[0] == p_full.shape[0]:
                drift_rate = float(np.mean(np.argmax(p_full, axis=1) != y_pred_arr))

        diag_entry: dict[str, Any] = {
            "method": m,
            "log_loss": ll_mean,
            "log_loss_cv": ll_mean if splits else None,
            "log_loss_full": ll_full,
            "fold_log_loss": fold_losses if splits else None,
            "drift": drift_rate,
        }
        diag["candidates"].append(diag_entry)
        candidates.append(
            {
                "method": m,
                "log_loss": ll_mean,
                "calibrator": full_cal,
                "drift": drift_rate,
            }
        )

    valid = [c for c in candidates if np.isfinite(c["log_loss"])]
    drift_guard_applied = False
    if y_pred is not None and drift_threshold is not None and valid:
        eligible = [
            c
            for c in valid
            if c["drift"] is not None and c["drift"] <= float(drift_threshold)
        ]
        if eligible:
            drift_guard_applied = True
            chosen = min(eligible, key=lambda c: c["log_loss"])
        else:
            chosen = min(valid, key=lambda c: c["log_loss"])
    elif valid:
        chosen = min(valid, key=lambda c: c["log_loss"])
    else:
        chosen = None

    if chosen is None:
        best = ProbaCalibrator(method="none", n_classes=n_classes, models=None)
        diag["selected"] = {"method": "none", "log_loss": None}
    else:
        best = chosen["calibrator"]
        if best is None:
            best = _fit_calibrator(chosen["method"], p, y)
        best_ll = float(chosen["log_loss"])
        diag["selected"] = {
            "method": best.method,
            "log_loss": best_ll,
            "drift_guard_applied": drift_guard_applied,
            "drift_threshold": None
            if drift_threshold is None
            else float(drift_threshold),
        }

    return best, diag


def crossfit_calibrated_proba(
    proba_oof: np.ndarray,
    y_true: np.ndarray,
    *,
    method: CalibratorMethod,
    n_splits: int = 3,
    seed: int = SEED,
    groups: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    p = normalize_proba(proba_oof)
    y = np.asarray(y_true).ravel()
    if p.shape[0] != y.shape[0]:
        raise ValueError("proba_oof and y_true must have the same length")

    diag: dict[str, Any] = {
        "method": method,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "grouped": False,
        "fallback": None,
    }

    if method == "none":
        diag["mode"] = "identity"
        return p, diag

    n_samples = int(y.size)
    n_splits_eff = max(2, min(int(n_splits), n_samples)) if n_samples > 1 else 0
    splits: Optional[list[tuple[np.ndarray, np.ndarray]]] = None

    if n_splits_eff >= 2 and n_samples >= 2:
        y_series = pd.Series(y)
        X_dummy = pd.DataFrame({"row_id": np.arange(n_samples)})
        g_series = None
        if groups is not None:
            g_arr = np.asarray(groups).ravel()
            if g_arr.shape[0] == n_samples:
                g_series = pd.Series(g_arr)
        diag["grouped"] = g_series is not None

        kf = OmniKFold(
            n_splits=n_splits_eff,
            is_classification=True,
            grouped=g_series is not None,
            shuffle=True,
            seed=seed,
            warn_on_fallback=True,
            df_analyze_phase="Proba calibrator crossfit",
        )
        splits, _ = kf.split(X_train=X_dummy, y_train=y_series, g_train=g_series)

    if not splits:
        cal = _fit_calibrator(method, p, y)
        diag["fallback"] = "full_fit"
        return cal.transform(p), diag

    p_cal = np.full_like(p, np.nan, dtype=float)
    used = np.zeros(n_samples, dtype=bool)

    for tr_idx, va_idx in splits:
        cal = _fit_calibrator(method, p[tr_idx], y[tr_idx])
        p_cal[va_idx] = cal.transform(p[va_idx])
        used[va_idx] = True

    if not used.all():
        diag["fallback"] = "partial_identity"
        diag["n_unfilled"] = int((~used).sum())
        p_cal[~used] = p[~used]

    return normalize_proba(p_cal), diag
