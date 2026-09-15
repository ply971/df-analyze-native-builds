"""adaptive Error Rate ensemble strategies"""

from __future__ import annotations

import numpy as np


def _sorted_model_indices(
    aer_vals: np.ndarray, oof_acc: np.ndarray, slugs: list[str]
) -> list[int]:
    idxs = list(range(len(slugs)))

    def _key(m: int) -> tuple[float, float, str]:
        aer_val = float(aer_vals[m])
        acc_val = float(oof_acc[m])
        if not np.isfinite(acc_val):
            acc_val = -np.inf
        return (aer_val, -acc_val, slugs[m])

    return sorted(idxs, key=_key)


def _apply_strategy_min_aer(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    oof_acc: np.ndarray,
    slugs: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        order = _sorted_model_indices(aer_stack[:, i], oof_acc, slugs)
        m_idx = order[0]
        p_ens[i] = proba_stack[m_idx, i]
        r_star[i] = aer_stack[m_idx, i]
    return p_ens, r_star


def _apply_strategy_topn(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    oof_acc: np.ndarray,
    slugs: list[str],
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    n_use = max(1, min(int(top_n), n_models))
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        order = _sorted_model_indices(aer_stack[:, i], oof_acc, slugs)
        sel = order[:n_use]
        p_ens[i] = np.mean(proba_stack[sel, i], axis=0)
        r_star[i] = float(np.mean(aer_stack[sel, i]))
    return p_ens, r_star


def _apply_strategy_exp_weighted(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_models = proba_stack.shape[0]
    aer_clip = np.clip(aer_stack, 1e-9, 1.0)
    weights = np.exp(-float(beta) * aer_clip)
    denom = weights.sum(axis=0, keepdims=True)
    w_norm = np.divide(
        weights,
        denom,
        out=np.full_like(weights, 1.0 / n_models),
        where=denom > 0,
    )
    p_ens = (w_norm[:, :, None] * proba_stack).sum(axis=0)
    r_star = np.sum(w_norm * aer_stack, axis=0)
    return p_ens, r_star


def _apply_strategy_acc_minus_aer(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    oof_acc: np.ndarray,
    slugs: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)

    acc = np.asarray(oof_acc, dtype=float)
    acc = np.where(np.isfinite(acc), acc, -np.inf)

    for i in range(n_samples):
        aer_i = aer_stack[:, i]
        score_i = acc - aer_i
        best = float(np.nanmax(score_i))
        cand = [
            m
            for m in range(n_models)
            if np.isfinite(score_i[m]) and abs(float(score_i[m]) - best) <= 1e-12
        ]
        if not cand:
            order = _sorted_model_indices(aer_i, oof_acc, slugs)
            m_idx = int(order[0])
        elif len(cand) == 1:
            m_idx = int(cand[0])
        else:
            m_idx = int(
                _sorted_model_indices(
                    aer_i[cand], oof_acc[cand], [slugs[m] for m in cand]
                )[0]
            )
            m_idx = int(cand[m_idx])

        p_ens[i] = proba_stack[m_idx, i]
        r_star[i] = float(aer_stack[m_idx, i])

    return p_ens, r_star


def _apply_strategy_dynamic_threshold(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    base_threshold: float,
    error_percentiles: tuple[float, float, float],
    easy_factor: float = 0.8,
    hard_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)

    tau0 = float(base_threshold)
    p25, _p50, p75 = map(float, error_percentiles)
    mean_err = np.mean(aer_stack, axis=0)
    thr = np.full(n_samples, tau0, dtype=float)
    thr[mean_err < p25] = tau0 * float(easy_factor)
    thr[mean_err > p75] = tau0 * float(hard_factor)

    for i in range(n_samples):
        aer_i = aer_stack[:, i]
        eligible = np.flatnonzero(aer_i <= thr[i])
        if eligible.size == 0:
            eligible = np.arange(n_models, dtype=int)
        p_ens[i] = np.mean(proba_stack[eligible, i], axis=0)
        r_star[i] = float(np.mean(aer_i[eligible]))

    return p_ens, r_star


def _apply_strategy_overconfidence_penalty(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    oof_acc: np.ndarray,
    slugs: list[str],
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)

    acc = np.asarray(oof_acc, dtype=float)
    acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    acc = np.clip(acc, 0.0, 1.0)

    for i in range(n_samples):
        aer_i = aer_stack[:, i]
        penalty = float(lam) * aer_i
        score_i = acc - penalty
        best = float(np.nanmax(score_i))
        cand = [
            m
            for m in range(n_models)
            if np.isfinite(score_i[m]) and abs(float(score_i[m]) - best) <= 1e-12
        ]
        if not cand:
            order = _sorted_model_indices(aer_i, oof_acc, slugs)
            m_idx = int(order[0])
        elif len(cand) == 1:
            m_idx = int(cand[0])
        else:
            m_idx = int(
                _sorted_model_indices(
                    aer_i[cand], oof_acc[cand], [slugs[m] for m in cand]
                )[0]
            )
            m_idx = int(cand[m_idx])

        p_ens[i] = proba_stack[m_idx, i]
        r_star[i] = float(aer_stack[m_idx, i])

    return p_ens, r_star


def _apply_strategy_calibration_aware(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    oof_acc: np.ndarray,
    slugs: list[str],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)

    acc = np.asarray(oof_acc, dtype=float)
    acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    acc = np.clip(acc, 0.0, 1.0)

    for i in range(n_samples):
        aer_i = aer_stack[:, i]
        score_i = float(alpha) * acc - (1.0 - float(alpha)) * aer_i
        best = float(np.nanmax(score_i))
        cand = [
            m
            for m in range(n_models)
            if np.isfinite(score_i[m]) and abs(float(score_i[m]) - best) <= 1e-12
        ]
        if not cand:
            order = _sorted_model_indices(aer_i, oof_acc, slugs)
            m_idx = int(order[0])
        elif len(cand) == 1:
            m_idx = int(cand[0])
        else:
            m_idx = int(
                _sorted_model_indices(
                    aer_i[cand], oof_acc[cand], [slugs[m] for m in cand]
                )[0]
            )
            m_idx = int(cand[m_idx])

        p_ens[i] = proba_stack[m_idx, i]
        r_star[i] = float(aer_stack[m_idx, i])

    return p_ens, r_star


def _apply_strategy_trimmed_weighted(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    beta: float,
    trim_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_models = proba_stack.shape[0]
    trim_q = float(trim_q)
    trim_q = float(np.clip(trim_q, 0.0, 1.0))

    p_ens = np.empty_like(proba_stack[0])
    r_star = np.empty(aer_stack.shape[1], dtype=float)
    for i in range(aer_stack.shape[1]):
        aer_i = aer_stack[:, i]
        if n_models > 1:
            cutoff = float(np.quantile(aer_i, trim_q))
            keep = np.where(aer_i <= cutoff)[0]
        else:
            keep = np.array([0], dtype=int)
        if keep.size == 0:
            keep = np.arange(n_models)
        aer_sel = np.clip(aer_i[keep], 1e-9, 1.0)
        w = np.exp(-float(beta) * aer_sel)
        denom = float(np.sum(w))
        if not np.isfinite(denom) or denom <= 0.0:
            w = np.full(len(keep), 1.0 / len(keep), dtype=float)
        else:
            w = w / denom
        p_ens[i] = np.sum(w[:, None] * proba_stack[keep, i], axis=0)
        r_star[i] = float(np.sum(w * aer_i[keep]))

    return p_ens, r_star


def _apply_strategy_borda_rank(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)

    aer_clip = np.clip(aer_stack, 1e-9, 1.0)
    borda_points = (n_classes - np.arange(n_classes)).astype(float)

    for i in range(n_samples):
        weights = np.exp(-float(beta) * aer_clip[:, i])
        denom = float(np.sum(weights))
        if not np.isfinite(denom) or denom <= 0.0:
            weights = np.full(n_models, 1.0 / max(1, n_models), dtype=float)
        else:
            weights = weights / denom

        scores = np.zeros(n_classes, dtype=float)
        for m in range(n_models):
            proba_m = proba_stack[m, i]
            order = np.argsort(-proba_m, kind="mergesort")
            scores[order] += weights[m] * borda_points

        score_sum = float(np.sum(scores))
        if not np.isfinite(score_sum) or score_sum <= 0.0:
            p_ens[i] = np.mean(proba_stack[:, i], axis=0)
        else:
            p_ens[i] = scores / score_sum
        r_star[i] = float(np.sum(weights * aer_stack[:, i]))

    return p_ens, r_star


def _apply_strategy_switching_hybrid(
    proba_stack: np.ndarray,
    aer_stack: np.ndarray,
    oof_acc: np.ndarray,
    slugs: list[str],
    top_n: int,
    beta: float,
    tau_low: float,
    tau_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_samples, n_classes = proba_stack.shape
    n_use = max(1, min(int(top_n), n_models))
    p_ens = np.empty((n_samples, n_classes), dtype=float)
    r_star = np.empty(n_samples, dtype=float)

    tl = float(tau_low)
    th = float(tau_high)
    if not np.isfinite(tl):
        tl = 0.0
    if not np.isfinite(th):
        th = 1.0
    if th < tl:
        tl, th = th, tl

    acc = np.asarray(oof_acc, dtype=float)
    acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    acc = np.clip(acc, 0.0, 1.0)

    for i in range(n_samples):
        aer_i = aer_stack[:, i]
        amin = float(np.min(aer_i)) if aer_i.size else 1.0

        if amin <= tl:
            order = _sorted_model_indices(aer_i, oof_acc, slugs)
            m_idx = int(order[0])
            p_ens[i] = proba_stack[m_idx, i]
            r_star[i] = float(aer_i[m_idx])
            continue

        if amin <= th:
            order = _sorted_model_indices(aer_i, oof_acc, slugs)
            sel = order[:n_use]
            aer_sel = np.clip(aer_i[sel], 1e-9, 1.0)
            w = np.exp(-float(beta) * aer_sel)
            denom = float(np.sum(w))
            if not np.isfinite(denom) or denom <= 0.0:
                w = np.full(len(sel), 1.0 / max(1, len(sel)), dtype=float)
            else:
                w = w / denom
            p_ens[i] = np.sum(w[:, None] * proba_stack[sel, i], axis=0)
            r_star[i] = float(np.sum(w * aer_i[sel]))
            continue

        aer_clip = np.clip(aer_i, 1e-9, 1.0)
        w = acc * np.exp(-float(beta) * aer_clip)
        denom = float(np.sum(w))
        if not np.isfinite(denom) or denom <= 0.0:
            w = np.full(n_models, 1.0 / max(1, n_models), dtype=float)
        else:
            w = w / denom
        p_ens[i] = np.sum(w[:, None] * proba_stack[:, i], axis=0)
        r_star[i] = float(np.sum(w * aer_i))

    return p_ens, r_star
