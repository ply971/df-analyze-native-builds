from __future__ import annotations

import re
from typing import (
    Any,
    Optional,
)
from warnings import warn

import numpy as np


def _model_slug(model_cls: type) -> str:
    name = getattr(model_cls, "__name__", "model")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
    return slug or "model"


def _slug_from_options(model_cls: type, options) -> str:
    source_attr = (
        "classifiers" if getattr(options, "is_classification", False) else "regressors"
    )
    sources = getattr(options, source_attr, None) or []
    for src in sources:
        getter = getattr(src, "get_model", None)
        if getter is None:
            continue
        if getter() is model_cls:
            return str(getattr(src, "value", _model_slug(model_cls))).lower()

    return _model_slug(model_cls)


def _select_top_results(
    results: list,
    metric_pref: Any,
    top_k: int,
    *,
    warn_on_nonfinite: bool = True,
    warn_context: Optional[str] = None,
) -> list:
    matching = [res for res in results if res.metric == metric_pref]
    candidates = matching or results
    if not candidates:
        return []
    finite = [res for res in candidates if np.isfinite(res.score)]
    non_finite = [res for res in candidates if not np.isfinite(res.score)]
    if not finite:
        if warn_on_nonfinite:
            prefix = f"{warn_context}: " if warn_context else ""
            warn(
                f"{prefix}No finite tuning scores found; using first available results."
            )
        ordered = candidates
    else:
        ordered = sorted(finite, key=lambda res: res.score, reverse=True) + non_finite
    k = max(1, int(top_k))
    return ordered[: min(k, len(ordered))]


def _select_unique_models(
    results: list,
    metric_pref: Any,
    options,
    top_k: int,
) -> list:
    if not results:
        return []
    requested_model_clses_raw = getattr(options, "models", None)
    if requested_model_clses_raw is None:
        requested_model_clses = sorted(
            {
                getattr(r, "model_cls", None)
                for r in results
                if getattr(r, "model_cls", None) is not None
            },
            key=_model_slug,
        )
    else:
        try:
            requested_model_clses = list(requested_model_clses_raw)
        except TypeError:
            requested_model_clses = []

    warn_on_nonfinite = True
    verbosity = getattr(options, "verbosity", None)
    if verbosity is not None:
        verbosity_value = getattr(verbosity, "value", verbosity)
        try:
            warn_on_nonfinite = int(verbosity_value) > 1
        except (TypeError, ValueError):
            warn_on_nonfinite = True

    per_model: list = []
    for mcls in requested_model_clses:
        group = [r for r in results if getattr(r, "model_cls", None) is mcls]
        if not group:
            continue
        warn_context = None
        if warn_on_nonfinite:
            warn_context = f"Model '{_slug_from_options(mcls, options)}'"
        best = _select_top_results(
            group,
            metric_pref,
            top_k=1,
            warn_on_nonfinite=warn_on_nonfinite,
            warn_context=warn_context,
        )
        if best:
            per_model.append(best[0])

    if not per_model:
        return []

    finite = [r for r in per_model if np.isfinite(getattr(r, "score", np.nan))]
    non_finite = [r for r in per_model if not np.isfinite(getattr(r, "score", np.nan))]
    ordered = sorted(finite, key=lambda r: r.score, reverse=True) + non_finite

    k = int(top_k) if int(top_k) > 0 else len(ordered)
    return ordered[: min(k, len(ordered))]
