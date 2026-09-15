"""adaptive Error Rate (aER) mapping from confidence to expected error"""
# Thesis proposal (sample-based error-rate via predictive confidence): https://www.stfx.ca/media/69566/download?attachment=

from __future__ import annotations
from typing import Any, Optional

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.stats import _wilson_interval


class AdaptiveErrorCalculator:
    def __init__(
        self,
        n_bins: int = 20,
        min_bin_count: int = 10,
        smooth: bool = True,
        enforce_monotonic: bool = False,
        prior_strength: float = 2.0,
        adaptive_binning: bool = False,
    ) -> None:
        self.n_bins = int(n_bins)
        self.min_bin_count = max(1, int(min_bin_count))
        self.smooth = bool(smooth)
        self.enforce_monotonic = bool(enforce_monotonic)
        self.prior_strength = float(prior_strength)
        self.adaptive_binning = bool(adaptive_binning)

        self.bin_edges: Optional[np.ndarray] = None
        self.bin_centers: Optional[np.ndarray] = None
        self.bin_counts: Optional[np.ndarray] = None
        self.bin_error_counts: Optional[np.ndarray] = None
        self.raw_error: Optional[np.ndarray] = None
        self.expected_error: Optional[np.ndarray] = None
        self.global_error: Optional[float] = None

    @staticmethod
    def _apply_isotonic_monotonic(
        *,
        expected: np.ndarray,
        edges: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            return expected

        centers = (edges[:-1] + edges[1:]) / 2.0
        iso = IsotonicRegression(
            increasing=False,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        )
        transformed = iso.fit_transform(centers, expected, sample_weight=counts)
        return np.asarray(transformed, dtype=float)

    def _effective_bins(self, conf: np.ndarray) -> int:
        n = int(conf.size)
        if n <= 0:
            return 1

        max_bins_by_count = max(1, n // self.min_bin_count)
        effective_bins = max(1, min(self.n_bins, max_bins_by_count))

        if self.adaptive_binning:
            return effective_bins

        conf_std = float(np.std(conf))
        if conf_std < 0.05 or n < 100:
            effective_bins = min(5, effective_bins)
        elif n < 500:
            effective_bins = min(10, effective_bins)
        elif n < 1000:
            effective_bins = min(15, effective_bins)
        return max(1, effective_bins)

    def _make_bin_edges(self, conf: np.ndarray) -> np.ndarray:
        conf = np.clip(np.asarray(conf, dtype=float).ravel(), 0.0, 1.0)
        if conf.size == 0:
            return np.array([0.0, 1.0], dtype=float)

        conf_min = float(np.min(conf))
        conf_max = float(np.max(conf))
        if conf_max - conf_min < 1e-6:
            return np.array([0.0, 1.0], dtype=float)

        effective_bins = self._effective_bins(conf)

        if not self.adaptive_binning:
            return np.linspace(0.0, 1.0, effective_bins + 1)

        qs = np.linspace(0.0, 1.0, effective_bins + 1)
        edges = np.quantile(conf, qs)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)

        if edges.size < 2:
            return np.array([0.0, 1.0], dtype=float)
        return edges.astype(float)

    def _merge_bins_min_count(
        self,
        edges: np.ndarray,
        counts: np.ndarray,
        err_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = np.asarray(edges, dtype=float).ravel()
        counts = np.asarray(counts, dtype=int).ravel()
        err_counts = np.asarray(err_counts, dtype=int).ravel()

        if edges.size < 2 or counts.size == 0:
            return edges, counts, err_counts
        if counts.size != edges.size - 1 or err_counts.size != counts.size:
            return edges, counts, err_counts

        min_count = max(1, int(self.min_bin_count))
        if np.all(counts >= min_count):
            return edges, counts, err_counts

        merged_edges: list[float] = [float(edges[0])]
        merged_counts: list[int] = []
        merged_errs: list[int] = []
        cur_count = 0
        cur_err = 0

        for i in range(counts.size):
            cur_count += int(counts[i])
            cur_err += int(err_counts[i])
            if cur_count >= min_count:
                merged_edges.append(float(edges[i + 1]))
                merged_counts.append(cur_count)
                merged_errs.append(cur_err)
                cur_count = 0
                cur_err = 0

        if cur_count > 0:
            if merged_counts:
                merged_counts[-1] += cur_count
                merged_errs[-1] += cur_err
                merged_edges[-1] = float(edges[-1])
            else:
                merged_edges.append(float(edges[-1]))
                merged_counts.append(cur_count)
                merged_errs.append(cur_err)
        elif merged_edges[-1] != float(edges[-1]) and merged_counts:
            merged_edges[-1] = float(edges[-1])

        return (
            np.asarray(merged_edges, dtype=float),
            np.asarray(merged_counts, dtype=int),
            np.asarray(merged_errs, dtype=int),
        )

    def fit_from_oof(
        self,
        conf_oof: Any,
        y_pred_oof: Any,
        y_true: Any,
    ) -> None:
        conf = np.asarray(conf_oof, dtype=float).ravel()
        y_pred = np.asarray(y_pred_oof).ravel()
        y_t = np.asarray(y_true).ravel()
        if conf.shape[0] != y_pred.shape[0] or conf.shape[0] != y_t.shape[0]:
            raise ValueError("OOF arrays must have the same length")
        incorrect = (y_pred != y_t).astype(int)
        self.fit(confidences=conf, incorrect=incorrect)

    def fit(self, confidences: Any, incorrect: Any) -> None:
        conf = np.clip(np.asarray(confidences, dtype=float).ravel(), 0.0, 1.0)
        inc = np.asarray(incorrect).ravel().astype(int)

        if conf.shape[0] != inc.shape[0]:
            raise ValueError(
                "confidences and incorrect must contain the same number of items"
            )
        n = conf.size
        if n == 0:
            raise ValueError("Cannot fit aER with 0 samples")

        edges = self._make_bin_edges(conf)
        n_bins_eff = edges.size - 1
        if n_bins_eff < 1:
            edges = np.array([0.0, 1.0], dtype=float)
            n_bins_eff = 1
        bin_idx = np.searchsorted(edges, conf, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_bins_eff - 1)
        counts = np.bincount(bin_idx, minlength=n_bins_eff).astype(int)
        err_counts = np.bincount(bin_idx, weights=inc, minlength=n_bins_eff).astype(int)
        edges, counts, err_counts = self._merge_bins_min_count(edges, counts, err_counts)
        n_bins_eff = max(1, int(edges.size - 1))
        raw = np.full(n_bins_eff, np.nan, dtype=float)
        nonzero = counts > 0
        raw[nonzero] = err_counts[nonzero] / counts[nonzero]

        global_err = float(np.mean(inc))
        if self.prior_strength > 0:
            prior_eps = 1e-6
            prior_mean = float(np.clip(global_err, prior_eps, 1.0 - prior_eps))
            alpha_prior = self.prior_strength * prior_mean
            beta_prior = self.prior_strength * (1.0 - prior_mean)
        else:
            alpha_prior = 0.0
            beta_prior = 0.0

        expected = np.empty(n_bins_eff, dtype=float)
        for i in range(n_bins_eff):
            if counts[i] > 0:
                if self.prior_strength > 0:
                    expected[i] = (err_counts[i] + alpha_prior) / (
                        counts[i] + alpha_prior + beta_prior
                    )
                else:
                    expected[i] = raw[i]
            else:
                expected[i] = global_err
        if self.smooth and expected.size >= 3:
            expected = self._smooth_error_rates(
                expected=expected,
                counts=counts,
            )

        # the optional choice from the sklearn.isotonic
        if self.enforce_monotonic and expected.size >= 2:
            expected = self._apply_isotonic_monotonic(
                expected=expected,
                edges=edges,
                counts=counts,
            )

        expected = np.clip(expected.astype(float), 0.0, 1.0)

        self.bin_edges = edges
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0
        self.bin_counts = counts
        self.bin_error_counts = err_counts
        self.raw_error = raw
        self.expected_error = expected
        self.global_error = global_err

    def _smooth_error_rates(
        self,
        expected: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        out = expected.astype(float).copy()
        for i in range(expected.size):
            s = max(0, i - 1)
            t = min(expected.size, i + 2)
            w = counts[s:t].astype(float)
            if w.sum() > 0:
                out[i] = float(np.average(expected[s:t], weights=w))
            else:
                out[i] = float(np.mean(expected[s:t]))
        return out

    def get_expected_error(self, confidences: Any) -> np.ndarray:
        if self.bin_edges is None or self.expected_error is None:
            raise RuntimeError("aER not fit")

        conf = np.asarray(confidences, dtype=float).ravel()
        out = np.empty_like(conf, dtype=float)
        fill = float(self.global_error) if self.global_error is not None else 0.5
        out[~np.isfinite(conf)] = fill

        finite = np.isfinite(conf)
        if not np.any(finite):
            return np.clip(out, 0.0, 1.0)

        c = np.clip(conf[finite], 0.0, 1.0)

        edges = np.asarray(self.bin_edges, dtype=float)
        exp = np.asarray(self.expected_error, dtype=float)
        n_bins = max(1, exp.size)
        idx = np.searchsorted(edges, c, side="right") - 1
        idx = np.clip(idx, 0, n_bins - 1)
        out[finite] = exp[idx]
        return np.clip(out, 0.0, 1.0)

    def bin_stats_df(self, z: float = 1.96, *, drop_empty: bool = True) -> pd.DataFrame:
        if (
            self.bin_edges is None
            or self.bin_centers is None
            or self.bin_counts is None
            or self.bin_error_counts is None
            or self.expected_error is None
        ):
            raise RuntimeError("aER not fit")

        lows, highs = [], []
        for k, n in zip(self.bin_error_counts.tolist(), self.bin_counts.tolist()):
            lo, hi = _wilson_interval(int(k), int(n), z=z)
            lows.append(lo)
            highs.append(hi)

        df = pd.DataFrame(
            {
                "bin_left": self.bin_edges[:-1],
                "bin_right": self.bin_edges[1:],
                "bin_center": self.bin_centers,
                "count": self.bin_counts,
                "error_count": self.bin_error_counts,
                "raw_error": self.raw_error,
                "expected_error": self.expected_error,
                "wilson_low": np.asarray(lows, dtype=float),
                "wilson_high": np.asarray(highs, dtype=float),
            }
        )
        if drop_empty:
            df = df[df["count"].fillna(0).astype(int) > 0].reset_index(drop=True)
        return df

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "n_bins": int(self.n_bins),
            "min_bin_count": int(self.min_bin_count),
            "smooth": bool(self.smooth),
            "enforce_monotonic": bool(self.enforce_monotonic),
            "prior_strength": float(self.prior_strength),
            "adaptive_binning": bool(self.adaptive_binning),
            "bin_edges": None if self.bin_edges is None else self.bin_edges.tolist(),
            "bin_centers": None
            if self.bin_centers is None
            else self.bin_centers.tolist(),
            "bin_counts": None if self.bin_counts is None else self.bin_counts.tolist(),
            "bin_error_counts": None
            if self.bin_error_counts is None
            else self.bin_error_counts.tolist(),
            "raw_error": None if self.raw_error is None else self.raw_error.tolist(),
            "expected_error": None
            if self.expected_error is None
            else self.expected_error.tolist(),
            "global_error": None
            if self.global_error is None
            else float(self.global_error),
        }

    def to_csv_dataframe(self, *, drop_empty: bool = True) -> pd.DataFrame:
        if (
            self.bin_edges is None
            or self.expected_error is None
            or self.bin_counts is None
        ):
            raise RuntimeError("aER not fit")
        df = pd.DataFrame(
            {
                "bin_left": self.bin_edges[:-1],
                "bin_right": self.bin_edges[1:],
                "bin_center": None if self.bin_centers is None else self.bin_centers,
                "count": self.bin_counts,
                "expected_error": self.expected_error,
            }
        )
        if drop_empty:
            df = df[df["count"].fillna(0).astype(int) > 0].reset_index(drop=True)
        return df
