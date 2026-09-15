from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Optional,
)

import pandas as pd


@dataclass
class BaseModelAnalysisResult:
    compare_bins: dict[str, pd.DataFrame]
    compare_test_bins: dict[str, pd.DataFrame]
    compare_tests: dict[str, Optional[pd.DataFrame]]
    compare_metrics: list[dict[str, Any]]
    model_run_info: list[dict[str, Any]]
    model_rows: list[dict[str, Any]]
    ensemble_models: list[dict[str, Any]]
    best_slug: Optional[str]
    best_curve_test: Optional[pd.DataFrame]
    base_test: pd.DataFrame
