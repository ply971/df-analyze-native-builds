#  a unified implementation for serialization of preds/probs, supporting df and dict for the multi-target
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

PredArray = Union[Series, DataFrame, ndarray]
ProbArray = Optional[
    Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]
]


def _array_equal(left: ndarray, right: ndarray) -> bool:
    if left.shape != right.shape:
        return False
    try:
        return bool(np.allclose(left, right, atol=1e-8, equal_nan=True))
    except TypeError:
        return bool(np.all(left == right))


def _preds_equal(left: PredArray, right: PredArray) -> bool:
    if isinstance(left, Series) and isinstance(right, Series):
        return left.index.equals(right.index) and _array_equal(
            left.to_numpy(), right.to_numpy()
        )
    if isinstance(left, DataFrame) and isinstance(right, DataFrame):
        return (
            left.index.equals(right.index)
            and [str(col) for col in left.columns] == [str(col) for col in right.columns]
            and _array_equal(left.to_numpy(), right.to_numpy())
        )
    if isinstance(left, ndarray) and isinstance(right, ndarray):
        return _array_equal(left, right)
    return False


def _probs_equal(left: ProbArray, right: ProbArray) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, ndarray) and isinstance(right, ndarray):
        return _array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        left_map = {str(k): np.asarray(v) for k, v in left.items()}
        right_map = {str(k): np.asarray(v) for k, v in right.items()}
        if set(left_map) != set(right_map):
            return False
        return all(_array_equal(left_map[key], right_map[key]) for key in left_map)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(
            _array_equal(np.asarray(lval), np.asarray(rval))
            for lval, rval in zip(left, right)
        )
    return False


def _serialize_preds(preds: PredArray) -> tuple[str, Any, str, Optional[list[str]]]:
    if isinstance(preds, Series):
        return "series", preds.to_list(), str(preds.dtype), None
    if isinstance(preds, DataFrame):
        return (
            "dataframe",
            preds.to_numpy().tolist(),
            str(preds.to_numpy().dtype),
            [str(col) for col in preds.columns],
        )

    arr = np.asarray(preds)
    return "ndarray", arr.tolist(), str(arr.dtype), None


def _serialize_probs(probs: ProbArray) -> tuple[str, Any, Any]:
    if probs is None:
        return "none", None, None
    if isinstance(probs, ndarray):
        return "ndarray", probs.tolist(), str(probs.dtype)
    if isinstance(probs, dict):
        payload = {str(k): np.asarray(v).tolist() for k, v in probs.items()}
        dtypes = {str(k): str(np.asarray(v).dtype) for k, v in probs.items()}
        return "dict", payload, dtypes
    if isinstance(probs, (list, tuple)):
        payload = [np.asarray(v).tolist() for v in probs]
        dtypes = [str(np.asarray(v).dtype) for v in probs]
        return "sequence", payload, dtypes

    arr = np.asarray(probs)
    return "ndarray", arr.tolist(), str(arr.dtype)


def _deserialize_preds(result: dict[str, Any], key: str) -> PredArray:
    kind = str(result.get("preds_kind", "series"))
    dtype = result.get("preds_dtype")
    values = result[key]

    if kind == "dataframe":
        cols = result.get("preds_columns")
        if not isinstance(cols, (list, tuple)):
            arr = np.asarray(values)
            n_cols = arr.shape[1] if arr.ndim > 1 else 1
            cols = [f"target_{i}" for i in range(n_cols)]
        return DataFrame(values, columns=[str(col) for col in cols])
    if kind == "ndarray":
        return np.asarray(values, dtype=dtype)
    if dtype is None:
        return Series(values)
    return Series(values, dtype=dtype)


def _deserialize_probs(result: dict[str, Any], key: str) -> ProbArray:
    kind = str(result.get("probs_kind", "ndarray"))
    dtype = result.get("probs_dtype")
    values = result.get(key)

    if kind == "none" or values is None:
        return None
    if kind == "dict":
        if not isinstance(values, dict):
            return None
        dtypes = dtype if isinstance(dtype, dict) else {}
        return {
            str(k): np.asarray(v, dtype=dtypes.get(str(k)))
            for k, v in values.items()
        }
    if kind == "sequence":
        if not isinstance(values, (list, tuple)):
            return None
        dtypes = dtype if isinstance(dtype, list) else []
        return [
            np.asarray(v, dtype=dtypes[i] if i < len(dtypes) else None)
            for i, v in enumerate(values)
        ]
    return np.asarray(values, dtype=dtype)
