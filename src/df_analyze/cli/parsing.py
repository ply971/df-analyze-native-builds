from __future__ import annotations

import re
from argparse import Action
from enum import Enum
from math import isnan
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Type, TypeVar, Union
from warnings import warn

E = TypeVar("E")

if TYPE_CHECKING:
    from src.df_analyze.enumerables import SeedKind


def resolved_path(p: Union[str, Path]) -> Path:
    try:
        path = Path(p)
    except Exception as e:
        raise ValueError(f"Could not interpret string {p} as path") from e
    try:
        path = path.resolve()
    except Exception as e:
        raise ValueError(f"Could not resolve path {path} to valid path.") from e
    return path


def resolved_path_list(ps: str) -> List[Path]:
    paths = []
    for p in ps.split(","):
        try:
            path = Path(p)
        except Exception as e:
            raise ValueError(
                f"Could not interpret string {p} as path. User-input argument: {ps}.\n\n"
                f'Note: path arguments are split on commas (","), so make sure none of'
                "your filepaths contain commas."
            ) from e
        try:
            path = path.resolve()
        except Exception as e:
            raise ValueError(f"Could not resolve path {path} to valid path.") from e
        paths.append(path)
    return paths


def cv_size(cv_str: str) -> Union[float, int]:
    try:
        cv = float(cv_str)
    except Exception as e:
        raise ValueError(
            "Could not convert a `... -size` argument (e.g. --htune-val-size) value to float"
        ) from e
    # validate
    if isnan(cv):
        raise ValueError("NaN is not a valid size")
    if cv <= 0:
        raise ValueError("`... -size` arguments (e.g. --htune-val-size) must be positive")
    if cv == 1:
        raise ValueError(
            "'1' is not a valid value for `... -size` arguments (e.g. --htune-val-size)."
        )
    if (cv > 1) and not cv.is_integer():
        raise ValueError(
            "Passing a float greater than 1.0 for `... -size` arguments "
            "(e.g. --htune-val-size) is not valid. See documentation for "
            "`--htune-val-size` or `--test-val-sizes`."
        )
    if cv > 1:
        return int(cv)
    return cv


def int_or_percent_parser(
    default: Union[int, float], _warn: bool = True
) -> Callable[[str], Union[int, float]]:
    def parser(arg: str) -> Union[int, float]:
        message = f"Setting argument to default value: {default}"
        try:
            parsed = float(arg)
        except Exception as e:
            if _warn:
                warn(f"Could not convert argument `{arg}` to float: {e}. {message}")
            parsed = float(default)
        # validate
        if isnan(parsed):
            if _warn:
                warn(f"NaN is not a valid size. {message}")
            parsed = float(default)
        if parsed < 0:
            if _warn:
                warn(
                    "Got a float or integer less than 0 for a percent argument. "
                    f"Percentages must be in [0.0, 1.0], and integers must be "
                    f"positive or zero. {message}"
                )
            parsed = float(default)
        if (parsed > 1) and not parsed.is_integer():
            if _warn:
                warn(
                    "Got a float greater than 1.0 for a percent argument. "
                    f"Percentages must be in [0.0, 1.0]. {message}"
                )
            parsed = float(default)
        if parsed > 1:
            return int(parsed)
        return parsed

    return parser


def int_or_percent_or_none_parser(
    default: Union[int, float, None],
) -> Callable[[str], Union[int, float, None]]:
    def inner(arg: str) -> Optional[Union[float, int]]:
        if "none" in arg.lower():
            return None
        if default is None:
            parser = int_or_percent_parser(default=-1, _warn=False)
        else:
            parser = int_or_percent_parser(default=default, _warn=False)
        parsed = parser(arg)
        if parsed < 0:
            return None
        return parsed

    return inner


def enum_or_none_parser(enum: E) -> Callable[[str], Optional[E]]:
    def inner(arg: str) -> Optional[E]:
        if "none" in arg.lower():
            return None
        return enum(arg)  # type: ignore

    return inner


def seed_parser(enum: Type[SeedKind]) -> Callable[[str], Union[int, SeedKind]]:
    def inner(arg: str) -> Union[int, SeedKind]:
        clean = str(arg).lower().strip()
        if clean in enum.choices():
            return enum(clean)
        try:
            seed = int(clean)
            return seed
        except Exception as e:
            raise ValueError(f"Could not parse `--seed` argument: {arg}") from e

    return inner


def separator(s: str) -> str:
    if s.lower().strip() == "tab":
        return "\t"
    if s.lower().strip() == "newline":
        return "\n"
    return s


def column_parser(s: str) -> list[str]:
    s = re.sub(r" +", " ", s)
    s = re.sub(r",+", ",", s)
    cols = [w for w in s.split(",") if len(w) > 0]
    return cols


class EnumParser(Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super().__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)
