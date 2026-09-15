from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional, cast
from warnings import warn

import jsonpickle
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Index, Series

from df_analyze._constants import N_CAT_LEVEL_MIN
from df_analyze.preprocessing.inspection.inference import Inference, InferredKind
from df_analyze.preprocessing.inspection.text import (
    BIG_INFO,
    BINARY_INFO,
    CAT_INFO,
    CONST_INFO,
    FLOAT_INFO,
    ID_INFO,
    INFLATION_HEADER,
    NYAN_INFO,
    ORD_INFO,
    TIME_INFO,
)


class ColumnType(Enum):
    Id = "id"
    Time = "time"
    Const = "const"
    Continuous = "cont"
    Ordinal = "ord"
    Categorical = "cat"
    BigCat = "big_cat"
    Binary = "bin"
    Nyan = "const_nan"
    UserCategorical = "user_cat"
    UserOrdinal = "user_ord"
    Other = "other"

    def fmt(self, info: str) -> str:
        fmt = {
            ColumnType.Id: ID_INFO,
            ColumnType.Time: TIME_INFO,
            ColumnType.Const: CONST_INFO,
            ColumnType.Continuous: FLOAT_INFO,
            ColumnType.Ordinal: ORD_INFO,
            ColumnType.Categorical: CAT_INFO,
            ColumnType.BigCat: BIG_INFO,
            ColumnType.Binary: BINARY_INFO,
            ColumnType.UserCategorical: "User-specified categorical",
            ColumnType.UserOrdinal: "User-specified ordinal",
            ColumnType.Nyan: NYAN_INFO,
            ColumnType.Other: None,
        }[self]
        if fmt is None:
            raise ValueError(f"Formatting not implemeted for enum {self}")
        if self in [ColumnType.UserCategorical, ColumnType.UserOrdinal]:
            return fmt

        return fmt.format(info=info)


@dataclass
class InflationInfo:
    col: str
    to_deflate: list[str]
    to_keep: list[str]
    n_deflate: int
    n_keep: int
    n_total: int

    @staticmethod
    def from_series(series: Series) -> tuple[InflationInfo, ndarray, ndarray]:
        unqs, cnts = np.unique(series.copy(deep=True).astype(str), return_counts=True)
        n_total = len(unqs)
        keep_idx = cnts >= N_CAT_LEVEL_MIN
        n_keep = keep_idx.sum()
        n_deflate = len(keep_idx) - n_keep
        return (
            InflationInfo(
                col=str(series.name),
                to_deflate=unqs[~keep_idx].tolist(),
                to_keep=unqs[keep_idx].tolist(),
                n_deflate=n_deflate,
                n_keep=n_keep,
                n_total=n_total,
            ),
            unqs,
            cnts,
        )


@dataclass
class ColumnDescriptions:
    col: str
    const: Optional[Inference] = None
    time: Optional[Inference] = None
    bin: Optional[Inference] = None
    ord: Optional[Inference] = None
    id: Optional[Inference] = None
    cont: Optional[Inference] = None
    cat: Optional[Inference] = None


def get_width(*cols: dict[str, str]) -> int:
    all_cols = {}
    for d in cols:
        all_cols.update(d)
    if len(all_cols) > 0:
        return max(len(col) for col in all_cols) + 2
    return 0


class InspectionInfo:
    """Container holding ColumnType and Inferences for that column"""

    def __init__(
        self,
        kind: ColumnType,
        infos: dict[str, Inference] | None = None,
    ) -> None:
        self.kind = kind
        self.infos = infos or {}
        self.cols = set([*self.infos.keys()])
        self.pad = get_width({col: info.reason for col, info in self.infos.items()})
        self.is_empty = len(self.infos) == 0

    def certains(self) -> dict[str, Inference]:
        return {col: infer for col, infer in self.infos.items() if infer.is_certain()}

    def uncertains(self) -> dict[str, Inference]:
        return {col: infer for col, infer in self.infos.items() if not infer.is_certain()}

    def certain_lines(self, pad: Optional[int] = None) -> list[str]:
        return self.lines_from_infos(self.certains(), pad)

    def uncertain_lines(self, pad: Optional[int] = None) -> list[str]:
        return self.lines_from_infos(self.uncertains(), pad)

    @staticmethod
    def lines_from_infos(
        infos: dict[str, Inference], pad: Optional[int] = None
    ) -> list[str]:
        pad = pad or get_width({col: info.reason for col, info in infos.items()})
        return [f"{col:<{pad}} {info.reason}" for col, info in infos.items()]

    def lines(self, pad: Optional[int] = None) -> list[str]:
        pad = pad or self.pad
        return [f"{col:<{pad}} {info.reason}" for col, info in self.infos.items()]

    def print_message(self, pad: Optional[int] = None) -> None:
        if self.is_empty:
            return

        cols = get_terminal_size((81, 24))[0]
        sep = "=" * cols
        underline = "." * (len(self.__class__.__name__) + 1)
        info = "\n".join(self.lines(pad))
        message = self.kind.fmt(info)
        formatted = f"\n{sep}\n{self.__class__.__name__}\n{underline}\n{message}\n{sep}"
        print(formatted, file=sys.stderr)

    def to_df(self) -> DataFrame:
        df = DataFrame(
            [[col, infer.kind.value, infer.reason] for col, infer in self.infos.items()],
            columns=["feature_name", "inferred", "reason"],
        )
        return df

    def __str__(self) -> str:
        cname = f"{self.kind.name}{self.__class__.__name__}"
        cols = [*self.infos.keys()]
        if len(cols) > 3:
            fmt = str(cols[:3]).replace("]", ", ... ")
        else:
            fmt = str(cols[:3])
        fmt = fmt.replace("[", "").replace("]", "")

        return f"{cname}({fmt})"

    __repr__ = __str__

    def __eq__(self, other: object | InspectionInfo) -> bool:
        if not isinstance(other, InspectionInfo):
            return False
        return (
            self.kind == other.kind
            and self.infos == other.infos
            and sorted(self.cols) == sorted(other.cols)
        )

    @staticmethod
    def merge(*infos: InspectionInfo) -> InspectionInfo:
        if len(infos) == 0:
            raise ValueError("Nothing to merge")
        kind = infos[0].kind
        if not all(info.kind == kind for info in infos):
            raise ValueError("Cannot merge when kinds differ.")
        new_infos = {}
        for info in infos:
            new_infos.update(info.infos)
        return InspectionInfo(kind=kind, infos=new_infos)

    @staticmethod
    def conflicts(
        *infos: InspectionInfo,
    ) -> tuple[dict[str, list[Inference]], dict[str, list[Inference]]]:
        cols = set()
        for info in infos:
            cols.update(info.cols)

        certains: dict[str, list[Inference]] = {col: [] for col in cols}
        maybes: dict[str, list[Inference]] = {col: [] for col in cols}
        for col in cols:
            for info in infos:
                if col not in info.cols:
                    continue
                infer = info.infos[col]
                if infer.is_certain():
                    certains[col].append(infer)
                else:
                    maybes[col].append(infer)
        for col in certains:
            if len(certains[col]) > 1:
                raise RuntimeError(
                    f"Conflicting certainties for column {col}: {certains[col]}"
                )

        return certains, maybes


@dataclass
class InspectionResults:
    conts: InspectionInfo
    ords: InspectionInfo
    ids: InspectionInfo
    times: InspectionInfo
    consts: InspectionInfo
    cats: InspectionInfo
    binaries: InspectionInfo
    big_cats: dict[str, int]
    multi_cats: list[str]
    inflation: list[InflationInfo]
    user_cats: set[str]
    user_ords: set[str]
    drops: set[str]

    def ordered_basic_infos(
        self,
    ) -> tuple[
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
    ]:
        return (
            self.ids,
            self.times,
            self.consts,
            self.binaries,
            self.conts,
            self.ords,
            self.cats,
        )

    def all_inferences(self) -> dict[str, Inference]:
        return {
            **self.cats.infos,
            **self.ords.infos,
            **self.conts.infos,
            **self.binaries.infos,
            **self.ids.infos,
            **self.times.infos,
            **self.consts.infos,
        }

    def inflation_lines(self, pad: Optional[int] = None) -> list[str]:
        infos = self.inflation
        infos = sorted(infos, key=lambda info: info.n_total, reverse=True)
        if len(infos) <= 0:
            return []
        w = pad or max(len(info.col) for info in infos) + 2
        header = "Deflated categorical variables (before --> after):"
        lines = [
            f"{info.col:<{w}} {info.n_total: >3} --> {info.n_keep:< 2}" for info in infos
        ]
        return [header, *lines, "\n"]

    def inflation_level_lines(self, pad: Optional[int] = None) -> list[str]:
        try:
            infos = self.inflation
            infos = sorted(infos, key=lambda info: info.n_total, reverse=True)
            if len(infos) <= 0:
                return []
            w = pad or max(len(info.col) for info in infos) + 2
            header = "Deflated levels"
            header = f"{header}\n{'-' * len(header)}\n"
            header_info = (
                "I.e. below lists each categorical feature having levels with "
                f"less than {N_CAT_LEVEL_MIN} total samples. The listed level "
                "values underneath each feature name are replaced with the value "
                "'DEFLATED' (or 'DEFLATED_OTHER', if somehow 'DEFLATED' is already "
                "a level value). Remaining level values are present in feature "
                "selection and other feature-related reports (e.g. "
                "`association_selection_report.md`)."
            )
            header = f"{header}{header_info}\n"
            lines = []
            for info in infos:
                lines.append(f"{info.col}\n{'.' * w}\n{info.to_deflate}\n")

            return [header, *lines, "\n"]
        except Exception as e:
            print(f"Got error generating level deflations info: {e}")
            return []

    def drop_cols(self) -> list[str]:
        cols = []
        for col, infer in self.all_inferences().items():
            if infer.kind.should_drop():
                cols.append(col)
        return cols

    def coercions(self) -> dict[str, Inference]:
        coerced = {}
        for col, infer in self.all_inferences().items():
            if infer.kind.is_coerced():
                coerced[col] = infer
        return coerced

    def final_binaries(self) -> dict[str, Inference]:
        coerced = {}
        for col, infer in self.all_inferences().items():
            if infer.kind.is_bin():
                coerced[col] = infer
        return coerced

    def final_categoricals(self) -> dict[str, Inference]:
        coerced = {}
        for col, infer in self.all_inferences().items():
            if infer.kind in [
                InferredKind.CertainCat,
                InferredKind.MaybeCat,
                InferredKind.UserCategorical,
                InferredKind.CoercedCat,
            ]:
                coerced[col] = infer
        return coerced

    def big_header(self, title: str, pad: int) -> str:
        big_sep = "=" * pad
        return f"{big_sep}\n{title.center(pad)}\n{big_sep}\n"

    def med_header(self, title: str) -> str:
        underline = "-" * len(title)
        return f"{title}\n{underline}\n"

    def small_header(self, title: str) -> str:
        underline = "┄" * (len(title))
        return f"{title.capitalize()}\n{underline}\n"

    def subsection(self, title: str, lines: list[str]) -> str:
        if len(lines) <= 0:
            return ""
        header = self.small_header(title)
        joined = "\n".join(lines)
        return f"\n{header}{joined}\n\n"

    def section(self, title: str, lines: list[str]) -> str:
        if len(lines) <= 0:
            return ""
        header = self.med_header(title)
        joined = "\n".join(lines)
        return f"\n{header}{joined}\n\n"

    def full_report(self, pad: Optional[int] = None) -> str: ...

    def short_report(self, pad: Optional[int] = None) -> str:
        """
        Ordered from most concerning to least, or from most amount of data
        manipulation (removal of features or feature levels, or coercion)
        """
        pad = pad or get_terminal_size((81, 24))[0]

        # fmt: off
        coercions    = self.coercions()
        categoricals = self.final_categoricals()
        binaries     = self.final_binaries()

        maybe_id_lines     = self.ids.uncertain_lines(None)
        certain_id_lines   = self.ids.certain_lines(None)
        maybe_time_lines   = self.times.uncertain_lines(None)
        certain_time_lines = self.times.certain_lines(None)
        consts_lines       = self.consts.certain_lines(None)

        coercion_lines = InspectionInfo.lines_from_infos(coercions, None)
        cat_lines      = InspectionInfo.lines_from_infos(categoricals, None)
        bin_lines      = InspectionInfo.lines_from_infos(binaries, None)
        ord_lines      = InspectionInfo.lines_from_infos(self.ords.infos, None)
        cont_lines     = InspectionInfo.lines_from_infos(self.conts.infos, None)
        # fmt: on

        id_lines = [*maybe_id_lines, *certain_id_lines]
        time_lines = [*maybe_time_lines, *certain_time_lines]
        destructives = [*id_lines, *time_lines, *consts_lines, *coercion_lines]
        removes = [*id_lines, *time_lines, *consts_lines]
        numerics = [*ord_lines, *cont_lines]

        destruct_header = (
            self.big_header("Destructive Data Changes", pad)
            if len(destructives) > 0
            else ""
        )
        remove_header = self.med_header("Removed Features") if len(removes) > 0 else ""
        id_section = self.subsection("Ids", id_lines)
        time_section = self.subsection("Times", time_lines)
        const_section = self.subsection("Constants", consts_lines)
        coerce_section = self.section("Cardinality Coercions", coercion_lines)
        dest_space = "\n\n" if len(destructives) > 0 else ""

        inference_header = self.big_header("Feature Cardinality Inferences", pad)

        bin_section = self.section("Binary Features", bin_lines)
        cat_section = self.section("Categorical Features", cat_lines)
        numeric_header = self.med_header("Numeric Features") if len(numerics) > 0 else ""
        ord_section = self.subsection("Ordinals", ord_lines)
        cont_section = self.subsection("Continuous", cont_lines)

        # shape_info = self.big_header("Shape info", pad)

        inflate_lines = self.inflation_lines()
        inflate_info = "\n".join(inflate_lines)
        inflate_header = (
            self.med_header("Deflated Categoricals") if len(inflate_lines) > 0 else ""
        )
        inflate_desc = f"{INFLATION_HEADER}\n\n" if len(inflate_lines) > 0 else ""

        inflate_level_lines = self.inflation_level_lines()
        inflate_level_info = "\n".join(inflate_level_lines)

        report = (
            f"{destruct_header}"
            f"{remove_header}"
            f"{id_section}{time_section}{const_section}{coerce_section}"
            f"{dest_space}"
            f"{inference_header}"
            f"{bin_section}{cat_section}"
            f"{inflate_header}"
            f"{inflate_desc}"
            f"{inflate_info}"
            f"{inflate_level_info}"
            f"{numeric_header}"
            f"{ord_section}{cont_section}"
            # f"{deflations}"
        )
        return re.sub(r"(:?\n){3,}", "\n\n", report)

    def print_basic_infos(self, pad: Optional[int] = None) -> None:
        basics = self.ordered_basic_infos()
        pad = pad or max(info.pad for info in basics)
        for info in basics:
            info.print_message(pad)

    def basic_df(self) -> DataFrame:
        dfs = []
        for info in self.ordered_basic_infos():
            dfs.append(info.to_df())
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df.index = Index(df["feature_name"].copy())
        df.drop(columns=["feature_name"], inplace=True)

        df.insert(0, "user", "")
        for col in self.user_cats:
            df.loc[col, "user"] = "cat"
        for col in self.user_ords:
            df.loc[col, "user"] = "ord"

        idx = df["reason"].str.contains("Coerced")
        coerced = df[idx].sort_index()
        df = df[~idx].sort_index().sort_values(by=["user", "inferred"])

        return pd.concat([df, coerced], axis=0, ignore_index=False)

    def to_json(self, path: Path) -> None:
        try:
            path.write_text(str(jsonpickle.encode(self)), encoding="utf-8")
        except Exception as e:
            warn(
                f"Got exception when saving inspection results to .json. Details:\n"
                f"{e}\n{traceback.format_exc()}"
            )

    @staticmethod
    def from_json(path: Path) -> InspectionResults:
        return cast(InspectionResults, jsonpickle.decode(path.read_text()))


@dataclass
class TargetInfo:
    name: str
    needs_logarithm: bool
    has_outliers: bool
    is_classification: bool


@dataclass
class RegTargetInfo:
    name: str
    needs_logarithm: bool
    has_outliers: bool
    p_nan: float


@dataclass
class ClsTargetInfo:
    name: str
    inflation: InflationInfo
    unqs: ndarray
    cnts: ndarray
    p_max_cls: float
    p_min_cls: float
    p_nan: float
