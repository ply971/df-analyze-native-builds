from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from argparse import ArgumentParser
from enum import Enum
from typing import Generic, Mapping, TypeVar


T = TypeVar("T")


class CliOption(Generic[T]):
    def __init__(
        self,
        longnames: list[str],
        shortnames: list[str],
        choices: list[str] | None,
        default: T,
        argparse_kwargs: Mapping | None = None,
    ) -> None:
        self.longnames = longnames
        self.shortnames = shortnames
        self.choices = choices
        self.default = default
        self.kwargs = argparse_kwargs or {}

    def add_to_parser(self, parser: ArgumentParser) -> None:
        kwargs = {}
        if (self.choices is not None) and (len(self.choices) > 0):
            kwargs["choices"] = self.choices
        kwargs["default"] = self.default

        parser.add_argument(*self.longnames, *self.shortnames)

    def default_str(self) -> str:
        if isinstance(self.default, Enum):
            return self.default.value
        return str(self.default)

    def to_line(self) -> str:
        return f"--{self.longnames[0]} {self.default_str()}"

    def to_cell_values(self) -> list[str]:
        return [f"--{self.longnames[0]}", f"{self.default_str()}"]
