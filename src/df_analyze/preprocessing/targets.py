from __future__ import annotations

from typing import Sequence, Union

TargetSpec = Union[str, Sequence[str]]


def as_target_list(targets: TargetSpec) -> list[str]:
    if isinstance(targets, str):
        return [targets]
    return list(targets)
