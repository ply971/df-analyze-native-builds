import json
from pathlib import Path
from typing import (
    Optional,
)

import pandas as pd
from pandas import DataFrame



def load_json_lines(path: Path) -> DataFrame:
    text = path.read_text()
    lines = [(s.strip() + "}").replace("}}", "}") for s in text.split("}\n")]
    objs = []
    for i, line in enumerate(lines):
        if len(line) <= 1:  # handles last line, blanks
            continue
        try:
            obj = json.loads(line)
            objs.append(obj)
        except json.decoder.JSONDecodeError as e:
            ix_prv = max(0, i - 1)
            ix_nxt = min(i + 1, len(lines) - 1)
            ix_cur = i
            prv = lines[ix_prv]
            nxt = lines[ix_nxt]
            raise ValueError(
                f"Got error parsing line {i}: `{line}` of file: {path}.\n"
                f"[{ix_prv:d}] Previous line: {prv}\n"
                f"[{ix_cur:d}] Current line:  {line}\n"
                f"[{ix_nxt:d}] Next line:     {nxt}\n"
            ) from e
    df = DataFrame(objs).infer_objects().convert_dtypes()
    return df


def _load_datafile(path: Optional[Path]) -> Optional[DataFrame]:
    if path is None:
        return None

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".jsonl":  # json-list, it seems, each line is an object
        return load_json_lines(path)

    try:
        if path.suffix == ".json":
            text = path.read_text()
            info = json.loads(text)
            return DataFrame(info)
    except json.decoder.JSONDecodeError:
        return load_json_lines(path)

    raise ValueError(f"Unrecognized filetype: `{path.suffix}` from data file: {path}")
