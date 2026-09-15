"""
This file is just used for manual bash testing, i.e. test/bash_test_embedding.sh
"""

from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
# ROOT2 = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# sys.path.append(str(ROOT2))  # isort: skip
# fmt: on

from src.df_analyze.embedding.testing import NLPTestingDataset

if __name__ == "__main__":
    for ds in NLPTestingDataset.get_all():
        ds.load()
