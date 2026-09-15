"""PyInstaller hook: put caches in user storage, never inside the installed app."""

import os
import sys
from pathlib import Path

if sys.platform == "darwin":
    cache_root = Path.home() / "Library" / "Caches" / "df-analyze"
elif sys.platform == "win32":
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "df-analyze" / "Cache"
else:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "df-analyze"

for variable, folder in (
    ("DF_ANALYZE_CACHE", cache_root / "analysis"),
    ("MPLCONFIGDIR", cache_root / "matplotlib"),
    ("NUMBA_CACHE_DIR", cache_root / "numba"),
):
    os.environ.setdefault(variable, str(folder))
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)

# A Finder launch may have no attached terminal. Preserve diagnostics and give
# libraries that inspect stdout.encoding a real stream in that case.
if sys.stdout is None or sys.stderr is None:
    cache_root.mkdir(parents=True, exist_ok=True)
    _log = (cache_root / "desktop.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log
