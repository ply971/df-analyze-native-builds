# This file needs to be a mess to run first to initialize the MEMOIZER object...
import os
import traceback
from pathlib import Path
from typing import Any
from warnings import warn

DF_ENV = "DF_ANALYZE_CACHE"
DF_CACHE_NAME = "__DF_ANALYZE_CACHE__"
TESTFILE = "permission_test"


def is_writeable(dir: Path) -> bool:
    testfile = dir / TESTFILE
    if testfile.exists():
        testfile.unlink(missing_ok=True)
    try:
        with open(testfile, "w"):
            pass
        testfile.unlink(missing_ok=False)
        return True
    except OSError:
        traceback.print_exc()
        return False
    finally:
        testfile.unlink(missing_ok=True)


def get_cache_dir() -> Path:
    existing = os.environ.get(DF_ENV)
    if existing is None:
        try:
            new = Path.home().resolve()
        except Exception as e:
            traceback.print_exc()
            raise EnvironmentError(
                "Could not resolve $HOME environment variable. Details above."
            ) from e
        existing = new / DF_CACHE_NAME
        if not is_writeable(existing):
            existing = Path.cwd().resolve() / DF_CACHE_NAME
        warn(
            f"""
No existing environment variable `DF_ANALYZE_CACHE` currently defined. A
default directory of {existing} will be used.
"""
        )

        os.environ[DF_ENV] = str(existing)

    cache = Path(existing)
    cache.mkdir(exist_ok=True, parents=True)

    if not is_writeable(cache):
        raise EnvironmentError(
            f"""
Unable to write to {cache} (details above). Specify a different cache directory either
by defining a permanent environment variable {DF_ENV} or, if in a Unix-based system, by
specifying the location prior to running `df-analyze`, e.g.:

    {DF_ENV}=<your/desired/cache/path/here> python df-analyze.py <options>

or by defining a persistent environment variable in a system-appropriate way. """
        )
    else:
        return cache


CACHE_DIR = get_cache_dir()


def get_memoizer() -> Any:
    from joblib import Memory

    return Memory(location=CACHE_DIR, compress=9, verbose=1)


MEMOIZER = get_memoizer()
