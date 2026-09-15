from time import perf_counter
from typing import Any, Callable


def timed(f: Callable, times: dict[str, float]) -> Callable:
    def _f(*args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        results = f(*args, **kwargs)
        elapsed = perf_counter() - start
        times.update({f.__name__: elapsed})
        return results

    return _f
