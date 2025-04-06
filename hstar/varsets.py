from functools import cache

from immutables import Map

from .hashcons import intern

EMPTY_VARS: Map[int, int] = intern(Map())


@cache
def max_vars(*args: Map[int, int]) -> Map[int, int]:
    """Element-wise maximum of multiple maps of variables."""
    if not args:
        return EMPTY_VARS
    result = dict(args[0])
    for arg in args[1:]:
        for k, v in arg.items():
            result[k] = max(result.get(k, 0), v)
    return intern(Map(result))


@cache
def min_vars(*args: Map[int, int]) -> Map[int, int]:
    """Element-wise minimum of multiple maps of variables."""
    if not args:
        return EMPTY_VARS
    result = dict(args[0])
    for arg in args[1:]:
        for k, v in arg.items():
            result[k] = min(result.get(k, 0), v)
    for k, v in list(result.items()):
        if v == 0:
            del result[k]
    return intern(Map(result))


@cache
def add_vars(*args: Map[int, int]) -> Map[int, int]:
    """Add multiple maps of variables."""
    result = dict(args[0])
    for arg in args[1:]:
        for k, v in arg.items():
            result[k] = result.get(k, 0) + v
    return intern(Map(result))


@cache
def shift_vars(vars: Map[int, int], *, start: int = 0, delta: int = 1) -> Map[int, int]:
    """Shift the variables in a map."""
    result = {}
    for k, v in vars.items():
        if k < start:
            result[k] = v
        else:
            result[k + delta] = v
    return intern(Map(result))
