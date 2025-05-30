import functools
import inspect
import logging
import weakref
from collections.abc import (
    Callable,
    Hashable,
    MutableMapping,
)
from typing import Any, NewType, ParamSpec, TypeVar
from weakref import WeakKeyDictionary, WeakValueDictionary

from .hashcons import intern
from .metrics import COUNTERS

logger = logging.getLogger(__name__)
counter = COUNTERS[__name__]

Qualname = NewType("Qualname", str)
A = ParamSpec("A")
B = TypeVar("B")
T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def qualname(x: type | Callable) -> Qualname:
    """
    Returns the fully qualified name of a class, function, method, or callable
    object.
    """
    # Unwrap partial objects.
    while isinstance(x, functools.partial):
        x = x.func

    # Check if x is a bound method.
    self = getattr(x, "__self__", None)
    if self is not None:
        cls = type(self)
        return Qualname(f"{cls.__module__}.{cls.__qualname__}.{x.__name__}")

    # Check if x is a function or class.
    if hasattr(x, "__qualname__"):
        return Qualname(f"{x.__module__}.{x.__qualname__}")

    # Assume x is a callable object.
    cls = type(x)
    return Qualname(f"{cls.__module__}.{cls.__qualname__}")


def weak_value_cache(func: Callable[A, B]) -> Callable[A, B]:
    """Like `functools.cache` but stores results in a `WeakValueDictionary`."""
    cache: MutableMapping[Hashable, B] = WeakValueDictionary()

    @functools.wraps(func)
    def wrapper(*args: A.args, **kwargs: A.kwargs) -> B:
        key = (args, frozenset(kwargs.items()))
        try:
            return cache[key]
        except KeyError:
            pass
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper


# Allow some non-cons-hashed args, including None and:
_WEAK_MEMOIZE_ATOMS = (int, float, str)


def _make_arg_kwarg_key(
    args: tuple, kwargs: dict[str, Any]
) -> tuple[Hashable, list[object]]:
    # Concatenate args and kwargs.
    kv: list[tuple[str | None, Any]] = [(None, v) for v in args]
    kv.extend(sorted(kwargs.items()))

    # Convert objects to ids.
    key: list[Hashable] = []
    objects: list[object] = []
    for k, v in kv:
        if v is None or isinstance(v, _WEAK_MEMOIZE_ATOMS):
            # Allow some non-cons-hashed args.
            key.append((k, v))
        else:
            key.append((k, id(v)))
            objects.append(v)
    assert objects, "Weak memoize cache leak"
    return tuple(key), objects


def weak_key_cache(func: Callable[A, B]) -> Callable[A, B]:
    """
    Decorator to memoize a function of variably many hash cons'd args.

    Uses inspect to determine the best caching strategy:
    - For single-argument functions, use _weak_arg_cache with WeakKeyDictionary
    - For functions with multiple args or kwargs, use _weak_args_kwargs_cache
    """
    sig = inspect.signature(func)

    # Check if the function accepts keyword arguments or has multiple parameters
    has_kwargs = any(
        param.kind in (param.VAR_KEYWORD, param.KEYWORD_ONLY)
        for param in sig.parameters.values()
    )
    multiple_params = len(sig.parameters) > 1

    if has_kwargs or multiple_params:
        return _weak_args_kwargs_cache(func)
    else:
        return _weak_arg_cache(func)  # type: ignore


def _weak_arg_cache(func: Callable[[H], B]) -> Callable[[H], B]:
    """
    Decorator to memoize a function with a single hashable argument.
    Uses a WeakKeyDictionary for caching to allow garbage collection.
    """
    cache: WeakKeyDictionary[H, B] = WeakKeyDictionary()
    name = qualname(func)
    miss = name + ".miss"
    hit = name + ".hit"

    @functools.wraps(func)
    def memoized_func(arg: H) -> B:
        arg = intern(arg)

        # Check cache.
        try:
            result = cache[arg]
        except KeyError:
            counter[miss] += 1
        else:
            counter[hit] += 1
            return result

        # Compute result and save in cache.
        result = func(arg)
        cache[arg] = result
        return result

    memoized_func.cache = cache  # type: ignore
    return memoized_func


def _weak_args_kwargs_cache(func: Callable[A, B]) -> Callable[A, B]:
    cache: dict[Hashable, B] = {}
    name = qualname(func)
    miss = name + ".miss"
    hit = name + ".hit"

    @functools.wraps(func)
    def memoized_func(*args: A.args, **kwargs: A.kwargs) -> B:
        # Intern.
        args = tuple(
            a if a is None or isinstance(a, _WEAK_MEMOIZE_ATOMS) else intern(a)
            for a in args
        )  # type: ignore
        kwargs = {
            k: v if v is None or isinstance(v, _WEAK_MEMOIZE_ATOMS) else intern(v)
            for k, v in kwargs.items()
        }  # type: ignore[assignment]

        # Check cache.
        key, objects = _make_arg_kwarg_key(args, kwargs)
        try:
            result = cache[key]
        except KeyError:
            counter[miss] += 1
        else:
            counter[hit] += 1
            return result

        result = func(*args, **kwargs)

        # Save result and register finalizers.
        cache[key] = result
        for arg in objects:
            f = weakref.finalize(arg, cache.pop, key, None)
            f.atexit = False
        return result

    memoized_func.cache = cache  # type: ignore

    return memoized_func
