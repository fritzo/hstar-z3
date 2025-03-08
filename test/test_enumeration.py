"""Tests for the hstar/grammar.py module."""

import itertools

import pytest
from immutables import Map

from hstar.enumeration import (
    Refinery,
    enumerator,
    env_enumerator,
    subst_enumerator,
)
from hstar.grammar import (
    ABS,
    APP,
    VAR,
    Term,
    complexity,
)


def test_enumerator() -> None:
    actual = list(itertools.islice(enumerator, 1000))
    # print("\n".join(str(x) for x in actual))
    expected = actual[:]
    expected.sort(key=lambda x: (complexity(x), repr(x)))
    assert actual == expected

    for c in range(4):
        for term in enumerator.level(c):
            assert complexity(term) == c


EXAMPLE_KEYS = [
    frozenset([0]),
    frozenset([0, 1]),
    frozenset([0, 1, 2]),
]


@pytest.mark.parametrize("keys", EXAMPLE_KEYS)
def test_env_enumerator(keys: frozenset[int]) -> None:
    enumerator = env_enumerator(keys)
    actual = list(itertools.islice(enumerator, 1000))
    # print("\n".join(str(x) for x in actual))
    for env in actual:
        assert isinstance(env, Map)
        assert set(env.keys()) <= keys


EXAMPLE_FREE_VARS = [
    Map({0: 1}),
    Map({0: 1, 1: 2}),
    Map({0: 1, 1: 2, 2: 3}),
]


@pytest.mark.parametrize("free_vars", EXAMPLE_FREE_VARS)
def test_subst_enumerator(free_vars: Map[int, int]) -> None:
    enumerator = subst_enumerator(free_vars)
    actual = list(itertools.islice(enumerator, 1000))
    # print("\n".join(str(x) for x in actual))
    for env in actual:
        assert isinstance(env, Map)
        assert set(env.keys()) <= set(free_vars.keys())


@pytest.mark.xfail(reason="timeout")
@pytest.mark.timeout(0.1)
def test_refinery() -> None:
    sketch = ABS(APP(APP(VAR(0), VAR(1)), VAR(2)))
    refinery = Refinery(sketch)
    actual = [refinery.next_candidate() for _ in range(100)]
    # print("\n".join(str(x) for x in actual))
    assert all(isinstance(x, Term) for x in actual)
