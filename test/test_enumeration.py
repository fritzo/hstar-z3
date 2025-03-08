import itertools

import pytest
from immutables import Map

from hstar.enumeration import (
    Refiner,
    enumerator,
    env_enumerator,
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


EXAMPLE_FREE_VARS = [
    Map({0: 1}),
    Map({0: 1, 1: 2}),
    Map({0: 1, 1: 2, 2: 3}),
]


@pytest.mark.parametrize("free_vars", EXAMPLE_FREE_VARS)
def test_env_enumerator(free_vars: Map[int, int]) -> None:
    enumerator = env_enumerator(free_vars)
    actual = list(itertools.islice(enumerator, 1000))
    # print("\n".join(str(x) for x in actual))
    for env in actual:
        assert isinstance(env, Map)
        assert set(env.keys()) <= set(free_vars.keys())


@pytest.mark.xfail(reason="timeout")
@pytest.mark.timeout(0.1)
def test_term_refiner() -> None:
    sketch = ABS(APP(APP(VAR(0), VAR(1)), VAR(2)))
    refiner = Refiner(sketch)
    refiner.validate()
    for _ in range(100):
        candidate = refiner.next_candidate()
        # print(candidate)
        assert isinstance(candidate, Term)
        refiner.validate()
