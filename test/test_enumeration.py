import itertools
import logging

import pytest
from immutables import Map

from hstar.enumeration import (
    EnvRefiner,
    Refiner,
    enumerator,
    env_enumerator,
)
from hstar.normal import ABS, APP, VAR, Env, Term, app, complexity, env_free_vars

logger = logging.getLogger(__name__)


def test_enumerator() -> None:
    actual = list(itertools.islice(enumerator, 1000))
    logger.debug("\n".join(str(x) for x in actual))
    expected = actual[:]
    expected.sort(key=lambda x: (complexity(x), repr(x)))
    assert actual == expected

    for c in range(4):
        for term in enumerator.level(c):
            assert complexity(term) == c


EXAMPLE_FREE_VARS = [
    {0: 1},
    {0: 1, 1: 2},
    {0: 1, 1: 2, 2: 3},
]


@pytest.mark.parametrize("free_vars", EXAMPLE_FREE_VARS, ids=str)
def test_env_enumerator(free_vars: dict[int, int]) -> None:
    enumerator = env_enumerator(Map(free_vars))
    actual = list(itertools.islice(enumerator, 1000))
    logger.debug("\n".join(str(x) for x in actual))
    for env in actual:
        assert isinstance(env, Env)
        assert set(env.keys()) <= set(free_vars.keys())


@pytest.mark.timeout(0.1)
def test_refiner() -> None:
    sketch = ABS(APP(APP(VAR(0), VAR(1)), VAR(2)))
    refiner = Refiner(sketch)
    refiner.validate()
    for _ in range(100):
        candidate = refiner.next_candidate()
        logger.debug(candidate)
        assert isinstance(candidate, Term)
        refiner.validate()


@pytest.mark.timeout(0.2)
def test_env_refiner() -> None:
    sketch = Env(
        {
            0: ABS(ABS(ABS(app(VAR(2), VAR(0), VAR(1))))),  # pair
            1: app(VAR(0), VAR(2), VAR(3)),  # <r,s>
        }
    )
    assert env_free_vars(sketch) == Map({2: 1, 3: 1})
    refiner = EnvRefiner(sketch)
    refiner.validate()
    for _ in range(100):
        candidate = refiner.next_candidate()
        logger.debug(candidate)
        assert isinstance(candidate, Env)
        refiner.validate()
