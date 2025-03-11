from typing import Any

import pytest

from hstar.ast import ABS, APP, BOT, JOIN, TOP, VAR, Term, py_to_ast

EXAMPLES: list[tuple[Any, Term]] = [
    (lambda x: x, ABS(VAR(0))),
    (lambda x, y: x, ABS(ABS(VAR(1)))),
    (lambda x, y: y, ABS(ABS(VAR(0)))),
    (lambda x: TOP, ABS(TOP)),
    (lambda x: BOT, ABS(BOT)),
    (lambda x, y: JOIN(x, y), ABS(ABS(JOIN(VAR(1), VAR(0))))),
    (lambda x, y: APP(x, y), ABS(ABS(APP(VAR(1), VAR(0))))),
    (lambda x, y: x(y), ABS(ABS(APP(VAR(1), VAR(0))))),
    (lambda x, y: x | y, ABS(ABS(JOIN(VAR(1), VAR(0))))),
    (lambda x, y, z: x | y | z, ABS(ABS(ABS(JOIN(JOIN(VAR(2), VAR(1)), VAR(0)))))),
    (
        lambda x, y, z: x(z, y(z)),
        ABS(ABS(ABS(APP(APP(VAR(2), VAR(0)), APP(VAR(1), VAR(0)))))),
    ),
]


@pytest.mark.parametrize(
    "pythonic, expected", EXAMPLES, ids=[str(e) for _, e in EXAMPLES]
)
def test_py_to_ast(pythonic: Any, expected: Term) -> None:
    assert py_to_ast(pythonic) == expected
