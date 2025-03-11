from typing import Any

import pytest

from hstar.ast import ABS, APP, BOT, COMP, JOIN, TOP, VAR, Term, py_to_ast

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
    # Nested lambdas (returning functions)
    (lambda x: lambda y: x, ABS(ABS(VAR(1)))),
    (lambda x: lambda y: y, ABS(ABS(VAR(0)))),
    (lambda x: lambda y: lambda z: x, ABS(ABS(ABS(VAR(2))))),
    # Higher-order functions
    (lambda f: lambda x: f(f(x)), ABS(ABS(APP(VAR(1), APP(VAR(1), VAR(0)))))),
    (
        lambda f: lambda g: lambda x: f(g(x)),
        ABS(ABS(ABS(APP(VAR(2), APP(VAR(1), VAR(0)))))),
    ),
    # Multiple applications of the same variable
    (lambda x: x(x), ABS(APP(VAR(0), VAR(0)))),
    (lambda x: x(x)(x), ABS(APP(APP(VAR(0), VAR(0)), VAR(0)))),
    # Complex combinations of JOIN and APP
    (lambda x, y: x | y(x), ABS(ABS(JOIN(VAR(1), APP(VAR(0), VAR(1)))))),
    (lambda x, y: (x | y)(x), ABS(ABS(APP(JOIN(VAR(1), VAR(0)), VAR(1))))),
    (
        lambda x, y, z: (x | y)(z | x),
        ABS(ABS(ABS(APP(JOIN(VAR(2), VAR(1)), JOIN(VAR(0), VAR(2)))))),
    ),
    # Multiple arguments with different reference patterns
    (lambda x, y, z: x(y)(z), ABS(ABS(ABS(APP(APP(VAR(2), VAR(1)), VAR(0)))))),
    (lambda x, y, z: x(z)(y), ABS(ABS(ABS(APP(APP(VAR(2), VAR(0)), VAR(1)))))),
    # Identity functions with more arguments
    (lambda w, x, y, z: w, ABS(ABS(ABS(ABS(VAR(3)))))),
    (lambda w, x, y, z: z, ABS(ABS(ABS(ABS(VAR(0)))))),
    # More complex examples
    (
        lambda f, g, x: f(g(x)) | g(f(x)),
        ABS(
            ABS(
                ABS(
                    JOIN(
                        APP(VAR(2), APP(VAR(1), VAR(0))),
                        APP(VAR(1), APP(VAR(2), VAR(0))),
                    )
                )
            )
        ),
    ),
    (
        lambda f: lambda x: f(lambda y: x(y)),
        ABS(ABS(APP(VAR(1), ABS(APP(VAR(1), VAR(0)))))),
    ),
    # Function composition examples
    (lambda f, g: f * g, ABS(ABS(COMP(VAR(1), VAR(0))))),
    (lambda f, g, x: (f * g)(x), ABS(ABS(ABS(APP(COMP(VAR(2), VAR(1)), VAR(0)))))),
    # The multiply operator creating composition
    (lambda f, g, x: f * g * x, ABS(ABS(ABS(COMP(COMP(VAR(2), VAR(1)), VAR(0)))))),
    (lambda f, g, h: f * (g * h), ABS(ABS(ABS(COMP(VAR(2), COMP(VAR(1), VAR(0))))))),
    # More complex compositions
    (
        lambda f, g, h: (f * g) | (g * h),
        ABS(ABS(ABS(JOIN(COMP(VAR(2), VAR(1)), COMP(VAR(1), VAR(0)))))),
    ),
    # Composition with application
    (
        lambda f, g, x, y: f(x) * g(y),
        ABS(ABS(ABS(ABS(COMP(APP(VAR(3), VAR(1)), APP(VAR(2), VAR(0))))))),
    ),
]


@pytest.mark.parametrize(
    "pythonic, expected", EXAMPLES, ids=[str(e) for _, e in EXAMPLES]
)
def test_py_to_ast(pythonic: Any, expected: Term) -> None:
    assert py_to_ast(pythonic) == expected
