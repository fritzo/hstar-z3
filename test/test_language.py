import logging
from typing import Any

import pytest
import z3

from hstar.language import (
    ABS,
    APP,
    BOT,
    COMP,
    JOIN,
    TOP,
    TUPLE,
    VAR,
    B,
    C,
    I,
    K,
    S,
    Term,
    Y,
    hoas,
    shift,
    subst,
)

logger = logging.getLogger(__name__)

f, g, h = z3.Consts("f g h", Term)
r, s, t = z3.Consts("r s t", Term)
x, y, z = z3.Consts("x y z", Term)
v0 = VAR(0)
v1 = VAR(1)
v2 = VAR(2)


def test_shift_eager() -> None:
    """Test eager evaluation of shift function."""
    # Test variable shifting
    assert shift(VAR(0)) == VAR(1)
    assert shift(VAR(1)) == VAR(2)
    assert shift(VAR(2)) == VAR(3)

    # Test variable with custom starting point
    assert shift(VAR(0), 1) == VAR(0)  # No shift, index < starting_at
    assert shift(VAR(1), 1) == VAR(2)  # Shift by 1, index >= starting_at

    # Test abstraction
    assert shift(ABS(VAR(0))) == ABS(VAR(0))  # Index bound by abstraction, so no shift
    assert shift(ABS(VAR(1))) == ABS(VAR(2))  # Free variable gets shifted

    # Test application
    assert shift(APP(VAR(0), VAR(1))) == APP(VAR(1), VAR(2))

    # Test join
    assert shift(JOIN(VAR(0), VAR(1))) == JOIN(VAR(1), VAR(2))

    # Test composition
    assert shift(COMP(VAR(0), VAR(1))) == COMP(VAR(1), VAR(2))

    # Test constants remain unchanged
    assert shift(TOP) == TOP
    assert shift(BOT) == BOT

    # Test basic combinators
    assert shift(I) == I
    assert shift(K) == K
    assert shift(B) == B
    assert shift(C) == C
    assert shift(S) == S
    assert shift(Y) == Y

    # Test nested structures
    assert shift(ABS(APP(VAR(0), VAR(1)))) == ABS(APP(VAR(0), VAR(2)))
    assert shift(APP(ABS(VAR(0)), VAR(1))) == APP(ABS(VAR(0)), VAR(2))


# Each example is a pair of (python_lambda, expected_z3_term)
HOAS_EXAMPLES = [
    (lambda x: x, ABS(VAR(0))),
    (lambda x, y: x, ABS(ABS(VAR(1)))),
    (lambda x, y: y, ABS(ABS(VAR(0)))),
    (lambda x: TOP, ABS(TOP)),
    (lambda x: BOT, ABS(BOT)),
    (lambda x, y: JOIN(x, y), ABS(ABS(JOIN(VAR(1), VAR(0))))),
    (lambda x, y: APP(x, y), ABS(ABS(APP(VAR(1), VAR(0))))),
    pytest.param(
        lambda x, y: x(y),
        ABS(ABS(APP(VAR(1), VAR(0)))),
        marks=[pytest.mark.xfail(reason="TODO handle Python application")],
    ),
    pytest.param(
        lambda x, y: x | y,  # Python or
        ABS(ABS(JOIN(VAR(1), VAR(0)))),
        marks=[pytest.mark.xfail(reason="TODO convert Python or to JOIN")],
    ),
    pytest.param(
        lambda x, y: (x, y),  # Python tuple
        ABS(ABS(TUPLE(VAR(1), VAR(0)))),
        marks=[pytest.mark.xfail(reason="TODO convert Python tuples to TUPLE")],
    ),
]


@pytest.mark.parametrize("pythonic, expected", HOAS_EXAMPLES, ids=str)
def test_hoas(pythonic: Any, expected: z3.ExprRef) -> None:
    assert hoas(pythonic) == expected


def test_subst_eager() -> None:
    """Test eager evaluation of substitution function."""
    # Basic variable substitution
    assert subst(0, TOP, VAR(0)) == TOP  # [TOP/0]0 = TOP
    assert subst(0, TOP, VAR(1)) == VAR(1)  # [TOP/0]1 = 1
    assert subst(1, TOP, VAR(0)) == VAR(0)  # [TOP/1]0 = 0
    assert subst(1, TOP, VAR(1)) == TOP  # [TOP/1]1 = TOP

    # Identity function substitution
    id_term = ABS(VAR(0))  # \x.x
    assert subst(0, TOP, id_term) == id_term  # [TOP/0](\x.x) = \x.x
    assert subst(1, TOP, id_term) == id_term  # [TOP/1](\x.x) = \x.x

    # Application substitution
    app_term = APP(VAR(0), VAR(1))  # 0 1
    assert subst(0, TOP, app_term) == APP(TOP, VAR(1))  # [TOP/0](0 1) = TOP 1
    assert subst(1, TOP, app_term) == APP(VAR(0), TOP)  # [TOP/1](0 1) = 0 TOP

    # Nested abstraction substitution
    nested = ABS(APP(VAR(1), VAR(0)))  # \x.1 x
    # For [TOP/0](\x.1 x), the VAR(1) becomes VAR(2) under the abstraction,
    # so TOP gets shifted
    assert subst(0, TOP, nested) == ABS(APP(TOP, VAR(0)))
    # For [TOP/1](\x.1 x), VAR(1) inside abstraction doesn't match VAR(2)
    # (the shifted index)
    assert (
        subst(1, TOP, nested) == nested
    )  # Term unchanged - VAR(1) inside abstraction is different from outer VAR(1)

    # More complex term with bound and free variables
    complex_term = ABS(APP(VAR(0), APP(VAR(1), VAR(2))))  # \x. x (1 2)
    # When substituting for var 1
    result = subst(1, TOP, complex_term)
    # Expected: \x. x (1 TOP) - this is what our implementation gives
    expected = ABS(APP(VAR(0), APP(VAR(1), TOP)))
    assert result == expected

    # When substituting for var 0
    result2 = subst(0, TOP, complex_term)
    # Expected: \x. x (TOP 2) - this is what our implementation gives
    expected2 = ABS(APP(VAR(0), APP(TOP, VAR(2))))
    assert result2 == expected2

    # Join substitution
    join_term = JOIN(VAR(0), VAR(1))  # 0 | 1
    assert subst(0, TOP, join_term) == JOIN(TOP, VAR(1))  # [TOP/0](0|1) = TOP|1
    assert subst(1, TOP, join_term) == JOIN(VAR(0), TOP)  # [TOP/1](0|1) = 0|TOP
