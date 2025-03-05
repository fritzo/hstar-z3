"""Tests for the hstar/grammar.py module."""

import pytest
from immutables import Map

from hstar.grammar import (
    APP,
    BOT,
    JOIN,
    LAM,
    TOP,
    VAR,
    Term,
    TermType,
    shift,
    subst,
)


def test_hash_consing() -> None:
    """Test that identical terms are hash-consed to the same object."""
    # Test basic terms
    assert VAR(0) is VAR(0)
    assert VAR(1) is VAR(1)

    # Test compound terms
    assert JOIN(VAR(0), VAR(1)) is JOIN(VAR(0), VAR(1))
    assert APP(VAR(0), JOIN(VAR(1))) is APP(VAR(0), JOIN(VAR(1)))

    # Test associativity, commutativity, and idempotence of JOIN
    assert JOIN(VAR(0), JOIN(VAR(1), VAR(2))) is JOIN(JOIN(VAR(0), VAR(1)), VAR(2))
    assert JOIN(VAR(0), VAR(1)) is JOIN(VAR(1), VAR(0))
    assert JOIN(VAR(0), VAR(0)) is VAR(0), VAR(1)
    assert JOIN(TOP) is TOP
    assert JOIN(BOT) is BOT

    # Test that different terms are different objects
    assert VAR(0) is not VAR(1)

    # Test LAM hash consing
    lambda_term1 = LAM(JOIN(VAR(0)))
    lambda_term2 = LAM(JOIN(VAR(0)))
    assert lambda_term1 is lambda_term2

    # Test that TOP absorbs other terms in a join
    assert JOIN(TOP, VAR(0)) is TOP


def test_shift_operation() -> None:
    """Test that the shift operation works correctly."""
    # Test variable shifting
    assert shift(VAR(0)) is VAR(1)
    assert shift(VAR(1)) is VAR(2)

    # Test shifting with a custom start point
    assert shift(VAR(0), 1) is VAR(0)  # No shift, as index < starting_at
    assert shift(VAR(1), 1) is VAR(2)  # Shift by 1, as index >= starting_at
    assert shift(VAR(2), 2) is VAR(3)  # Only shift if index >= starting_at

    # Test that TOP and BOT are invariant under shift
    assert shift(TOP) is TOP
    assert shift(BOT) is BOT

    # Test shifting of lambda terms
    lam_var0 = LAM(JOIN(VAR(0)))
    assert shift(lam_var0) is lam_var0  # Bound variable, shouldn't change

    # Test that free variables in abstractions are correctly shifted
    lam_with_free = LAM(JOIN(VAR(1)))  # \x.y
    assert shift(lam_with_free) is LAM(JOIN(VAR(2)))  # \x.z

    # Test application shift
    app_term = APP(VAR(0), JOIN(VAR(1)))
    shifted_app = shift(app_term)
    assert shifted_app is APP(VAR(1), JOIN(VAR(2)))

    # Test nested terms
    nested = APP(VAR(0), JOIN(APP(VAR(1), JOIN(VAR(2)))))
    shifted_nested = shift(nested)
    assert shifted_nested is APP(VAR(1), JOIN(APP(VAR(2), JOIN(VAR(3)))))


@pytest.mark.xfail(reason="TODO")
def test_subst_operation() -> None:
    """Test that the substitution operation works correctly."""
    # Basic variable substitution
    assert subst(VAR(0), 0, JOIN(TOP)) is TOP  # [TOP/0]0 = TOP
    assert subst(VAR(1), 0, JOIN(TOP)) is VAR(1)  # [TOP/0]1 = 1
    assert subst(VAR(0), 1, JOIN(TOP)) is VAR(0)  # [TOP/1]0 = 0
    assert subst(VAR(1), 1, JOIN(TOP)) is TOP  # [TOP/1]1 = TOP

    # Identity function substitution
    id_term = LAM(JOIN(VAR(0)))  # \x.x
    assert subst(id_term, 0, JOIN(TOP)) is id_term  # [TOP/0](\x.x) = \x.x
    assert subst(id_term, 1, JOIN(TOP)) is id_term  # [TOP/1](\x.x) = \x.x

    # Application substitution
    app_term = APP(VAR(0), JOIN(VAR(1)))  # 0 1
    assert subst(app_term, 0, JOIN(TOP)) is APP(
        TOP, JOIN(VAR(1))
    )  # [TOP/0](0 1) = TOP 1
    assert subst(app_term, 1, JOIN(TOP)) is APP(
        VAR(0), JOIN(TOP)
    )  # [TOP/1](0 1) = 0 TOP

    # Nested abstraction substitution
    # \x.1 x
    nested = LAM(JOIN(APP(VAR(1), JOIN(VAR(0)))))
    # [TOP/1](\x.1 x)
    subst_result = subst(nested, 1, JOIN(TOP))
    expected = LAM(JOIN(APP(TOP, JOIN(VAR(0)))))
    assert subst_result is expected

    # More complex substitution cases
    complex_term = LAM(
        JOIN(APP(VAR(0), JOIN(APP(VAR(1), JOIN(VAR(2))))))
    )  # \x. x (1 2)
    # When substituting for var 1
    result = subst(complex_term, 1, JOIN(TOP))
    # Expected: \x. x (TOP 2)
    expected = LAM(JOIN(APP(VAR(0), JOIN(APP(TOP, JOIN(VAR(2)))))))
    assert result is expected

    # Join substitution
    join_term = JOIN(VAR(0), VAR(1))  # 0 | 1
    assert subst(join_term, 0, JOIN(TOP)) is JOIN(TOP, VAR(1))
    assert subst(join_term, 1, JOIN(TOP)) is JOIN(VAR(0), TOP)

    # Test BOT substitution
    assert subst(VAR(0), 0, BOT) is BOT
    assert subst(APP(VAR(0), JOIN(VAR(1))), 0, BOT) is APP(BOT, JOIN(VAR(1)))


@pytest.mark.xfail(reason="TODO")
def test_lam_operation() -> None:
    """Test that the LAM operation works correctly with hash consing."""
    # Test basic lambda abstraction
    lam0 = LAM(JOIN(VAR(0)))  # \x.x
    free_vars = Map({0: 1})
    assert lam0.parts == frozenset(
        [Term(TermType.ABS1, head=Term(TermType.VAR, varname=0, free_vars=free_vars))]
    )

    # Test that LAM correctly identifies different cases (ABS0, ABS1, ABS)
    # ABS0 - zero occurrences
    lam_constant = LAM(JOIN(VAR(1)))  # \x.y
    assert list(lam_constant.parts)[0].typ == TermType.ABS0

    # ABS1 - one occurrence
    lam_linear = LAM(JOIN(VAR(0)))  # \x.x
    assert list(lam_linear.parts)[0].typ == TermType.ABS1

    # ABS - two or more occurrences
    lam_nonlinear = LAM(JOIN(APP(VAR(0), JOIN(VAR(0)))))  # \x.x x
    assert list(lam_nonlinear.parts)[0].typ == TermType.ABS

    # Test that LAM is idempotent with hash consing
    assert LAM(JOIN(VAR(0))) is LAM(JOIN(VAR(0)))

    # Test that LAM(TOP) is TOP
    assert LAM(TOP) is TOP


@pytest.mark.xfail(reason="TODO")
def test_eager_linear_reduction() -> None:
    # JOIN reduction
    assert JOIN(TOP) == TOP
    assert JOIN(BOT) == BOT
    assert JOIN(VAR(0)) == VAR(0)
    assert JOIN(VAR(0), VAR(0)) == VAR(0)
    assert JOIN(VAR(0), VAR(1)) == JOIN(VAR(0), VAR(1))
    assert JOIN(JOIN(VAR(0), VAR(1)), VAR(2)) == JOIN(VAR(0), VAR(1), VAR(2))
    assert JOIN(VAR(0), JOIN(VAR(1), VAR(2))) == JOIN(VAR(0), VAR(1), VAR(2))
    assert JOIN(BOT, VAR(0)) is VAR(0)
    assert JOIN(TOP, VAR(0)) is TOP

    # Linear beta reduction
    assert LAM(TOP) == TOP
    assert LAM(BOT) == BOT
    assert APP(LAM(VAR(1)), VAR(1)) == VAR(0)
    assert APP(LAM(VAR(0)), VAR(1)) == VAR(1)
