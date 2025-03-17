import logging
from typing import Any

import pytest
import z3
from z3 import ForAll

from hstar import normal
from hstar.bridge import z3_to_nf
from hstar.language import (
    ABS,
    APP,
    BOT,
    CB,
    COMP,
    JOIN,
    TOP,
    TUPLE,
    VAR,
    B,
    C,
    I,
    J,
    K,
    QEHindley,
    S,
    Term,
    W,
    Y,
    abstract,
    app,
    forall_to_open,
    free_vars,
    hoas,
    iter_closure_maps,
    iter_closures,
    iter_eta_substitutions,
    shift,
    subst,
)
from hstar.theory import hindley_theory

logger = logging.getLogger(__name__)

a = z3.Const("a", Term)
f, g, h = z3.Consts("f g h", Term)
r, s, t = z3.Consts("r s t", Term)
x, y, z = z3.Consts("x y z", Term)
v0 = VAR(0)
v1 = VAR(1)
v2 = VAR(2)


def assert_eq_weak(lhs_z3: z3.ExprRef, rhs_z3: z3.ExprRef) -> None:
    """Weakly check equality of two z3 expressions using hstar.normal."""
    lhs = z3_to_nf(lhs_z3)
    rhs = z3_to_nf(rhs_z3)
    fresh = 1 + max(set(lhs.free_vars) | set(rhs.free_vars), default=-1)

    # Try to normalize by first applying fresh variables, then approximating.
    for i in range(4):
        if normal.is_normal(lhs) and normal.is_normal(rhs):
            break
        var = normal.VAR(fresh + i)
        lhs = normal.APP(lhs, var)
        rhs = normal.APP(rhs, var)
    lhs = normal.approximate(lhs)
    rhs = normal.approximate(rhs)

    # Check equality.
    if lhs != rhs:
        logger.error(f"{lhs} != {rhs}")
    assert lhs == rhs


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

    # Test equations
    assert z3.eq(shift(v0 == v1), VAR(1) == VAR(2))


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

    # Equation substitution
    assert z3.eq(subst(0, TOP, v0 == v1), TOP == v1)
    assert z3.eq(subst(1, TOP, v0 == v1), v0 == TOP)


def test_free_vars() -> None:
    # Note ABS binds VAR(0) and shifts other variables.
    assert set(free_vars(v0)) == {0}
    assert set(free_vars(v1)) == {1}
    assert set(free_vars(v2)) == {2}
    assert set(free_vars(v0 == v1)) == {0, 1}
    assert set(free_vars(APP(v0, v1))) == {0, 1}
    assert set(free_vars(ABS(v0))) == set()
    assert set(free_vars(ABS(v1))) == {0}
    assert set(free_vars(ABS(v2))) == {1}
    assert set(free_vars(ABS(ABS(v0)))) == set()
    assert set(free_vars(ABS(ABS(v1)))) == set()
    assert set(free_vars(ABS(ABS(v2)))) == {0}
    assert set(free_vars(ABS(APP(v0, v1)))) == {0}
    assert set(free_vars(APP(ABS(v0), v1))) == {1}
    assert set(free_vars(APP(ABS(v0), APP(v1, v2)))) == {1, 2}
    assert set(free_vars(ABS(APP(ABS(v0), v1)))) == {0}
    assert set(free_vars(APP(ABS(APP(v0, v1)), v2))) == {0, 2}


ABSTRACTION_EXAMPLES = [
    # Term doesn't contain v0 - K abstraction
    ("const", v1, app(K, v0)),
    # I abstraction - term is exactly v0
    ("var", v0, I),
    # APP cases:
    # Case 1: lhs has v0, rhs is v0 - W abstraction
    ("app-w", app(v1, v0, v0), app(W, v0)),
    # Case 2: lhs has v0, rhs has v0 but isn't v0 - S abstraction
    ("app-s", app(v1, v0, app(v2, v0)), app(S, v0, v1)),
    # Case 3: lhs has v0, rhs doesn't have v0 - C abstraction
    ("app-c", app(v1, v0, v2), app(C, v0, v1)),
    # Case 4: lhs doesn't have v0, rhs is v0 - lhs identity
    ("app-id", app(v1, v0), v0),
    # Case 5: lhs doesn't have v0, rhs has v0 but isn't v0 - COMP abstraction
    ("app-comp", app(v1, app(v2, v0)), COMP(v0, v1)),
    # COMP cases:
    # Case 1: lhs has v0, rhs has v0 - S (B o lhs) rhs abstraction
    ("comp-s", COMP(app(v1, v0), app(v2, v0)), app(S, COMP(B, v0), v1)),
    # Case 2: lhs is v0, rhs doesn't have v0 - CB rhs abstraction
    ("comp-cb", COMP(v0, v1), app(CB, v0)),
    # Case 3: lhs has v0 but isn't v0, rhs doesn't have v0 - COMP(APP(CB, rhs), lhs)
    ("comp-cb-lhs", COMP(app(v1, v0), v2), COMP(app(CB, v1), v0)),
    # Case 4: lhs doesn't have v0, rhs is v0 - B lhs abstraction
    ("comp-b-lhs", COMP(v1, v0), app(B, v0)),
    # Case 5: lhs doesn't have v0, rhs has v0 but isn't v0 - COMP(APP(B, lhs), rhs)
    ("comp-b-rhs", COMP(v1, app(v2, v0)), COMP(app(B, v0), v1)),
    # JOIN cases:
    # Case 1: lhs is v0, rhs is v0 - I abstraction (idempotent join)
    ("join-id", JOIN(v0, v0), I),
    # Case 2: lhs is v0, rhs has v0 but isn't v0 - S J rhs
    ("join-s-rhs", JOIN(v0, app(v1, v0)), app(S, J, v0)),
    # Case 3: lhs is v0, rhs doesn't have v0 - J rhs
    ("join-j-rhs", JOIN(v0, v1), app(J, v0)),
    # Case 4: lhs has v0 but isn't v0, rhs is v0 - S J lhs
    ("join-s-lhs", JOIN(app(v1, v0), v0), app(S, J, v0)),
    # Case 5: lhs has v0 but isn't v0, rhs has v0 but isn't v0 - JOIN(lhs, rhs)
    ("join-lhs-rhs", JOIN(app(v1, v0), app(v2, v0)), JOIN(v0, v1)),
    # Case 6: lhs has v0 but isn't v0, rhs doesn't have v0 - COMP(APP(J, rhs), lhs)
    ("join-comp-rhs", JOIN(app(v1, v0), v2), COMP(app(J, v1), v0)),
    # Case 7: lhs doesn't have v0, rhs is v0 - J lhs
    ("join-j-lhs", JOIN(v1, v0), app(J, v0)),
    # Case 8: lhs doesn't have v0, rhs has v0 but isn't v0 - COMP(APP(J, lhs), rhs)
    ("join-comp-lhs", JOIN(v1, app(v2, v0)), COMP(app(J, v0), v1)),
    # Equality
    ("eq", v0 == v1, I == app(K, v0)),
]
ABSTRACTION_IDS = [str(x[0]) for x in ABSTRACTION_EXAMPLES]


@pytest.mark.parametrize("_, expr, expected", ABSTRACTION_EXAMPLES, ids=ABSTRACTION_IDS)
def test_abstract(_: str, expr: z3.ExprRef, expected: z3.ExprRef) -> None:
    actual = abstract(expr)
    if not z3.is_eq(expr):
        # Check free variables
        ABS_expr = ABS(expr)
        assert free_vars(actual) == free_vars(ABS_expr)

        # Check back substitution
        back = APP(shift(actual), v0)
        assert_eq_weak(back, expr)

    # Check hand-coded expectation
    assert z3.eq(actual, expected)


def test_iter_eta_substitutions() -> None:
    # Test with a variable
    actual = set(iter_eta_substitutions(v1))
    expected = {v1, abstract(v0), abstract(APP(v1, v0))}
    assert actual == expected

    # Test with application term
    term = APP(v0, v0)
    actual = set(iter_eta_substitutions(term))
    expected = {
        term,  # [x/x]
        abstract(subst(0, VAR(2), term), 2),  # [x/a]
        abstract(subst(0, APP(v0, VAR(2)), term), 2),  # [x/APP x a]
    }
    assert actual == expected

    # Test with complex term including abstraction
    term = APP(ABS(v0), v1)
    actual = set(iter_eta_substitutions(term))
    expected = {
        term,  # [x/x]
        abstract(subst(1, VAR(2), term), 2),  # [x/a]
        abstract(subst(1, APP(v1, VAR(2)), term), 2),  # [x/APP x a]
    }
    assert actual == expected


CLOSURE_MAPS_EXAMPLES = [
    (v0, {I}),
    (APP(v0, v0), {APP(W, I)}),
    (APP(v0, v1), {I, APP(C, I), APP(W, I)}),
]


@pytest.mark.parametrize(
    "term, expected",
    CLOSURE_MAPS_EXAMPLES,
    ids=[str(x[0]) for x in CLOSURE_MAPS_EXAMPLES],
)
def test_iter_closure_maps(term: z3.ExprRef, expected: set[z3.ExprRef]) -> None:
    actual = set(iter_closure_maps(term))
    assert actual == expected


def test_iter_eta_substitutions_beta() -> None:
    """Test eta substitutions for beta reduction axiom."""
    # The open form of ForAll(x, APP(ABS(VAR(0)), x) == x)
    beta_eq = APP(ABS(v0), v0) == v0

    # Get all eta substitutions
    substitutions = list(iter_eta_substitutions(beta_eq))

    # Print for debugging
    for i, expr in enumerate(substitutions):
        logger.debug(f"Substitution {i}: {expr}")

    # Check that we get the expected number of substitutions
    assert len(substitutions) == 3  # Should be 3: [x/x], [x/a], [x/APP x a]

    # The original equation should be in the list
    assert beta_eq in substitutions


def test_iter_closure_maps_beta() -> None:
    """Test closure maps for beta reduction axiom."""
    # The open form of ForAll(x, APP(ABS(VAR(0)), x) == x)
    beta_eq = APP(ABS(v0), v0) == v0

    # Get all closure maps
    closures = list(iter_closure_maps(beta_eq))

    # Print for debugging
    for i, closure in enumerate(closures):
        logger.debug(f"Closure {i}: {closure}")
        lhs, rhs = closure.children()
        logger.debug(f"  Free vars: {free_vars(lhs) | free_vars(rhs)}")

    # All resulting expressions should have fewer free variables
    for expr in closures:
        assert len(free_vars(expr)) < len(free_vars(beta_eq))


@pytest.mark.xfail(reason="FIXME")
def test_iter_closures_beta() -> None:
    """Test combined closures for beta reduction axiom."""
    # The open form of ForAll(x, APP(ABS(VAR(0)), x) == x)
    beta_eq = APP(ABS(v0), v0) == v0

    # Get all closures
    closures = list(iter_closures(beta_eq))

    # Print for debugging
    for i, closure in enumerate(closures):
        logger.debug(f"Complete closure {i}: {closure}")
        lhs, rhs = closure.children()
        logger.debug(f"  LHS: {lhs}")
        logger.debug(f"  RHS: {rhs}")
        logger.debug(f"  Free vars: {free_vars(lhs) | free_vars(rhs)}")
        assert_eq_weak(lhs, rhs)  # FIXME fails

    # All resulting expressions should be closed (no free variables)
    for expr in closures:
        assert not free_vars(expr), f"Expression has free variables: {expr}"


def test_qe_hindley_count() -> None:
    actual = QEHindley(ForAll([x], app(I, x) == x))
    assert len(actual) == 1

    actual = QEHindley(ForAll([x, y], app(K, x, y) == x))
    assert len(actual) == 13

    actual = QEHindley(ForAll([x, y, z], app(B, x, y, z) == x))
    assert len(actual) == 134


FORALL_TO_OPEN_EXAMPLES = [
    (z3.ForAll([x], app(I, x) == x), app(I, v0) == v0),
    (z3.ForAll([x, y], app(K, x, y) == x), app(K, v0, v1) == v0),
    (
        z3.ForAll([x, y, z], app(B, x, y, z) == app(x, app(y, z))),
        app(B, v0, v1, v2) == app(v0, app(v1, v2)),
    ),
]


@pytest.mark.parametrize(
    "expr, expected",
    FORALL_TO_OPEN_EXAMPLES,
    ids=[str(x[1]) for x in FORALL_TO_OPEN_EXAMPLES],
)
def test_forall_to_open(expr: z3.ExprRef, expected: z3.ExprRef) -> None:
    actual = forall_to_open(expr)
    assert z3.eq(actual, expected)


HINDLEY_AXIOMS = hindley_theory()
HINDLEY_IDS = [" ".join(str(x).split()) for x in HINDLEY_AXIOMS]


@pytest.mark.xfail(reason="FIXME")
@pytest.mark.parametrize("axiom", HINDLEY_AXIOMS, ids=HINDLEY_IDS)
def test_qe_hindley(axiom: z3.ExprRef) -> None:
    equations = list(QEHindley(axiom))
    assert equations
    for e in equations:
        assert z3.is_eq(e)
        assert not free_vars(e)
        lhs, rhs = e.children()
        assert_eq_weak(lhs, rhs)
