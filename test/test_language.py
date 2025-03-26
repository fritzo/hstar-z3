import logging

import pytest
import z3
from z3 import ForAll

from hstar import normal
from hstar.bridge import z3_to_nf
from hstar.language import (
    APP,
    CB,
    COMP,
    JOIN,
    LEQ,
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
    app,
    free_vars,
    is_eq_or_leq,
    iter_closure_maps,
    iter_closures,
    iter_eta_substitutions,
    lam,
)
from hstar.theory import get_theory

logger = logging.getLogger(__name__)

a = z3.Const("a", Term)
f, g, h = z3.Consts("f g h", Term)
r, s, t = z3.Consts("r s t", Term)
x, y, z = z3.Consts("x y z", Term)
v0 = VAR(0)
v1 = VAR(1)
v2 = VAR(2)


def assert_weak(expr: z3.ExprRef) -> None:
    """Weakly check equality of two z3 expressions using hstar.normal."""
    assert is_eq_or_leq(expr)
    lhs, rhs = map(z3_to_nf, expr.children())
    fresh = 1 + max(set(lhs.free_vars) | set(rhs.free_vars), default=-1)

    # Try to normalize by first applying fresh variables, then approximating.
    for i in range(4):
        if not any(part.typ == normal.TermType.ABS for part in (lhs.parts | rhs.parts)):
            break
        var = normal.VAR(fresh + i)
        lhs = normal.APP(lhs, var)
        rhs = normal.APP(rhs, var)
    lhs_lb, lhs_ub = normal.approximate(lhs)
    rhs_lb, rhs_ub = normal.approximate(rhs)

    if z3.is_eq(expr):
        if normal.leq(lhs_lb, rhs_ub) is False:
            raise AssertionError(f"failed {lhs} [= {rhs}")
        if normal.leq(rhs_lb, lhs_ub) is False:
            raise AssertionError(f"failed {rhs} [= {lhs}")
    elif z3.is_app(expr) and str(expr.decl()) == "LEQ":
        if normal.leq(lhs_lb, rhs_ub) is False:
            raise AssertionError(f"failed {lhs} [= {rhs}")
    else:
        raise TypeError(f"unexpected expression: {expr}")


def test_free_vars() -> None:
    # Note ABS binds VAR(0) and shifts other variables.
    assert set(free_vars(v0)) == {v0}
    assert set(free_vars(v1)) == {v1}
    assert set(free_vars(v2)) == {v2}
    assert set(free_vars(v0 == v1)) == {v0, v1}
    assert set(free_vars(app(v0, v1))) == {v0, v1}
    assert set(free_vars(lam(v0, v0))) == set()
    assert set(free_vars(lam(v0, v1))) == {v1}
    assert set(free_vars(lam(v0, v2))) == {v2}
    assert set(free_vars(lam(v1, lam(v0, v0)))) == set()
    assert set(free_vars(lam(v1, lam(v0, v1)))) == set()
    assert set(free_vars(lam(v1, lam(v0, v2)))) == {v2}
    assert set(free_vars(lam(v0, app(v0, v1)))) == {v1}
    assert set(free_vars(app(lam(v0, v0), v1))) == {v1}
    assert set(free_vars(app(lam(v0, v0), app(v1, v2)))) == {v1, v2}
    assert set(free_vars(lam(v0, app(lam(v0, v0), v1)))) == {v1}
    assert set(free_vars(app(lam(v0, app(v0, v1)), v2))) == {v1, v2}


ABSTRACTION_EXAMPLES = [
    # Term doesn't contain v0 - K abstraction
    ("const", v1, app(K, v1)),
    # I abstraction - term is exactly v0
    ("var", v0, I),
    # APP cases:
    # Case 1: lhs has v0, rhs is v0 - W abstraction
    ("app-w", app(v1, v0, v0), app(W, v1)),
    # Case 2: lhs has v0, rhs has v0 but isn't v0 - S abstraction
    ("app-s", app(v1, v0, app(v2, v0)), app(S, v1, v2)),
    # Case 3: lhs has v0, rhs doesn't have v0 - C abstraction
    ("app-c", app(v1, v0, v2), app(C, v1, v2)),
    # Case 4: lhs doesn't have v0, rhs is v0 - lhs identity
    ("app-id", app(v1, v0), v1),
    # Case 5: lhs doesn't have v0, rhs has v0 but isn't v0 - COMP abstraction
    ("app-comp", app(v1, app(v2, v0)), COMP(v1, v2)),
    # COMP cases:
    # Case 1: lhs has v0, rhs has v0 - S (B o lhs) rhs abstraction
    ("comp-s", COMP(app(v1, v0), app(v2, v0)), app(S, COMP(B, v1), v2)),
    # Case 2: lhs is v0, rhs doesn't have v0 - CB rhs abstraction
    ("comp-cb", COMP(v0, v1), app(CB, v1)),
    # Case 3: lhs has v0 but isn't v0, rhs doesn't have v0 - COMP(APP(CB, rhs), lhs)
    ("comp-cb-lhs", COMP(app(v1, v0), v2), COMP(app(CB, v2), v1)),
    # Case 4: lhs doesn't have v0, rhs is v0 - B lhs abstraction
    ("comp-b-lhs", COMP(v1, v0), app(B, v1)),
    # Case 5: lhs doesn't have v0, rhs has v0 but isn't v0 - COMP(APP(B, lhs), rhs)
    ("comp-b-rhs", COMP(v1, app(v2, v0)), COMP(app(B, v1), v2)),
    # JOIN cases:
    # Case 1: lhs is v0, rhs is v0 - I abstraction (idempotent join)
    ("join-id", JOIN(v0, v0), I),
    # Case 2: lhs is v0, rhs has v0 but isn't v0 - S J rhs
    ("join-s-rhs", JOIN(v0, app(v1, v0)), app(S, J, v1)),
    # Case 3: lhs is v0, rhs doesn't have v0 - J rhs
    ("join-j-rhs", JOIN(v0, v1), app(J, v1)),
    # Case 4: lhs has v0 but isn't v0, rhs is v0 - S J lhs
    ("join-s-lhs", JOIN(app(v1, v0), v0), app(S, J, v1)),
    # Case 5: lhs has v0 but isn't v0, rhs has v0 but isn't v0 - JOIN(lhs, rhs)
    ("join-lhs-rhs", JOIN(app(v1, v0), app(v2, v0)), JOIN(v1, v2)),
    # Case 6: lhs has v0 but isn't v0, rhs doesn't have v0 - COMP(APP(J, rhs), lhs)
    ("join-comp-rhs", JOIN(app(v1, v0), v2), COMP(app(J, v2), v1)),
    # Case 7: lhs doesn't have v0, rhs is v0 - J lhs
    ("join-j-lhs", JOIN(v1, v0), app(J, v1)),
    # Case 8: lhs doesn't have v0, rhs has v0 but isn't v0 - COMP(APP(J, lhs), rhs)
    ("join-comp-lhs", JOIN(v1, app(v2, v0)), COMP(app(J, v1), v2)),
    # Relations
    ("eq", v0 == v1, I == app(K, v1)),
    ("leq", LEQ(v0, v1), LEQ(I, app(K, v1))),
]
ABSTRACTION_IDS = [str(x[0]) for x in ABSTRACTION_EXAMPLES]


@pytest.mark.parametrize("_, expr, expected", ABSTRACTION_EXAMPLES, ids=ABSTRACTION_IDS)
def test_lam(_: str, expr: z3.ExprRef, expected: z3.ExprRef) -> None:
    actual = lam(v0, expr)
    if not is_eq_or_leq(expr):
        # Check free variables
        assert free_vars(actual) == free_vars(expr) - {v0}

        # Check back substitution
        back = APP(actual, v0)
        assert_weak(back == expr)

    # Check hand-coded expectation
    assert actual == expected


def test_iter_eta_substitutions() -> None:
    # Test with a variable
    actual = set(iter_eta_substitutions(v1))
    expected = {v1, lam(v0, v0), lam(v0, APP(v1, v0))}
    assert actual == expected

    # Test with application term
    term = APP(v0, v0)
    actual = set(iter_eta_substitutions(term))
    expected = {
        term,  # [x/x]
        lam(v2, z3.substitute(term, (v0, v2))),  # [x/a]
        lam(v2, z3.substitute(term, (v0, APP(v0, v2)))),  # [x/APP x a]
    }
    assert actual == expected

    # Test with complex term
    term = APP(I, v1)
    actual = set(iter_eta_substitutions(term))
    expected = {
        term,  # [x/x]
        lam(v2, z3.substitute(term, (v1, v2))),  # [x/a]
        lam(v2, z3.substitute(term, (v1, APP(v1, v2)))),  # [x/APP x a]
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
    # The open form of ForAll(x, APP(I, x) == x)
    beta_eq = APP(I, v0) == v0

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
    # The open form of ForAll(x, APP(I, x) == x)
    beta_eq = APP(I, v0) == v0

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


def test_iter_closures_beta() -> None:
    """Test combined closures for beta reduction axiom."""
    # The open form of ForAll(x, APP(ABS(VAR(0)), x) == x)
    beta_eq = APP(I, v0) == v0

    # Get all closures
    closures = list(iter_closures(beta_eq))

    # Print for debugging
    for i, expr in enumerate(closures):
        logger.debug(f"Complete closure {i}: {expr}")
        lhs, rhs = expr.children()
        logger.debug(f"  LHS: {lhs}")
        logger.debug(f"  RHS: {rhs}")
        logger.debug(f"  Free vars: {free_vars(lhs) | free_vars(rhs)}")
        assert_weak(expr)

    # All resulting expressions should be closed (no free variables)
    for expr in closures:
        assert not free_vars(expr), f"Expression has free variables: {expr}"


@pytest.mark.parametrize("trivial", [False, True])
def test_qe_hindley_count(trivial: bool) -> None:
    actual = QEHindley(ForAll([x], app(I, x) == x), trivial=trivial)
    assert len(actual) == [1, 5][trivial]

    actual = QEHindley(ForAll([x, y], app(K, x, y) == x), trivial=trivial)
    assert len(actual) == [13, 47][trivial]

    actual = QEHindley(ForAll([x, y, z], app(B, x, y, z) == x), trivial=trivial)
    assert len(actual) == [134, 516][trivial]


HINDLEY_AXIOMS = [
    ax
    for _, ax in get_theory()
    if z3.is_quantifier(ax)
    if ax.is_forall()
    if is_eq_or_leq(ax.body())
]
HINDLEY_IDS = ["".join(str(x).split()) for x in HINDLEY_AXIOMS]


@pytest.mark.parametrize("axiom", HINDLEY_AXIOMS, ids=HINDLEY_IDS)
def test_qe_hindley(axiom: z3.ExprRef) -> None:
    for expr in QEHindley(axiom):
        assert is_eq_or_leq(expr)
        assert not free_vars(expr)
        assert_weak(expr)
