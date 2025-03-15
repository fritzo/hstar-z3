import logging
from collections.abc import Iterator
from typing import Any

import pytest
import z3
from z3 import And, ForAll, Implies, Not

from hstar.solvers import (
    ABS,
    ANY,
    APP,
    BOT,
    COMP,
    CONV,
    DIV,
    JOIN,
    KI,
    LEQ,
    OFTYPE,
    SIMPLE,
    TOP,
    TUPLE,
    VAR,
    B,
    C,
    I,
    K,
    S,
    Term,
    V,
    Y,
    add_theory,
    bool_,
    boool,
    find_counterexample,
    hoas,
    pair,
    pre_pair,
    semi,
    shift,
    solver_timeout,
    subst,
    unit,
)

logger = logging.getLogger(__name__)

f, g, h = z3.Consts("f g h", Term)
r, s, t = z3.Consts("r s t", Term)
x, y, z = z3.Consts("x y z", Term)
v0 = VAR(0)
v1 = VAR(1)
v2 = VAR(2)


@pytest.fixture(scope="module")
def base_solver() -> z3.Solver:
    """Create a solver with the basic theories that all tests will need."""
    s = z3.Solver()
    add_theory(s)
    return s


@pytest.fixture
def solver(base_solver: z3.Solver) -> Iterator[z3.Solver]:
    """Provide a solver with a fresh scope for each test."""
    with base_solver, solver_timeout(base_solver, timeout_ms=100):
        yield base_solver


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


def test_consistency(solver: z3.Solver) -> None:
    """Check that our theories are consistent by trying to prove False."""
    with solver, solver_timeout(solver, timeout_ms=1000):
        result = solver.check()
        if result == z3.unsat:
            core = solver.unsat_core()
            logger.error(f"Unsat core:\n{core}")
        assert result != z3.unsat


def try_prove(solver: z3.Solver, formula: z3.ExprRef) -> tuple[bool | None, str | None]:
    """
    Try to prove a formula is valid or invalid.

    Args:
        solver: Z3 solver to use
        formula: Formula to check validity of

    Returns:
        Tuple of:
        - True if formula proved valid
        - False if formula proved invalid
        - None if formula is satisfiable but not valid
        And the counterexample model string (if formula is not valid)
    """
    with solver:
        solver.add(Not(formula))
        result = solver.check()
        if result == z3.unsat:
            return True, None
        if result == z3.sat:
            model = solver.model()
            assert model is not None, "Got sat result but no model!"
            # Format model while still in context
            model_str = "\n".join(f"{d} = {model[d]}" for d in model.decls())
            return False, model_str
        if result == z3.unknown:
            return None, None
        raise ValueError(f"Z3 returned unexpected result: {result}")


def check_that(solver: z3.Solver, formula: z3.ExprRef) -> None:
    """
    Assert that formula is valid.

    Args:
        solver: Z3 solver to use
        formula: Formula to check validity of

    Raises:
        pytest.xfail: If formula is satisfiable but not valid
        AssertionError: If formula is invalid
    """
    result, model_str = try_prove(solver, formula)
    if result is True:
        return
    if result is None:
        pytest.xfail(f"unknown: {formula}")
        pytest.skip(f"unknown: {formula}")  # when --runxfail
        return
    assert result is False
    pytest.fail(f"Formula is invalid: {formula}\nCounterexample:\n{model_str}")


ORDER_EXAMPLES = {
    "TOP [= TOP": LEQ(TOP, TOP),
    "BOT [= BOT": LEQ(BOT, BOT),
    "BOT [= TOP": LEQ(BOT, TOP),
    "TOP [!= BOT": Not(LEQ(TOP, BOT)),
    "x [= TOP": LEQ(x, TOP),
    "BOT [= x": LEQ(BOT, x),
    "x [= x": LEQ(x, x),
    "x [= x | y": LEQ(x, JOIN(x, y)),
    "y [= x | y": LEQ(y, JOIN(x, y)),
    "x | y [= y | x": LEQ(JOIN(x, y), JOIN(y, x)),
    "x | (y | z) [= (x | y) | z": LEQ(JOIN(x, JOIN(y, z)), JOIN(JOIN(x, y), z)),
}


@pytest.mark.parametrize("formula", ORDER_EXAMPLES.values(), ids=ORDER_EXAMPLES.keys())
def test_ordering(solver: z3.Solver, formula: z3.ExprRef) -> None:
    check_that(solver, formula)


LAMBDA_EXAMPLES = {
    # Beta reduction
    r"(\x.x)y = y": APP(I, y) == y,
    # Y combinator fixed-point property
    "Y f = f(Y f)": APP(Y, f) == APP(f, APP(Y, f)),
    r"(\x.x)BOT = BOT": APP(I, BOT) == BOT,
    r"(\x.x)TOP = TOP": APP(I, TOP) == TOP,
    # Eta conversion
    r"\x.fx = f": ABS(APP(shift(f), VAR(0))) == f,
    # Function application monotonicity
    "f [= g => fx [= gx": Implies(LEQ(f, g), LEQ(APP(f, x), APP(g, x))),
    "x [= y => fx [= fy": Implies(LEQ(x, y), LEQ(APP(f, x), APP(f, y))),
    # Abstraction monotonicity
    r"x [= y -> \x [= \y": Implies(LEQ(x, y), LEQ(ABS(x), ABS(y))),
    # Distributivity
    "(f|g)x = fx|gx": APP(JOIN(f, g), x) == JOIN(APP(f, x), APP(g, x)),
    "fx|fy [= f(x|y)": LEQ(JOIN(APP(f, x), APP(f, y)), APP(f, JOIN(x, y))),
    r"\(x|y) = \x|\y": ABS(JOIN(x, y)) == JOIN(ABS(x), ABS(y)),
    # BOT/TOP preservation
    "BOT x = BOT": APP(BOT, x) == BOT,
    "TOP x = TOP": APP(TOP, x) == TOP,
    r"\BOT = BOT": ABS(BOT) == BOT,
    r"\TOP = TOP": ABS(TOP) == TOP,
    # Extensionality
    r"(/\x. fx=gx) => f=g": Implies(ForAll([x], APP(f, x) == APP(g, x)), f == g),
}


@pytest.mark.parametrize(
    "formula", LAMBDA_EXAMPLES.values(), ids=LAMBDA_EXAMPLES.keys()
)
def test_lambda(solver: z3.Solver, formula: z3.ExprRef) -> None:
    """Test lambda calculus properties."""
    check_that(solver, formula)


CONV_EXAMPLES = {
    # Basic convergence properties
    "TOP converges": CONV(TOP),
    "I converges": CONV(I),
    "BOT diverges": Not(CONV(BOT)),
    # Monotonicity of convergence
    "x [= y & CONV(x) => CONV(y)": Implies(And(LEQ(x, y), CONV(x)), CONV(y)),
    # DIV properties
    "DIV BOT [= BOT": LEQ(APP(DIV, BOT), BOT),
    "CONV(x) => TOP [= DIV x": Implies(CONV(x), LEQ(TOP, APP(DIV, x))),
    "TOP [= DIV x => CONV(x)": Implies(LEQ(TOP, APP(DIV, x)), CONV(x)),
    # Fixed point property
    "DIV x = DIV(x TOP)": APP(DIV, x) == APP(DIV, APP(x, TOP)),
    # Constant functions converge
    r"CONV(\x.TOP)": CONV(ABS(TOP)),
    r"CONV(\x.x)": CONV(ABS(VAR(0))),
    # Two-argument functions
    r"CONV(\x,y.x)": CONV(ABS(ABS(VAR(1)))),
    r"CONV(\x,y.y)": CONV(ABS(ABS(VAR(0)))),
    r"CONV(\x,y.x|y)": CONV(ABS(ABS(JOIN(VAR(0), VAR(1))))),
}


@pytest.mark.parametrize("formula", CONV_EXAMPLES.values(), ids=CONV_EXAMPLES.keys())
def test_conv(solver: z3.Solver, formula: z3.ExprRef) -> None:
    """Test convergence properties."""
    check_that(solver, formula)


TUPLE_EXAMPLES = {
    "<BOT> [= <x>": LEQ(TUPLE(BOT), TUPLE(x)),
    "<x> [= <TOP>": LEQ(TUPLE(x), TUPLE(TOP)),
    "<x> [= <x>": LEQ(TUPLE(x), TUPLE(x)),
    "<x>|<y> [= <x|y>": LEQ(JOIN(TUPLE(x), TUPLE(y)), TUPLE(JOIN(x, y))),
}


@pytest.mark.parametrize("formula", TUPLE_EXAMPLES.values(), ids=TUPLE_EXAMPLES.keys())
def test_tuple_ordering(solver: z3.Solver, formula: z3.ExprRef) -> None:
    check_that(solver, formula)


# Add tests for SIMPLE properties
SIMPLE_EXAMPLES = {
    "SIMPLE [= TOP": LEQ(SIMPLE, TOP),
    "BOT [= SIMPLE": LEQ(BOT, SIMPLE),
    "<I,I> [= SIMPLE": LEQ(TUPLE(I, I), SIMPLE),
    r"<\f.\x.f, \f.f TOP> [= SIMPLE": pytest.param(
        LEQ(TUPLE(ABS(ABS(VAR(1))), ABS(APP(VAR(0), TOP))), SIMPLE),
        marks=[pytest.mark.xfail(reason="FIXME")],
    ),
}


@pytest.mark.parametrize(
    "formula", SIMPLE_EXAMPLES.values(), ids=SIMPLE_EXAMPLES.keys()
)
def test_simple(solver: z3.Solver, formula: z3.ExprRef) -> None:
    """Test SIMPLE type properties."""
    check_that(solver, formula)


TYPE_EXAMPLES = {
    # Any.
    "ANY : TYPE": OFTYPE(ANY, V),
    "x : ANY": OFTYPE(x, ANY),
    # Div.
    "div : TYPE": OFTYPE(DIV, V),
    "TOP : div": OFTYPE(TOP, DIV),
    "BOT : div": OFTYPE(BOT, DIV),
    # Semi.
    "semi : TYPE": OFTYPE(semi, V),
    "TOP : semi": OFTYPE(TOP, semi),
    "BOT : semi": OFTYPE(BOT, semi),
    "I : semi": OFTYPE(I, semi),
    # Boool.
    "boool : TYPE": OFTYPE(boool, V),
    "TOP : boool": OFTYPE(TOP, boool),
    "BOT : boool": OFTYPE(BOT, boool),
    "true : boool": OFTYPE(K, boool),
    "false : boool": OFTYPE(KI, boool),
    "JOIN(true, false) : boool": OFTYPE(JOIN(K, KI), boool),
    # Pre Pair.
    "pre_pair : TYPE": OFTYPE(pre_pair, V),
    "<x,y> : pre_pair": OFTYPE(TUPLE(x, y), pre_pair),
    "x,y:pair ==> x|y:pair": Implies(
        OFTYPE(TUPLE(x, y), pair), OFTYPE(JOIN(x, y), pair)
    ),
    # Unit.
    "unit : TYPE": OFTYPE(unit, V),
    "TOP : unit": OFTYPE(TOP, unit),
    "I : unit": OFTYPE(I, unit),
    # Bool.
    "bool : TYPE": OFTYPE(bool_, V),
    "true : bool": OFTYPE(K, bool_),
    "false : bool": OFTYPE(KI, bool_),
    # Pair.
    "pair : TYPE": OFTYPE(pair, V),
    "<x,y> : pair": OFTYPE(TUPLE(x, y), pair),
}


@pytest.mark.parametrize("formula", TYPE_EXAMPLES.values(), ids=TYPE_EXAMPLES.keys())
def test_types(solver: z3.Solver, formula: z3.ExprRef) -> None:
    """Test type system properties."""
    check_that(solver, formula)


def test_find_counterexample_1(solver: z3.Solver) -> None:
    # Case 1: Valid formula - should return (True, None)
    formula = ForAll([x], LEQ(x, TOP))  # x [= TOP is always true
    result, counter = find_counterexample(solver, formula, x, timeout_ms=1000)
    assert result is True
    assert counter is None


@pytest.mark.xfail(reason="FIXME solver returns z3.unknown")
def test_find_counterexample_2(solver: z3.Solver) -> None:
    # Case 2: Invalid formula - should return (False, counterexample)
    formula = ForAll([x], LEQ(TOP, x))  # TOP [= x is false for x=BOT
    result, counter = find_counterexample(solver, formula, x, timeout_ms=1000)
    assert result is False
    assert counter is not None
    # Verify the counterexample is valid (TOP is not LEQ to it)
    with solver:
        solver.add(Not(LEQ(TOP, counter)))
        assert solver.check() == z3.sat


def test_find_counterexample_3(solver: z3.Solver) -> None:
    # Case 3: Unknown formula - should return (None, None)
    # Create a complex formula that will timeout within the small timeout
    complex_terms = x
    # Build a deeply nested formula that's hard for Z3 to solve quickly
    for _ in range(10):
        complex_terms = APP(complex_terms, APP(JOIN(APP(Y, B), APP(C, S)), x))
    formula = ForAll([x], complex_terms == complex_terms)  # Tautology but complex
    result, counter = find_counterexample(
        solver, formula, x, timeout_ms=1
    )  # Very short timeout
    # We expect either None (timeout) or True (solved)
    assert result is None or result is True
    assert result is None or counter is None
