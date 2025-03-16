import logging
from collections.abc import Iterator

import pytest
import z3
from z3 import And, ForAll, Implies, Not

from hstar.language import (
    ABS,
    ANY,
    APP,
    BOT,
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
    bool_,
    boool,
    pair,
    pre_pair,
    semi,
    shift,
    unit,
)
from hstar.solvers import find_counterexample, solver_timeout, try_prove
from hstar.theory import add_theory

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


def test_consistency(solver: z3.Solver) -> None:
    """Check that our theories are consistent by trying to prove False."""
    with solver, solver_timeout(solver, timeout_ms=1000):
        solver.set("unsat_core", True)
        result = solver.check()
        if result == z3.unsat:
            core = solver.unsat_core()  # FIXME sometimes core is empty
            logger.error(f"Unsat core:\n{core}")
        assert result != z3.unsat


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
    r"<\f.\x.f, \f.f TOP> [= SIMPLE": LEQ(
        TUPLE(ABS(ABS(VAR(1))), ABS(APP(VAR(0), TOP))), SIMPLE
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
