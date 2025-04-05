import logging
from collections.abc import Iterator

import pytest
import z3
from z3 import And, ForAll, Implies, Not

from hstar.language import (
    ANY,
    APP,
    BOT,
    CI,
    COMP,
    CONV,
    DIV,
    INHABITS,
    JOIN,
    KI,
    LEQ,
    TOP,
    TUPLE,
    VAR,
    A,
    B,
    C,
    I,
    J,
    K,
    S,
    Term,
    V,
    Y,
    bool_,
    boool,
    box,
    lam,
    pair,
    pre_box,
    pre_pair,
    semi,
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
    solver = z3.Solver()
    # Enable unsat core generation
    solver.set("unsat_core", True)
    solver.set("proof", True)
    add_theory(solver)
    return solver


@pytest.fixture
def solver(base_solver: z3.Solver) -> Iterator[z3.Solver]:
    """Provide a solver with a fresh scope for each test."""
    with base_solver, solver_timeout(base_solver, timeout_ms=100):
        yield base_solver


def test_unsat_core() -> None:
    """Check that our theories are consistent by trying to prove False."""
    solver = z3.Solver()
    solver.set("unsat_core", True)
    x = z3.Const("x", z3.IntSort())
    solver.assert_and_track(x < x, "inferiority")
    result = solver.check()
    if result == z3.unsat:
        core = solver.unsat_core()
        logger.info(f"Unsat core:\n{core}")
    assert result == z3.unsat


def test_consistency(solver: z3.Solver) -> None:
    """Check that our theories are consistent by trying to prove False."""
    with solver, solver_timeout(solver, timeout_ms=1000):
        result = solver.check()
        if result == z3.unsat:
            core = solver.unsat_core()
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
    "Y o J = I": COMP(Y, J) == I,
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
    r"\x.fx = f": lam(v0, APP(f, v0)) == f,
    # Function application monotonicity
    "f [= g => fx [= gx": Implies(LEQ(f, g), LEQ(APP(f, x), APP(g, x))),
    "x [= y => fx [= fy": Implies(LEQ(x, y), LEQ(APP(f, x), APP(f, y))),
    # Distributivity
    "(f|g)x = fx|gx": APP(JOIN(f, g), x) == JOIN(APP(f, x), APP(g, x)),
    "fx|fy [= f(x|y)": LEQ(JOIN(APP(f, x), APP(f, y)), APP(f, JOIN(x, y))),
    # BOT/TOP preservation
    "BOT x = BOT": APP(BOT, x) == BOT,
    "TOP x = TOP": APP(TOP, x) == TOP,
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
    r"CONV(\x.TOP)": CONV(APP(K, TOP)),
    r"CONV(\x.I)": CONV(APP(K, I)),
    # Two-argument functions
    r"CONV(\x,y.x)": CONV(K),
    r"CONV(\x,y.y)": CONV(KI),
    r"CONV(\x,y.x|y)": CONV(J),
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


# Add tests for properties of A
SIMPLE_EXAMPLES = {
    "A [= TOP": LEQ(A, TOP),
    "BOT [= A": LEQ(BOT, A),
    "<I,I> [= A": LEQ(TUPLE(I, I), A),
    r"<\f.\x.f, \f.f TOP> [= A": LEQ(TUPLE(K, APP(CI, TOP)), A),
}


@pytest.mark.parametrize(
    "formula", SIMPLE_EXAMPLES.values(), ids=SIMPLE_EXAMPLES.keys()
)
def test_simple(solver: z3.Solver, formula: z3.ExprRef) -> None:
    """Test simple type properties."""
    check_that(solver, formula)


TYPE_EXAMPLES = {
    # Any.
    "ANY : TYPE": INHABITS(ANY, V),
    "x : ANY": INHABITS(x, ANY),
    # Div.
    "div : TYPE": INHABITS(DIV, V),
    "TOP : div": INHABITS(TOP, DIV),
    "BOT : div": INHABITS(BOT, DIV),
    # Semi.
    "semi : TYPE": INHABITS(semi, V),
    "TOP : semi": INHABITS(TOP, semi),
    "BOT : semi": INHABITS(BOT, semi),
    "I : semi": INHABITS(I, semi),
    # Boool.
    "boool : TYPE": INHABITS(boool, V),
    "TOP : boool": INHABITS(TOP, boool),
    "BOT : boool": INHABITS(BOT, boool),
    "true : boool": INHABITS(K, boool),
    "false : boool": INHABITS(KI, boool),
    "J : boool": INHABITS(J, boool),
    "x,y : boool ==> x|y : boool": Implies(
        And(INHABITS(x, boool), INHABITS(y, boool)),
        INHABITS(JOIN(x, y), boool),
    ),
    # Unit.
    "unit : TYPE": INHABITS(unit, V),
    "TOP : unit": INHABITS(TOP, unit),
    "BOT : unit": INHABITS(BOT, unit),
    "I : unit": INHABITS(I, unit),
    # Bool.
    "bool : TYPE": INHABITS(bool_, V),
    "TOP : bool": INHABITS(TOP, bool_),
    "BOT : bool": INHABITS(BOT, bool_),
    "true : bool": INHABITS(K, bool_),
    "false : bool": INHABITS(KI, bool_),
    # Pre Box.
    "pre_box : TYPE": INHABITS(pre_box, V),
    "TOP : pre_box": INHABITS(TOP, pre_box),
    "BOT : pre_box": INHABITS(BOT, pre_box),
    "<x> : pre_box": INHABITS(TUPLE(x), pre_box),
    "x,y : pre_box ==> x|y : pre_box": Implies(
        And(INHABITS(x, pre_box), INHABITS(y, pre_box)),
        INHABITS(JOIN(x, y), pre_box),
    ),
    # Box.
    "box : TYPE": INHABITS(box, V),
    "TOP : box": INHABITS(TOP, box),
    "<x> : box": INHABITS(TUPLE(x), box),
    # Pre Pair.
    "pre_pair : TYPE": INHABITS(pre_pair, V),
    "TOP : pre_pair": INHABITS(TOP, pre_pair),
    "BOT : pre_pair": INHABITS(BOT, pre_pair),
    "<x,y> : pre_pair": INHABITS(TUPLE(x, y), pre_pair),
    "x,y : pre_pair ==> x|y : pre_pair": Implies(
        And(INHABITS(x, pre_pair), INHABITS(y, pre_pair)),
        INHABITS(JOIN(x, y), pre_pair),
    ),
    # Pair.
    "pair : TYPE": INHABITS(pair, V),
    "TOP : pair": INHABITS(TOP, pair),
    "<x,y> : pair": INHABITS(TUPLE(x, y), pair),
}


@pytest.mark.parametrize("formula", TYPE_EXAMPLES.values(), ids=TYPE_EXAMPLES.keys())
def test_types(solver: z3.Solver, formula: z3.ExprRef) -> None:
    """Test type system properties."""
    check_that(solver, formula)


def test_find_counterexample_1(solver: z3.Solver) -> None:
    # Case 1: Valid formula - should return (True, None)
    formula = ForAll([x], LEQ(x, x))  # x [= x is always true
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


def test_hand_partial_order() -> None:
    T = z3.DeclareSort("T")
    Less = z3.Function("Less", T, T, z3.BoolSort())
    Bot, x, y, z = z3.Consts("Bot x y z", T)
    solver = z3.Solver()

    # Reflexivity, antisymmetry, transitivity
    solver.add(ForAll([x], Less(x, x)))
    solver.add(ForAll([x, y], Implies(Less(x, y), Implies(Less(y, x), x == y))))
    solver.add(ForAll([x, y, z], Implies(Less(x, y), Implies(Less(y, z), Less(x, z)))))

    solver.add(ForAll([x], Less(Bot, x)))
    assert solver.check(Less(Bot, x)) == z3.sat


@pytest.mark.xfail(reason="https://github.com/Z3Prover/z3/issues/7592", run=False)
def test_builtin_partial_order() -> None:
    T = z3.DeclareSort("T")
    Less = z3.PartialOrder(T, 0)
    Bot, x = z3.Consts("Bot x", T)
    solver = z3.Solver()
    solver.add(ForAll([x], Less(Bot, x)))
    assert solver.check(Less(Bot, x)) == z3.sat
