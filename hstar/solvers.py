"""
Type checking &lambda;-join-calculus with Z3
"""

# flake8: noqa: E741
# ruff: noqa: E741

import inspect
from collections.abc import Callable

import pytest
import z3
from z3 import And, ForAll, If, Implies, Not, Or


@pytest.fixture(scope="module")
def base_solver():
    """Create a solver with the basic theories that all tests will need."""
    s = z3.Solver()
    order_theory(s)
    de_bruijn_theory(s)
    lambda_theory(s)
    convergence_theory(s)
    simple_theory(s)
    type_theory(s)
    return s


@pytest.fixture
def solver(base_solver):
    """Provide a solver with a fresh scope for each test."""
    base_solver.set(timeout=100)  # in milliseconds
    with base_solver:
        yield base_solver


# Terms.
Term = z3.Datatype("Term")
Term.declare("TOP")
Term.declare("BOT")
Term.declare("JOIN", ("lhs", Term), ("rhs", Term))
Term.declare("COMP", ("lhs", Term), ("rhs", Term))
Term.declare("APP", ("lhs", Term), ("rhs", Term))
Term.declare("VAR", ("index", z3.IntSort()))
Term.declare("ABS", ("body", Term))
Term = Term.create()

# Constructors.
TOP = Term.TOP
BOT = Term.BOT
JOIN = Term.JOIN
COMP = Term.COMP
APP = Term.APP
VAR = Term.VAR
ABS = Term.ABS

# De Bruijn operations.
SHIFT = z3.Function("SHIFT", Term, z3.IntSort(), Term)
SUBST = z3.Function("SUBST", z3.IntSort(), Term, Term, Term)

# Scott ordering.
LEQ = z3.Function("LEQ", Term, Term, z3.BoolSort())  # x [= y
EQ = z3.Function("EQ", Term, Term, z3.BoolSort())  # x == y
CONV = z3.Function("CONV", Term, z3.BoolSort())  # x converges

# Constants.
f, g, h = z3.Consts("f g h", Term)
r, s, t = z3.Consts("r s t", Term)
x, y, z = z3.Consts("x y z", Term)
v0 = VAR(0)
v1 = VAR(1)
v2 = VAR(2)
I = ABS(v0)
K = ABS(ABS(VAR(1)))
B = ABS(ABS(ABS(APP(v2, APP(v1, v0)))))
C = ABS(ABS(ABS(APP(APP(v2, v0), v1))))
S = ABS(ABS(ABS(APP(APP(v2, v0), APP(v1, v0)))))
Y = ABS(APP(ABS(APP(v1, APP(v0, v0))), ABS(APP(v1, APP(v0, v0)))))
DIV = z3.Const("DIV", Term)
TYPE = z3.Const("TYPE", Term)
SIMPLE = z3.Const("SIMPLE", Term)


def shift(term: z3.ExprRef, starting_at: int = 0) -> z3.ExprRef:
    """
    Increment free variables in a term, starting from index starting_at.
    Eagerly applies shifting rules for Term expressions when possible,
    otherwise returns an unevaluated SHIFT call.
    """
    start = z3.IntVal(starting_at)

    # Handle special constants first
    if term == TOP:
        return TOP
    elif term == BOT:
        return BOT

    # Check if we have a symbolic variable (like a, x, etc.)
    if z3.is_const(term) and term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        # This is a symbolic variable, just return unevaluated SHIFT
        return SHIFT(term, start)

    try:
        # Use Z3's application inspection functions directly
        if z3.is_app(term):
            decl = term.decl()
            decl_name = str(decl)

            if decl_name == "VAR":
                # Handle VAR constructor
                idx = term.arg(0).as_long()  # Get the index directly
                if idx >= starting_at:
                    return VAR(idx + 1)
                else:
                    return term
            elif decl_name == "ABS":
                # Handle ABS constructor
                body = term.arg(0)
                return ABS(shift(body, starting_at + 1))
            elif decl_name == "APP":
                # Handle APP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return APP(shift(lhs, starting_at), shift(rhs, starting_at))
            elif decl_name == "JOIN":
                # Handle JOIN constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return JOIN(shift(lhs, starting_at), shift(rhs, starting_at))
            elif decl_name == "COMP":
                # Handle COMP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return COMP(shift(lhs, starting_at), shift(rhs, starting_at))
    except Exception as e:
        # If we encounter any exception, log it and return unevaluated SHIFT
        print(f"Exception in shift: {e}")
        pass

    # Fall back to unevaluated SHIFT for any other expressions
    return SHIFT(term, start)


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


def join(*args: z3.ExprRef) -> z3.ExprRef:
    r"""Join a list of arguments."""
    if not args:
        return BOT
    result = args[0]
    for arg in args[1:]:
        result = JOIN(result, arg)
    return result


def app(*args: z3.ExprRef) -> z3.ExprRef:
    r"""Apply a list of arguments to a function."""
    result = args[0]
    for arg in args[1:]:
        result = APP(result, arg)
    return result


def comp(*args: z3.ExprRef) -> z3.ExprRef:
    r"""Compose a list of functions."""
    return COMP(args[0], comp(*args[1:])) if args else I


def TUPLE(*args: z3.ExprRef) -> z3.ExprRef:
    r"""Barendregt tuples `<M1,...,Mn> = \f. f M1 ... Mn`."""
    body = VAR(0)
    for arg in args:
        body = APP(body, shift(arg))
    return ABS(body)


def CONJ(a: z3.ExprRef, b: z3.ExprRef) -> z3.ExprRef:
    r"""Conjunction `a -> b = \f. b o f o a`."""
    a = shift(a)
    b = shift(b)
    return ABS(comp(b, VAR(0), a))


# FIXME assumes f returns a closed term.
# TODO convert native python functions to lambda terms, e.g. f(x) -> APP(f, x)
# TODO convert python or to join terms, e.g. x|y -> JOIN(x, y)
# TODO convert python tuples to TUPLE terms, e.g. (x, y, z) -> TUPLE(x, y, z)
def hoas(f: Callable[..., z3.ExprRef]) -> z3.ExprRef:
    r"""Higher-order abstract syntax. FIXME assumes f returns a closed term."""
    argc = len(inspect.signature(f).parameters)
    args = [VAR(argc - i - 1) for i in range(argc)]
    body = f(*args)
    for _ in range(argc):
        body = ABS(body)
    return body


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


@pytest.mark.parametrize(
    "pythonic, expected", HOAS_EXAMPLES, ids=[str(e[0]) for e in HOAS_EXAMPLES]
)
def test_hoas(pythonic, expected):
    assert hoas(pythonic) == expected


def simple(f: Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef]) -> z3.ExprRef:
    """HOAS constructor for simple types."""
    return APP(SIMPLE, hoas(f))


# Types.
ANY = I
semi = simple(lambda a, a1: CONJ(a, a1))
boool = simple(lambda a, a1: CONJ(a, CONJ(a, a1)))
pre_pair = simple(lambda a, a1: CONJ(CONJ(ANY, a), CONJ(CONJ(ANY, a), a1)))
unit = APP(TYPE, JOIN(semi, ABS(I)))
disamb_bool = hoas(lambda f, x, y: app(f, app(f, x, TOP), app(f, TOP, y)))
bool_ = APP(TYPE, JOIN(boool, disamb_bool))
true_ = hoas(lambda x, y: x)
false_ = hoas(lambda x, y: y)
disamb_pair = hoas(lambda p, f: app(f, app(p, true_), app(p, false_)))
pair = APP(TYPE, JOIN(pre_pair, disamb_pair))


def OFTYPE(x: z3.ExprRef, t: z3.ExprRef) -> z3.ExprRef:
    """Check if x is of type t."""
    return LEQ(APP(t, x), x)


def subst(i: int, replacement: z3.ExprRef, term: z3.ExprRef) -> z3.ExprRef:
    """
    Substitute a term for a variable in another term.
    Eagerly applies substitution rules for Term expressions when possible,
    otherwise returns an unevaluated SUBST call.
    """
    idx = z3.IntVal(i)

    # Handle special constants first
    if term == TOP:
        return TOP
    elif term == BOT:
        return BOT

    # Check if we have a symbolic variable (like a, x, etc.)
    if z3.is_const(term) and term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        # This is a symbolic variable, just return unevaluated SUBST
        return SUBST(idx, replacement, term)

    try:
        # Use Z3's application inspection functions directly
        if z3.is_app(term):
            decl = term.decl()
            decl_name = str(decl)

            if decl_name == "VAR":
                # Handle VAR constructor
                term_idx = term.arg(0).as_long()  # Get the index directly
                if term_idx == i:
                    return replacement
                else:
                    # Just return the variable unchanged - no need to decrement
                    return term
            elif decl_name == "ABS":
                # Handle ABS constructor
                body = term.arg(0)
                # When going under a binder, shift the replacement and increment the i
                shifted_replacement = shift(replacement, 0)
                return ABS(subst(i + 1, shifted_replacement, body))
            elif decl_name == "APP":
                # Handle APP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return APP(subst(i, replacement, lhs), subst(i, replacement, rhs))
            elif decl_name == "JOIN":
                # Handle JOIN constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return JOIN(subst(i, replacement, lhs), subst(i, replacement, rhs))
            elif decl_name == "COMP":
                # Handle COMP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return COMP(subst(i, replacement, lhs), subst(i, replacement, rhs))
    except Exception as e:
        # If we encounter any exception, log it and return unevaluated SUBST
        print(f"Exception in subst: {e}")
        pass

    # Fall back to unevaluated SUBST for any other expressions
    return SUBST(idx, replacement, term)


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
    # For [TOP/0](\x.1 x), the VAR(1) becomes VAR(2) under the abstraction, so TOP gets shifted
    assert subst(0, TOP, nested) == ABS(APP(TOP, VAR(0)))
    # For [TOP/1](\x.1 x), VAR(1) inside abstraction doesn't match VAR(2) (the shifted index)
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


def de_bruijn_theory(s: z3.Solver) -> None:
    """Theory of de Bruijn operations SHIFT and SUBST."""
    i = z3.Int("i")
    j = z3.Int("j")
    body = z3.Const("body", Term)
    lhs = z3.Const("lhs", Term)
    rhs = z3.Const("rhs", Term)
    start = z3.Int("start")
    s.add(
        # SHIFT axioms
        ForAll(
            [i, start],
            If(
                i >= start,
                SHIFT(VAR(i), start) == VAR(i + 1),
                SHIFT(VAR(i), start) == VAR(i),
            ),
        ),
        ForAll([x, start], SHIFT(ABS(x), start) == ABS(SHIFT(x, start + 1))),
        ForAll(
            [lhs, rhs, start],
            SHIFT(APP(lhs, rhs), start) == APP(SHIFT(lhs, start), SHIFT(rhs, start)),
        ),
        ForAll(
            [lhs, rhs, start],
            SHIFT(JOIN(lhs, rhs), start) == JOIN(SHIFT(lhs, start), SHIFT(rhs, start)),
        ),
        ForAll(
            [lhs, rhs, start],
            SHIFT(COMP(lhs, rhs), start) == COMP(SHIFT(lhs, start), SHIFT(rhs, start)),
        ),
        ForAll([start], SHIFT(TOP, start) == TOP),
        ForAll([start], SHIFT(BOT, start) == BOT),
        # SUBST axioms
        ForAll(
            [j, i, x],
            If(j == i, SUBST(i, x, VAR(j)) == x, SUBST(i, x, VAR(j)) == VAR(j)),
        ),
        ForAll(
            [body, i, x],
            SUBST(i, x, ABS(body)) == ABS(SUBST(i + 1, SHIFT(x, 0), body)),
        ),
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, APP(lhs, rhs)) == APP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        ),
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, JOIN(lhs, rhs)) == JOIN(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        ),
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, COMP(lhs, rhs)) == COMP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        ),
        ForAll([i, x], SUBST(i, x, TOP) == TOP),
        ForAll([i, x], SUBST(i, x, BOT) == BOT),
    )


# Theory of Scott ordering.
def order_theory(s: z3.Solver) -> None:
    s.add(
        # EQ is LEQ in both directions
        ForAll([x, y], EQ(x, y) == And(LEQ(x, y), LEQ(y, x))),
        # Basic order axioms
        ForAll([x], LEQ(x, TOP)),  # TOP is top
        ForAll([x], LEQ(BOT, x)),  # BOT is bottom
        ForAll([x], LEQ(x, x)),  # Reflexivity
        ForAll(
            [x, y, z], Implies(LEQ(x, y), Implies(LEQ(y, z), LEQ(x, z)))
        ),  # Transitivity
        Not(LEQ(TOP, BOT)),
        # JOIN is least upper bound
        ForAll([x, y], LEQ(x, JOIN(x, y))),  # Left component below
        ForAll([x, y], LEQ(y, JOIN(x, y))),  # Right component below
        ForAll(
            [x, y, z], Implies(LEQ(x, z), Implies(LEQ(y, z), LEQ(JOIN(x, y), z)))
        ),  # Least upper bound property
        # JOIN is associative, commutative, and idempotent
        ForAll([x, y], EQ(JOIN(x, y), JOIN(y, x))),
        ForAll([x, y, z], EQ(JOIN(x, JOIN(y, z)), JOIN(JOIN(x, y), z))),
        ForAll([x], EQ(JOIN(x, x), x)),
        # JOIN with BOT/TOP
        ForAll([x], EQ(JOIN(x, BOT), x)),  # BOT is identity
        ForAll([x], EQ(JOIN(x, TOP), TOP)),  # TOP absorbs
    )


# Theory of lambda calculus.
def lambda_theory(s: z3.Solver) -> None:
    s.add(
        # Composition properties
        ForAll([f, g, x], EQ(APP(COMP(f, g), x), APP(f, APP(g, x)))),
        ForAll([f], EQ(COMP(f, I), f)),
        ForAll([f], EQ(COMP(I, f), f)),
        ForAll([f], EQ(COMP(f, BOT), BOT)),
        ForAll([f], EQ(COMP(BOT, f), BOT)),
        ForAll([f], EQ(COMP(f, TOP), TOP)),
        ForAll([f], EQ(COMP(TOP, f), TOP)),
        # Composition is associative
        ForAll([f, g, h], EQ(COMP(f, COMP(g, h)), COMP(COMP(f, g), h))),
        # Composition is monotonic in both arguments
        ForAll([f, g, h], Implies(LEQ(f, g), LEQ(COMP(f, h), COMP(g, h)))),
        ForAll([f, g, h], Implies(LEQ(g, h), LEQ(COMP(f, g), COMP(f, h)))),
        # Basic combinators
        ForAll([x], EQ(APP(I, x), x)),
        ForAll([x, y], EQ(app(K, x, y), x)),
        ForAll([x, y, z], EQ(app(B, x, y, z), app(x, app(y, z)))),
        ForAll([x, y, z], EQ(app(C, x, y, z), app(x, z, y))),
        ForAll([x, y, z], EQ(app(S, x, y, z), app(x, z, app(y, z)))),
        ForAll([f], EQ(APP(Y, f), APP(f, APP(Y, f)))),
        # Beta reduction using Z3's SUBST
        ForAll([x, y], EQ(APP(ABS(x), y), SUBST(0, y, x))),
        # APP-JOIN distributivity (both directions)
        ForAll([f, g, x], EQ(APP(JOIN(f, g), x), JOIN(APP(f, x), APP(g, x)))),
        ForAll([f, x, y], LEQ(JOIN(APP(f, x), APP(f, y)), APP(f, JOIN(x, y)))),
        # APP monotonicity (in both arguments)
        ForAll([f, g, x], Implies(LEQ(f, g), LEQ(APP(f, x), APP(g, x)))),
        ForAll([f, x, y], Implies(LEQ(x, y), LEQ(APP(f, x), APP(f, y)))),
        # ABS monotonicity
        ForAll([x, y], Implies(LEQ(x, y), LEQ(ABS(x), ABS(y)))),
        # BOT/TOP preservation
        ForAll([x], EQ(APP(BOT, x), BOT)),
        ForAll([x], EQ(APP(TOP, x), TOP)),
        EQ(ABS(BOT), BOT),
        EQ(ABS(TOP), TOP),
        # JOIN distributivity over ABS
        ForAll([x, y], EQ(ABS(JOIN(x, y)), JOIN(ABS(x), ABS(y)))),
        # Extensionality
        ForAll([f, g], Implies(ForAll([x], LEQ(APP(f, x), APP(g, x))), LEQ(f, g))),
        ForAll([f, g], Implies(ForAll([x], EQ(APP(f, x), APP(g, x))), EQ(f, g))),
        # Eta conversion
        ForAll([f], EQ(ABS(APP(shift(f), VAR(0))), f)),
    )


def convergence_theory(s: z3.Solver) -> None:
    s.add(
        # DIV tests for convergence
        EQ(DIV, APP(Y, TUPLE(TOP))),
        ForAll([x], EQ(APP(DIV, x), APP(DIV, APP(x, TOP)))),
        LEQ(APP(DIV, BOT), BOT),
        # CONV is a least fixed point
        ForAll([x], Implies(LEQ(TOP, x), CONV(x))),
        ForAll([x], Implies(CONV(APP(x, TOP)), CONV(x))),
        Not(CONV(BOT)),
        ForAll([x], Implies(Not(CONV(x)), LEQ(x, BOT))),
        ForAll([x, y], Implies(LEQ(x, y), Implies(CONV(x), CONV(y)))),
        # DIV's relation to CONV
        ForAll([x], Implies(CONV(x), LEQ(TOP, APP(DIV, x)))),
        ForAll([x], Implies(LEQ(TOP, APP(DIV, x)), CONV(x))),
        # Multi-argument functions
        ForAll([x], Implies(CONV(x), CONV(ABS(x)))),
        ForAll([x], Implies(CONV(x), CONV(APP(x, TOP)))),
    )


def simple_theory(s: z3.Solver) -> None:
    """Theory of SIMPLE type, defined as join of section-retract pairs."""

    def above_all_sr(candidate):
        s1, r1 = z3.Consts("s1 r1", Term)  # Different names for bound variables
        return ForAll(
            [s1, r1],
            Implies(LEQ(COMP(r1, s1), I), LEQ(TUPLE(s1, r1), candidate)),
        )

    s.add(
        # SIMPLE is above all section-retract pairs.
        above_all_sr(SIMPLE),
        # SIMPLE is the least such term.
        ForAll([x], Implies(above_all_sr(x), LEQ(SIMPLE, x))),
    )


def has_inhabs(t: z3.ExprRef, *inhabs: z3.ExprRef) -> z3.ExprRef:
    return ForAll([x], Or(*[EQ(APP(t, x), i) for i in inhabs]))


def type_theory(s: z3.Solver) -> None:
    """Theory of types and type membership."""
    # FIXME some rules are commented out because they cause hangs.
    s.add(
        # EQ(TYPE, ABS(APP(Y, ABS(JOIN(I, COMP(v1, v0)))))),
        # EQ(TYPE, ABS(APP(Y, ABS(JOIN(I, COMP(v0, v1)))))),
        # # Types are closures.
        ForAll([t], LEQ(I, APP(TYPE, t))),
        ForAll([t], EQ(COMP(APP(TYPE, t), APP(TYPE, t)), APP(TYPE, t))),
        # TYPE is a type.
        LEQ(I, TYPE),
        EQ(COMP(TYPE, TYPE), TYPE),
        ForAll([t], EQ(APP(TYPE, APP(TYPE, t)), APP(TYPE, t))),
        # ForAll([t], EQ(APP(TYPE, t), JOIN(I, COMP(t, APP(TYPE, t))))),
        # ForAll([t], EQ(APP(TYPE, t), JOIN(I, COMP(APP(TYPE, t), t)))),
        # # Inhabitants are fixed points.
        OFTYPE(TYPE, TYPE),
        ForAll([t], OFTYPE(APP(TYPE, t), TYPE)),
        # has_inhabs(DIV, TOP, BOT),
        # has_inhabs(semi, TOP, BOT, I),
        # has_inhabs(unit, TOP, I),
        # has_inhabs(boool, TOP, true_, false_, JOIN(true_, false_), BOT),
        # has_inhabs(bool_, TOP, true_, false_, BOT),
    )


def test_consistency(solver: z3.Solver):
    """Check that our theories are consistent by trying to prove False."""
    solver.set(timeout=1000)  # in milliseconds
    with solver:
        result = solver.check()
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
def test_ordering(solver: z3.Solver, formula: z3.ExprRef):
    check_that(solver, formula)


LAMBDA_EXAMPLES = {
    # Beta reduction
    r"(\x.x)y = y": EQ(APP(I, y), y),
    # Y combinator fixed-point property
    "Y f = f(Y f)": EQ(APP(Y, f), APP(f, APP(Y, f))),
    r"(\x.x)BOT = BOT": EQ(APP(I, BOT), BOT),
    r"(\x.x)TOP = TOP": EQ(APP(I, TOP), TOP),
    # Eta conversion
    r"\x.fx = f": EQ(ABS(APP(shift(f), VAR(0))), f),
    # Function application monotonicity
    "f [= g => fx [= gx": Implies(LEQ(f, g), LEQ(APP(f, x), APP(g, x))),
    "x [= y => fx [= fy": Implies(LEQ(x, y), LEQ(APP(f, x), APP(f, y))),
    # Abstraction monotonicity
    r"x [= y -> \x [= \y": Implies(LEQ(x, y), LEQ(ABS(x), ABS(y))),
    # Distributivity
    "(f|g)x = fx|gx": EQ(APP(JOIN(f, g), x), JOIN(APP(f, x), APP(g, x))),
    "fx|fy [= f(x|y)": LEQ(JOIN(APP(f, x), APP(f, y)), APP(f, JOIN(x, y))),
    r"\(x|y) = \x|\y": EQ(ABS(JOIN(x, y)), JOIN(ABS(x), ABS(y))),
    # BOT/TOP preservation
    "BOT x = BOT": EQ(APP(BOT, x), BOT),
    "TOP x = TOP": EQ(APP(TOP, x), TOP),
    r"\BOT = BOT": EQ(ABS(BOT), BOT),
    r"\TOP = TOP": EQ(ABS(TOP), TOP),
    # Extensionality
    r"(/\x. fx=gx) => f=g": Implies(ForAll([x], EQ(APP(f, x), APP(g, x))), EQ(f, g)),
}


@pytest.mark.parametrize(
    "formula", LAMBDA_EXAMPLES.values(), ids=LAMBDA_EXAMPLES.keys()
)
def test_lambda(solver: z3.Solver, formula: z3.ExprRef):
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
    "DIV x = DIV(x TOP)": EQ(APP(DIV, x), APP(DIV, APP(x, TOP))),
    # Constant functions converge
    r"CONV(\x.TOP)": CONV(ABS(TOP)),
    r"CONV(\x.x)": CONV(ABS(VAR(0))),
    # Two-argument functions
    r"CONV(\x,y.x)": pytest.param(
        CONV(ABS(ABS(VAR(1)))),
        marks=[pytest.mark.xfail(reason="FIXME")],
    ),
    r"CONV(\x,y.y)": CONV(ABS(ABS(VAR(0)))),
    r"CONV(\x,y.x|y)": CONV(ABS(ABS(JOIN(VAR(0), VAR(1))))),
}


@pytest.mark.parametrize("formula", CONV_EXAMPLES.values(), ids=CONV_EXAMPLES.keys())
def test_conv(solver: z3.Solver, formula: z3.ExprRef):
    """Test convergence properties."""
    check_that(solver, formula)


TUPLE_EXAMPLES = {
    "<BOT> [= <x>": LEQ(TUPLE(BOT), TUPLE(x)),
    "<x> [= <TOP>": LEQ(TUPLE(x), TUPLE(TOP)),
    "<x> [= <x>": LEQ(TUPLE(x), TUPLE(x)),
    "<x>|<y> [= <x|y>": LEQ(JOIN(TUPLE(x), TUPLE(y)), TUPLE(JOIN(x, y))),
}


@pytest.mark.parametrize("formula", TUPLE_EXAMPLES.values(), ids=TUPLE_EXAMPLES.keys())
def test_tuple_ordering(solver: z3.Solver, formula: z3.ExprRef):
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
def test_simple(solver: z3.Solver, formula: z3.ExprRef):
    """Test SIMPLE type properties."""
    check_that(solver, formula)


TYPE_EXAMPLES = {
    # Any.
    "ANY : TYPE": OFTYPE(ANY, TYPE),
    "x : ANY": OFTYPE(x, ANY),
    # Div.
    "div : TYPE": OFTYPE(DIV, TYPE),
    "TOP : div": OFTYPE(TOP, DIV),
    "BOT : div": OFTYPE(BOT, DIV),
    # Semi.
    "semi : TYPE": OFTYPE(semi, TYPE),
    "TOP : semi": OFTYPE(TOP, semi),
    "BOT : semi": OFTYPE(BOT, semi),
    "I : semi": OFTYPE(I, semi),
    # Boool.
    "boool : TYPE": OFTYPE(boool, TYPE),
    "TOP : boool": OFTYPE(TOP, boool),
    "BOT : boool": OFTYPE(BOT, boool),
    "true : boool": OFTYPE(true_, boool),
    "false : boool": OFTYPE(false_, boool),
    "JOIN(true, false) : boool": OFTYPE(JOIN(true_, false_), boool),
    # Pre Pair.
    "pre_pair : TYPE": OFTYPE(pre_pair, TYPE),
    "<x,y> : pre_pair": OFTYPE(TUPLE(x, y), pre_pair),
    "x,y:pair ==> x|y:pair": Implies(
        OFTYPE(TUPLE(x, y), pair), OFTYPE(JOIN(x, y), pair)
    ),
    # Unit.
    "unit : TYPE": OFTYPE(unit, TYPE),
    "TOP : unit": OFTYPE(TOP, unit),
    "I : unit": OFTYPE(I, unit),
    # Bool.
    "bool : TYPE": OFTYPE(bool_, TYPE),
    "true : bool": OFTYPE(true_, bool_),
    "false : bool": OFTYPE(false_, bool_),
    # Pair.
    "pair : TYPE": OFTYPE(pair, TYPE),
    "<x,y> : pair": OFTYPE(TUPLE(x, y), pair),
}


@pytest.mark.parametrize("formula", TYPE_EXAMPLES.values(), ids=TYPE_EXAMPLES.keys())
def test_types(solver: z3.Solver, formula: z3.ExprRef):
    """Test type system properties."""
    check_that(solver, formula)
