"""
# Proving properties of λ-join-calculus expressions with Z3.

This uses a de Bruijn indexed representation of λ-join-calculus terms, with a
`LEQ` relation for the Scott ordering, and explicit `BOT` (bottom), `TOP` (top),
and binary `JOIN` operation wrt the Scott ordering.

The theory includes de Bruijn syntax, Scott ordering, lambda calculus, and
a types-as-closures.
"""

import inspect
from collections.abc import Callable, Generator
from contextlib import contextmanager

import z3
from z3 import And, ForAll, If, Implies, Not, Or

from .metrics import COUNTERS

counter = COUNTERS[__name__]

# Terms.
Term = z3.Datatype("Term")
Term.declare("TOP")
Term.declare("BOT")
Term.declare("JOIN", ("join_lhs", Term), ("join_rhs", Term))
Term.declare("COMP", ("comp_lhs", Term), ("comp_rhs", Term))  # TODO remove?
Term.declare("APP", ("app_lhs", Term), ("app_rhs", Term))
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


def shift(term: z3.ExprRef, start: int = 0) -> z3.ExprRef:
    """
    Increment free variables in a term, starting from index `start`.
    Eagerly applies shifting rules for Term expressions when possible,
    otherwise returns an unevaluated SHIFT call.
    """
    # Handle special constants first
    if term == TOP:
        return TOP
    elif term == BOT:
        return BOT

    # Check if we have a symbolic variable (like a, x, etc.)
    if z3.is_const(term) and term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        # This is a symbolic variable, just return unevaluated SHIFT
        return SHIFT(term, z3.IntVal(start))

    try:
        # Use Z3's application inspection functions directly
        if z3.is_app(term):
            decl = term.decl()
            decl_name = str(decl)

            if decl_name == "VAR":
                # Handle VAR constructor
                idx = term.arg(0).as_long()  # Get the index directly
                if idx >= start:
                    return VAR(idx + 1)
                else:
                    return term
            elif decl_name == "ABS":
                # Handle ABS constructor
                body = term.arg(0)
                return ABS(shift(body, start + 1))
            elif decl_name == "APP":
                # Handle APP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return APP(shift(lhs, start), shift(rhs, start))
            elif decl_name == "JOIN":
                # Handle JOIN constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return JOIN(shift(lhs, start), shift(rhs, start))
            elif decl_name == "COMP":
                # Handle COMP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return COMP(shift(lhs, start), shift(rhs, start))
    except Exception as e:
        # If we encounter any exception, log it and return unevaluated SHIFT
        print(f"Exception in shift: {e}")
        pass

    # Fall back to unevaluated SHIFT for any other expressions
    return SHIFT(term, z3.IntVal(start))


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


class Adder:
    def __init__(self, solver: z3.Solver, groupname: str) -> None:
        self.solver = solver
        self.groupname = groupname

    def __call__(self, expr: z3.ExprRef, name: str) -> None:
        self.solver.assert_and_track(expr, f"{self.groupname}_{name}")


def de_bruijn_theory(s: z3.Solver) -> None:
    """Theory of de Bruijn operations SHIFT and SUBST."""
    add = Adder(s, "de_bruijn")
    i = z3.Int("i")
    j = z3.Int("j")
    body = z3.Const("body", Term)
    lhs = z3.Const("lhs", Term)
    rhs = z3.Const("rhs", Term)
    start = z3.Int("start")
    # SHIFT axioms
    add(
        ForAll([i, start], SHIFT(VAR(i), start) == VAR(If(i >= start, i + 1, i))),
        "shift_var",
    )
    add(
        ForAll([x, start], SHIFT(ABS(x), start) == ABS(SHIFT(x, start + 1))),
        "shift_abs",
    )
    add(
        ForAll(
            [lhs, rhs, start],
            SHIFT(APP(lhs, rhs), start) == APP(SHIFT(lhs, start), SHIFT(rhs, start)),
        ),
        "shift_app",
    )
    add(
        ForAll(
            [lhs, rhs, start],
            SHIFT(JOIN(lhs, rhs), start) == JOIN(SHIFT(lhs, start), SHIFT(rhs, start)),
        ),
        "shift_join",
    )
    add(
        ForAll(
            [lhs, rhs, start],
            SHIFT(COMP(lhs, rhs), start) == COMP(SHIFT(lhs, start), SHIFT(rhs, start)),
        ),
        "shift_comp",
    )
    add(ForAll([start], SHIFT(TOP, start) == TOP), "shift_top")
    add(ForAll([start], SHIFT(BOT, start) == BOT), "shift_bot")
    # SUBST axioms
    add(
        ForAll(
            [j, i, x],
            If(j == i, SUBST(i, x, VAR(j)) == x, SUBST(i, x, VAR(j)) == VAR(j)),
        ),
        "subst_var",
    )
    add(
        ForAll(
            [body, i, x],
            SUBST(i, x, ABS(body)) == ABS(SUBST(i + 1, SHIFT(x, 0), body)),
        ),
        "subst_abs",
    )
    add(
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, APP(lhs, rhs)) == APP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        ),
        "subst_app",
    )
    add(
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, JOIN(lhs, rhs)) == JOIN(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        ),
        "subst_join",
    )
    add(
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, COMP(lhs, rhs)) == COMP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        ),
        "subst_comp",
    )
    add(ForAll([i, x], SUBST(i, x, TOP) == TOP), "subst_top")
    add(ForAll([i, x], SUBST(i, x, BOT) == BOT), "subst_bot")


# Theory of Scott ordering.
def order_theory(s: z3.Solver) -> None:
    add = Adder(s, "order")

    # EQ is LEQ in both directions
    add(ForAll([x, y], EQ(x, y) == And(LEQ(x, y), LEQ(y, x))), "eq_def")

    # Basic order axioms
    add(ForAll([x], LEQ(x, TOP)), "top_upper")
    add(ForAll([x], LEQ(BOT, x)), "bot_lower")
    add(ForAll([x], LEQ(x, x)), "leq_reflexive")
    add(
        ForAll([x, y, z], Implies(LEQ(x, y), Implies(LEQ(y, z), LEQ(x, z)))),
        "transitivity",
    )
    add(Not(LEQ(TOP, BOT)), "top_bot")

    # JOIN is least upper bound
    add(ForAll([x, y], LEQ(x, JOIN(x, y))), "join_left_bound")
    add(ForAll([x, y], LEQ(y, JOIN(x, y))), "join_right_bound")
    add(
        ForAll([x, y, z], Implies(LEQ(x, z), Implies(LEQ(y, z), LEQ(JOIN(x, y), z)))),
        "join_least_upper",
    )

    # JOIN is associative, commutative, and idempotent
    add(ForAll([x, y], EQ(JOIN(x, y), JOIN(y, x))), "join_commutative")
    add(
        ForAll([x, y, z], EQ(JOIN(x, JOIN(y, z)), JOIN(JOIN(x, y), z))),
        "join_associative",
    )
    add(ForAll([x], EQ(JOIN(x, x), x)), "join_idempotent")

    # JOIN with BOT/TOP
    add(ForAll([x], EQ(JOIN(x, BOT), x)), "join_bot")
    add(ForAll([x], EQ(JOIN(x, TOP), TOP)), "join_top")


# Theory of lambda calculus.
def lambda_theory(s: z3.Solver) -> None:
    add = Adder(s, "lambda")

    # Composition properties
    add(
        ForAll([f, g, x], EQ(APP(COMP(f, g), x), APP(f, APP(g, x)))), "comp_application"
    )
    add(ForAll([f], EQ(COMP(f, I), f)), "comp_identity_right")
    add(ForAll([f], EQ(COMP(I, f), f)), "comp_identity_left")
    add(ForAll([f], EQ(COMP(f, BOT), BOT)), "comp_bot_right")
    add(ForAll([f], EQ(COMP(BOT, f), BOT)), "comp_bot_left")
    add(ForAll([f], EQ(COMP(f, TOP), TOP)), "comp_top_right")
    add(ForAll([f], EQ(COMP(TOP, f), TOP)), "comp_top_left")

    # Composition is associative
    add(
        ForAll([f, g, h], EQ(COMP(f, COMP(g, h)), COMP(COMP(f, g), h))),
        "comp_associative",
    )

    # Composition is monotonic in both arguments
    add(
        ForAll([f, g, h], Implies(LEQ(f, g), LEQ(COMP(f, h), COMP(g, h)))),
        "comp_monotonic_left",
    )
    add(
        ForAll([f, g, h], Implies(LEQ(g, h), LEQ(COMP(f, g), COMP(f, h)))),
        "comp_monotonic_right",
    )

    # Basic combinators
    add(ForAll([x], EQ(APP(I, x), x)), "combinator_I")
    add(ForAll([x, y], EQ(app(K, x, y), x)), "combinator_K")
    add(ForAll([x, y, z], EQ(app(B, x, y, z), app(x, app(y, z)))), "combinator_B")
    add(ForAll([x, y, z], EQ(app(C, x, y, z), app(x, z, y))), "combinator_C")
    add(ForAll([x, y, z], EQ(app(S, x, y, z), app(x, z, app(y, z)))), "combinator_S")
    add(ForAll([f], EQ(APP(Y, f), APP(f, APP(Y, f)))), "combinator_Y")

    # Beta reduction using Z3's SUBST
    add(ForAll([x, y], EQ(APP(ABS(x), y), SUBST(0, y, x))), "beta_reduction")

    # APP-JOIN distributivity (both directions)
    add(
        ForAll([f, g, x], EQ(APP(JOIN(f, g), x), JOIN(APP(f, x), APP(g, x)))),
        "app_join_distributivity",
    )
    add(
        ForAll([f, x, y], LEQ(JOIN(APP(f, x), APP(f, y)), APP(f, JOIN(x, y)))),
        "app_join_reverse_ineq",
    )

    # APP monotonicity (in both arguments)
    add(
        ForAll([f, g, x], Implies(LEQ(f, g), LEQ(APP(f, x), APP(g, x)))),
        "app_monotonic_func",
    )
    add(
        ForAll([f, x, y], Implies(LEQ(x, y), LEQ(APP(f, x), APP(f, y)))),
        "app_monotonic_arg",
    )

    # ABS monotonicity
    add(ForAll([x, y], Implies(LEQ(x, y), LEQ(ABS(x), ABS(y)))), "abs_monotonic")

    # BOT/TOP preservation
    add(ForAll([x], EQ(APP(BOT, x), BOT)), "app_bot_preservation")
    add(ForAll([x], EQ(APP(TOP, x), TOP)), "app_top_preservation")
    add(EQ(ABS(BOT), BOT), "abs_bot_preservation")
    add(EQ(ABS(TOP), TOP), "abs_top_preservation")

    # JOIN distributivity over ABS
    add(
        ForAll([x, y], EQ(ABS(JOIN(x, y)), JOIN(ABS(x), ABS(y)))),
        "abs_join_distributivity",
    )

    # Extensionality
    add(
        ForAll([f, g], Implies(ForAll([x], LEQ(APP(f, x), APP(g, x))), LEQ(f, g))),
        "extensionality_leq",
    )
    add(
        ForAll([f, g], Implies(ForAll([x], EQ(APP(f, x), APP(g, x))), EQ(f, g))),
        "extensionality_eq",
    )

    # Eta conversion
    add(ForAll([f], EQ(ABS(APP(shift(f), VAR(0))), f)), "eta_conversion")


def convergence_theory(s: z3.Solver) -> None:
    add = Adder(s, "conv")
    i = z3.Int("i")

    # DIV tests for convergence
    add(EQ(DIV, APP(Y, TUPLE(TOP))), "div_definition")
    add(ForAll([x], EQ(APP(DIV, x), APP(DIV, APP(x, TOP)))), "div_recursion")
    add(LEQ(APP(DIV, BOT), BOT), "div_bot")

    # CONV is a least fixed point
    add(CONV(TOP), "top_converges")
    add(
        ForAll(
            [x, y],
            Implies(CONV(APP(x, y)), CONV(x)),
            patterns=[CONV(APP(x, y))],  # prevents hang
        ),
        "app_conv_lhs",
    )
    add(
        ForAll(
            [x, y],
            Implies(CONV(COMP(x, y)), CONV(x)),
            patterns=[CONV(COMP(x, y))],  # prevents hang
        ),
        "comp_conv_lhs",
    )
    add(Not(CONV(BOT)), "bot_diverges")
    add(ForAll([x], Implies(Not(CONV(x)), LEQ(x, BOT))), "non_conv_below_bot")
    add(ForAll([x, y], Implies(LEQ(x, y), Implies(CONV(x), CONV(y)))), "conv_monotonic")

    # Base cases
    add(ForAll([i], CONV(VAR(i))), "var_converges")

    # DIV's relation to CONV
    add(ForAll([x], Implies(CONV(x), LEQ(TOP, APP(DIV, x)))), "conv_implies_div")
    add(ForAll([x], Implies(LEQ(TOP, APP(DIV, x)), CONV(x))), "div_implies_conv")

    # Multi-argument functions
    add(ForAll([x], Implies(CONV(x), CONV(ABS(x)))), "abs_preserves_conv")
    add(ForAll([x], Implies(CONV(x), CONV(APP(x, TOP)))), "app_top_preserves_conv")


def simple_theory(s: z3.Solver) -> None:
    """Theory of SIMPLE type, defined as join of section-retract pairs."""
    add = Adder(s, "simple")

    def above_all_sr(candidate: z3.ExprRef) -> z3.ExprRef:
        s1, r1 = z3.Consts("s1 r1", Term)  # Different names for bound variables
        return ForAll(
            [s1, r1],
            Implies(LEQ(COMP(r1, s1), I), LEQ(TUPLE(s1, r1), candidate)),
        )

    # SIMPLE is above all section-retract pairs.
    add(above_all_sr(SIMPLE), "simple_upper_bound")

    # SIMPLE is the least such term.
    add(
        ForAll([x], Implies(above_all_sr(x), LEQ(SIMPLE, x))),
        "simple_least_upper_bound",
    )


def has_inhabs(t: z3.ExprRef, *inhabs: z3.ExprRef) -> z3.ExprRef:
    return ForAll([x], Or(*[EQ(APP(t, x), i) for i in inhabs]))


def type_theory(s: z3.Solver) -> None:
    """Theory of types and type membership."""
    add = Adder(s, "type")

    # Definition of TYPE
    add(EQ(TYPE, ABS(APP(Y, ABS(JOIN(I, COMP(v1, v0)))))), "type_def_1")
    add(EQ(TYPE, ABS(APP(Y, ABS(JOIN(I, COMP(v0, v1)))))), "type_def_2")

    # Types are closures
    add(ForAll([t], LEQ(I, APP(TYPE, t))), "type_includes_identity")
    add(
        ForAll([t], EQ(COMP(APP(TYPE, t), APP(TYPE, t)), APP(TYPE, t))),
        "type_idempotent",
    )

    # TYPE is a type
    add(LEQ(I, TYPE), "type_is_a_type_1")
    add(EQ(COMP(TYPE, TYPE), TYPE), "type_is_a_type_2")
    add(
        ForAll([t], EQ(APP(TYPE, APP(TYPE, t)), APP(TYPE, t))),
        "type_application_idempotent",
    )
    add(
        ForAll([t], EQ(APP(TYPE, t), JOIN(I, COMP(t, APP(TYPE, t))))),
        "type_fixed_point_1",
    )
    add(
        ForAll([t], EQ(APP(TYPE, t), JOIN(I, COMP(APP(TYPE, t), t)))),
        "type_fixed_point_2",
    )

    # Inhabitants are fixed points
    add(OFTYPE(TYPE, TYPE), "type_inhabits_itself")
    add(ForAll([t], OFTYPE(APP(TYPE, t), TYPE)), "type_app_inhabits_type")
    add(has_inhabs(DIV, TOP, BOT), "div_inhabitants")
    add(has_inhabs(semi, TOP, BOT, I), "semi_inhabitants")
    add(has_inhabs(unit, TOP, I), "unit_inhabitants")
    add(
        has_inhabs(boool, TOP, true_, false_, JOIN(true_, false_), BOT),
        "boool_inhabitants",
    )
    add(has_inhabs(bool_, TOP, true_, false_, BOT), "bool_inhabitants")


def add_theory(s: z3.Solver) -> None:
    counter["add_theory"] += 1
    de_bruijn_theory(s)
    order_theory(s)
    lambda_theory(s)
    convergence_theory(s)
    simple_theory(s)
    type_theory(s)


# https://microsoft.github.io/z3guide/programming/Parameters/#global-parameters
DEFAULT_TIMEOUT_MS = 4294967295


@contextmanager
def solver_timeout(
    solver: z3.Solver, *, timeout_ms: int
) -> Generator[None, None, None]:
    """Context manager to set a timeout on a Z3 solver."""
    # This works around latch of .get() interface by patching the solver object
    # with a .timeout_ms attribute.
    old_timeout_ms = getattr(solver, "timeout_ms", DEFAULT_TIMEOUT_MS)
    solver.set(timeout=timeout_ms)
    solver.timeout_ms = timeout_ms
    try:
        yield
    finally:
        solver.set(timeout=old_timeout_ms)
        solver.timeout_ms = old_timeout_ms


def try_prove(
    solver: z3.Solver, formula: z3.ExprRef, *, timeout_ms: int = 1000
) -> tuple[bool | None, str | None]:
    """
    Try to prove a formula is valid or invalid.

    Args:
        solver: Z3 solver to use
        formula: Formula to check validity of
        timeout_seconds: Maximum time (in seconds) to spend on the check

    Returns:
        Tuple of:
        - True if formula proved valid
        - False if formula proved invalid
        - None if formula is satisfiable but not valid
        And the counterexample model string (if formula is not valid)
    """
    counter["try_prove"] += 1
    with solver, solver_timeout(solver, timeout_ms=timeout_ms):
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
