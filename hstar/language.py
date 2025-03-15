"""
# Language of λ-join-calculus in Z3.

This uses a de Bruijn indexed representation of λ-join-calculus terms, with a
`LEQ` relation for the Scott ordering, and explicit `BOT` (bottom), `TOP` (top),
and binary `JOIN` operation wrt the Scott ordering.
"""

import inspect
from collections.abc import Callable

import z3
from z3 import ExprRef, Not

from .functools import weak_key_cache

# Term sort
Term = z3.DeclareSort("Term")

# Term constructors as uninterpreted functions
TOP = z3.Const("TOP", Term)
BOT = z3.Const("BOT", Term)
JOIN = z3.Function("JOIN", Term, Term, Term)
COMP = z3.Function("COMP", Term, Term, Term)
APP = z3.Function("APP", Term, Term, Term)
VAR = z3.Function("VAR", z3.IntSort(), Term)
ABS = z3.Function("ABS", Term, Term)

# De Bruijn operations.
SHIFT = z3.Function("SHIFT", Term, z3.IntSort(), Term)
SUBST = z3.Function("SUBST", z3.IntSort(), Term, Term, Term)

# Scott ordering.
LEQ = z3.Function("LEQ", Term, Term, z3.BoolSort())  # x [= y

# Constants.
v0 = VAR(0)
v1 = VAR(1)
v2 = VAR(2)
I = ABS(v0)
K = ABS(ABS(VAR(1)))
KI = ABS(ABS(VAR(0)))
J = JOIN(K, KI)
B = ABS(ABS(ABS(APP(v2, APP(v1, v0)))))
CB = ABS(ABS(ABS(APP(v2, APP(v0, v1)))))
C = ABS(ABS(ABS(APP(APP(v2, v0), v1))))
W = ABS(ABS(APP(APP(v1, v0), v0)))
S = ABS(ABS(ABS(APP(APP(v2, v0), APP(v1, v0)))))
Y = ABS(APP(ABS(APP(v1, APP(v0, v0))), ABS(APP(v1, APP(v0, v0)))))
Y_ = ABS(APP(ABS(APP(v0, v0)), ABS(APP(v1, APP(v0, v0)))))
V = ABS(APP(Y, ABS(JOIN(I, COMP(v1, v0)))))
DIV = APP(V, ABS(APP(v0, TOP)))
SIMPLE = z3.Const("SIMPLE", Term)  # TODO define


def CONV(term: ExprRef) -> ExprRef:
    """Check whether a term converges (is not bottom)."""
    return Not(LEQ(term, BOT))


def shift(term: ExprRef, start: int = 0, delta: int = 1) -> ExprRef:
    """
    Shift free variables in a term by delta, starting from index `start`.

    Increments (delta > 0) or decrements (delta < 0) all free variables
    with indices >= start by abs(delta).

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
                    return VAR(idx + delta)
                else:
                    return term
            elif decl_name == "ABS":
                # Handle ABS constructor
                body = term.arg(0)
                return ABS(shift(body, start + 1, delta))
            elif decl_name == "APP":
                # Handle APP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return APP(shift(lhs, start, delta), shift(rhs, start, delta))
            elif decl_name == "JOIN":
                # Handle JOIN constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return JOIN(shift(lhs, start, delta), shift(rhs, start, delta))
            elif decl_name == "COMP":
                # Handle COMP constructor
                lhs = term.arg(0)
                rhs = term.arg(1)
                return COMP(shift(lhs, start, delta), shift(rhs, start, delta))
    except Exception as e:
        # If we encounter any exception, log it and return unevaluated SHIFT
        print(f"Exception in shift: {e}")
        pass

    # Fall back to unevaluated SHIFT for any other expressions
    return SHIFT(term, z3.IntVal(start))


def join(*args: ExprRef) -> ExprRef:
    r"""Join a list of arguments."""
    if not args:
        return BOT
    result = args[0]
    for arg in args[1:]:
        result = JOIN(result, arg)
    return result


def app(*args: ExprRef) -> ExprRef:
    r"""Apply a list of arguments to a function."""
    result = args[0]
    for arg in args[1:]:
        result = APP(result, arg)
    return result


def comp(*args: ExprRef) -> ExprRef:
    r"""Compose a list of functions."""
    return COMP(args[0], comp(*args[1:])) if args else I


def TUPLE(*args: ExprRef) -> ExprRef:
    r"""Barendregt tuples `<M1,...,Mn> = \f. f M1 ... Mn`."""
    body = VAR(0)
    for arg in args:
        body = APP(body, shift(arg))
    return ABS(body)


def CONJ(a: ExprRef, b: ExprRef) -> ExprRef:
    r"""Conjunction `a -> b = \f. b o f o a`."""
    a = shift(a)
    b = shift(b)
    return ABS(comp(b, VAR(0), a))


# FIXME assumes f returns a closed term.
# TODO convert native python functions to lambda terms, e.g. f(x) -> APP(f, x)
# TODO convert python or to join terms, e.g. x|y -> JOIN(x, y)
# TODO convert python tuples to TUPLE terms, e.g. (x, y, z) -> TUPLE(x, y, z)
def hoas(f: Callable[..., ExprRef]) -> ExprRef:
    r"""Higher-order abstract syntax. FIXME assumes f returns a closed term."""
    argc = len(inspect.signature(f).parameters)
    args = [VAR(argc - i - 1) for i in range(argc)]
    body = f(*args)
    for _ in range(argc):
        body = ABS(body)
    return body


def simple(f: Callable[[ExprRef, ExprRef], ExprRef]) -> ExprRef:
    """HOAS constructor for simple types."""
    return APP(SIMPLE, hoas(f))


# Types.
ANY = I
semi = simple(lambda a, a1: CONJ(a, a1))
boool = simple(lambda a, a1: CONJ(a, CONJ(a, a1)))
pre_pair = simple(lambda a, a1: CONJ(CONJ(ANY, a), CONJ(CONJ(ANY, a), a1)))
unit = APP(V, JOIN(semi, ABS(I)))
disamb_bool = hoas(lambda f, x, y: app(f, app(f, x, TOP), app(f, TOP, y)))
bool_ = APP(V, JOIN(boool, disamb_bool))
disamb_pair = hoas(lambda p, f: app(f, app(p, K), app(p, KI)))
pair = APP(V, JOIN(pre_pair, disamb_pair))


def OFTYPE(x: ExprRef, t: ExprRef) -> ExprRef:
    """Check if x is of type t."""
    return LEQ(APP(t, x), x)


def subst(i: int, replacement: ExprRef, term: ExprRef) -> ExprRef:
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


@weak_key_cache
def free_vars(term: ExprRef) -> frozenset[int]:
    """Return the set of de Bruijn VARs of free variables in a term."""
    if z3.is_const(term) and term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        return frozenset()
    if z3.is_app(term):
        decl = term.decl()
        decl_name = str(decl)
        if decl_name == "VAR":
            i: int = term.arg(0).as_long()
            return frozenset([i])
        if decl_name == "ABS":
            return frozenset(i + 1 for i in free_vars(term.arg(0)))
        if decl_name == "APP" or decl_name == "JOIN" or decl_name == "COMP":
            return free_vars(term.arg(0)) | free_vars(term.arg(1))
    raise ValueError(f"Unknown term: {term}")


def abstract(term: ExprRef) -> ExprRef:
    """
    A Curry-style combinatory abstraction algorithm.

    This abstracts the de Bruijn indexed variable `VAR(0)` and shifts remaining
    variables. This should satisfy `beta,eta |- ABS(term) = abstract(term)`, but
    without introducing an `ABS` term.

    Warning: this assumes the VAR in question never appears inside an ABS. It is
    ok if ABS appears in term, but only in closed subterms.
    """
    # Handle expressions that don't contain VAR(0) directly
    if not has_var_0(term):
        # K abstraction
        return APP(K, shift(term, delta=-1))

    if z3.is_app(term):
        decl = term.decl()
        decl_name = str(decl)
        if decl_name == "VAR":
            # I abstraction
            assert term.arg(0).as_long() == 0
            return I

        if decl_name == "APP":
            # I,K,C,S,W,COMP,eta abstraction
            lhs, rhs = term.arg(0), term.arg(1)
            if has_var_0(lhs):
                lhs_abs = abstract(lhs)
                if has_var_0(rhs):
                    if is_var_0(rhs):  # rhs is exactly VAR(0)
                        return APP(W, lhs_abs)
                    else:
                        return APP(APP(S, lhs_abs), abstract(rhs))
                else:
                    return APP(APP(C, lhs_abs), shift(rhs, delta=-1))
            else:
                assert has_var_0(rhs)
                if is_var_0(rhs):  # rhs is exactly VAR(0)
                    return shift(lhs, delta=-1)
                else:
                    return COMP(shift(lhs, delta=-1), abstract(rhs))

        elif decl_name == "COMP":
            # K,B,CB,C,S,COMP,eta abstraction
            lhs, rhs = term.arg(0), term.arg(1)
            if has_var_0(lhs):
                lhs_abs = abstract(lhs)
                if has_var_0(rhs):
                    return APP(APP(S, COMP(B, lhs_abs)), abstract(rhs))
                else:
                    if is_var_0(lhs):
                        return APP(CB, shift(rhs, delta=-1))
                    else:
                        return COMP(APP(CB, shift(rhs, delta=-1)), lhs_abs)
            else:
                assert has_var_0(rhs)
                if is_var_0(rhs):
                    return APP(B, shift(lhs, delta=-1))
                else:
                    return COMP(APP(B, shift(lhs, delta=-1)), abstract(rhs))

        elif decl_name == "JOIN":
            # K-compose-eta abstraction
            lhs, rhs = term.arg(0), term.arg(1)
            if has_var_0(lhs):
                if has_var_0(rhs):
                    return JOIN(abstract(lhs), abstract(rhs))
                elif is_var_0(lhs):
                    return APP(J, shift(rhs, delta=-1))
                else:
                    return COMP(APP(J, shift(rhs, delta=-1)), abstract(lhs))
            else:
                assert has_var_0(rhs)
                if is_var_0(rhs):
                    return APP(J, shift(lhs, delta=-1))
                else:
                    return COMP(APP(J, shift(lhs, delta=-1)), abstract(rhs))

    raise ValueError(f"Unsupported term: {term}")


def has_var_0(term: ExprRef) -> bool:
    """Check if VAR(0) appears in a term."""
    return 0 in free_vars(term)


def is_var_0(term: ExprRef) -> bool:
    """Check if a term is exactly VAR(0)."""
    if z3.is_app(term) and str(term.decl()) == "VAR":
        i: int = term.arg(0).as_long()
        return i == 0
    return False
