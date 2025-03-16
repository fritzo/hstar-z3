"""
# Language of λ-join-calculus in Z3.

This uses a de Bruijn indexed representation of λ-join-calculus terms, with a
`LEQ` relation for the Scott ordering, and explicit `BOT` (bottom), `TOP` (top),
and binary `JOIN` operation wrt the Scott ordering.
"""

import functools
import inspect
import itertools
import logging
from collections.abc import Callable, Iterator

import z3
from z3 import ExprRef, Not

from hstar.itertools import iter_subsets

logger = logging.getLogger(__name__)

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
C = ABS(ABS(ABS(APP(APP(v2, v0), v1))))
CI = ABS(ABS(APP(v1, v0)))
CB = ABS(ABS(ABS(APP(v2, APP(v0, v1)))))
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


@functools.lru_cache
def shift(term: ExprRef, start: int = 0, delta: int = 1) -> ExprRef:
    """
    Shift free variables in a term by delta, starting from index `start`.

    Increments (delta > 0) or decrements (delta < 0) all free variables
    with indices >= start by abs(delta).

    Eagerly applies shifting rules for Term expressions when possible,
    otherwise returns an unevaluated SHIFT call.
    """
    # Handle special constants first
    if z3.is_const(term):
        if term == TOP:
            return TOP
        elif term == BOT:
            return BOT
        # Check if we have a symbolic variable (like a, x, etc.)
        if term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
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
            elif z3.is_eq(term):
                lhs, rhs = term.children()
                return shift(lhs, start, delta) == shift(rhs, start, delta)
    except Exception as e:
        # If we encounter any exception, log it and return unevaluated SHIFT
        logger.warning(f"Exception in shift: {e}")
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


@functools.lru_cache
def subst(i: int, replacement: ExprRef, term: ExprRef) -> ExprRef:
    """
    Substitute a term for a variable in another term.
    Eagerly applies substitution rules for Term expressions when possible,
    otherwise returns an unevaluated SUBST call.
    """
    idx = z3.IntVal(i)

    # Handle special constants first
    if z3.is_const(term):
        if term == TOP:
            return TOP
        elif term == BOT:
            return BOT
        # Check if we have a symbolic variable (like a, x, etc.)
        if term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
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
            elif z3.is_eq(term):
                lhs, rhs = term.children()
                return subst(i, replacement, lhs) == subst(i, replacement, rhs)
    except Exception as e:
        # If we encounter any exception, log it and return unevaluated SUBST
        print(f"Exception in subst: {e}")
        pass

    # Fall back to unevaluated SUBST for any other expressions
    return SUBST(idx, replacement, term)


@functools.lru_cache
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
            return frozenset(i - 1 for i in free_vars(term.arg(0)) if i)
        if decl_name == "APP" or decl_name == "JOIN" or decl_name == "COMP":
            return free_vars(term.arg(0)) | free_vars(term.arg(1))
        if z3.is_eq(term):
            lhs, rhs = term.children()
            return free_vars(lhs) | free_vars(rhs)
    raise TypeError(f"Unknown term: {term}")


def abstract(term: ExprRef, i: int = 0) -> ExprRef:
    """
    A Curry-style combinatory abstraction algorithm.

    This abstracts the de Bruijn indexed variable `VAR(0)` and shifts remaining
    variables. This should satisfy `beta,eta |- ABS(term) = abstract(term)`, but
    without introducing an `ABS` term.

    Warning: this assumes the VAR in question never appears inside an ABS. It is
    ok if ABS appears in term, but only in closed subterms.
    """
    # Reduce to case i == 0.
    assert i >= 0
    if i != 0:
        if 0 in free_vars(term):
            term = shift(term, delta=1)  # [a, b, i, x, y] -> [a, b, i, x, y, _]
            i += 1
        term = subst(i, VAR(0), term)  # [a, b, i, x, y, _] -> [a, b, _, x, y, i]
        term = shift(term, start=i, delta=-1)  # [a, b, _, x, y, i] -> [a, b, x, y, i]

    if z3.is_eq(term):
        lhs, rhs = term.children()
        return _abstract(lhs) == _abstract(rhs)

    return _abstract(term)


@functools.lru_cache
def _abstract(term: ExprRef) -> ExprRef:
    # Handle terms that don't contain VAR(0) directly
    if not _has_v0(term):
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
            if _has_v0(lhs):
                lhs_abs = _abstract(lhs)
                if _has_v0(rhs):
                    if _is_v0(rhs):  # rhs is exactly VAR(0)
                        return APP(W, lhs_abs)
                    else:
                        return APP(APP(S, lhs_abs), _abstract(rhs))
                else:
                    return APP(APP(C, lhs_abs), shift(rhs, delta=-1))
            else:
                assert _has_v0(rhs)
                if _is_v0(rhs):  # rhs is exactly VAR(0)
                    return shift(lhs, delta=-1)
                else:
                    return COMP(shift(lhs, delta=-1), _abstract(rhs))

        elif decl_name == "COMP":
            # K,B,CB,C,S,COMP,eta abstraction
            lhs, rhs = term.arg(0), term.arg(1)
            if _has_v0(lhs):
                lhs_abs = _abstract(lhs)
                if _has_v0(rhs):
                    return APP(APP(S, COMP(B, lhs_abs)), _abstract(rhs))
                else:
                    if _is_v0(lhs):
                        return APP(CB, shift(rhs, delta=-1))
                    else:
                        return COMP(APP(CB, shift(rhs, delta=-1)), lhs_abs)
            else:
                assert _has_v0(rhs)
                if _is_v0(rhs):
                    return APP(B, shift(lhs, delta=-1))
                else:
                    return COMP(APP(B, shift(lhs, delta=-1)), _abstract(rhs))

        elif decl_name == "JOIN":
            # K-compose-eta abstraction
            lhs, rhs = term.arg(0), term.arg(1)
            if _has_v0(lhs):
                if _has_v0(rhs):
                    return JOIN(_abstract(lhs), _abstract(rhs))
                elif _is_v0(lhs):
                    return APP(J, shift(rhs, delta=-1))
                else:
                    return COMP(APP(J, shift(rhs, delta=-1)), _abstract(lhs))
            else:
                assert _has_v0(rhs)
                if _is_v0(rhs):
                    return APP(J, shift(lhs, delta=-1))
                else:
                    return COMP(APP(J, shift(lhs, delta=-1)), _abstract(rhs))

    raise ValueError(f"Unsupported term: {term}")


def _has_v0(term: ExprRef) -> bool:
    """Check if VAR(0) appears in a term."""
    return 0 in free_vars(term)


def _is_v0(term: ExprRef) -> bool:
    """Check if a term is exactly VAR(0)."""
    if z3.is_app(term) and str(term.decl()) == "VAR":
        i: int = term.arg(0).as_long()
        return i == 0
    return False


def iter_eta_substitutions(expr: ExprRef) -> Iterator[ExprRef]:
    """
    Iterate over Hindley-style substitutions:
        [x/x], [x/a], [x/APP x a] (and maybe [x/COMP x a])
    """
    varlist = sorted(free_vars(expr))
    assert varlist
    fresh = next(i for i in itertools.count() if i not in varlist)
    for cases in itertools.product(range(3), repeat=len(varlist)):
        result = expr
        for var, case in zip(varlist, cases, strict=False):
            if case == 0:
                pass  # do nothing
            elif case == 1:
                result = subst(var, VAR(fresh), result)
            elif case == 2:
                result = subst(var, APP(VAR(var), VAR(fresh)), result)
            # elif case == 3:
            #    result = subst(var, COMP(VAR(var), VAR(fresh)), result)
        if any(cases):
            yield abstract(result, fresh)
        else:
            yield result


def iter_closure_maps(expr: ExprRef) -> Iterator[ExprRef]:
    """Iterate over all closing abstractions, including variable
    coincidence."""
    if not free_vars(expr):
        yield expr
        return
    for group in iter_subsets(free_vars(expr)):
        if not group:
            continue
        var = min(group)
        abstracted = expr
        for other in group - {var}:
            abstracted = subst(other, VAR(var), abstracted)
        abstracted = abstract(abstracted, var)
        assert len(free_vars(abstracted)) < len(free_vars(expr))
        yield from iter_closure_maps(abstracted)


def iter_closures(expr: ExprRef) -> Iterator[ExprRef]:
    assert z3.is_eq(expr), expr
    if not free_vars(expr):
        yield expr
        return
    for expr2 in iter_eta_substitutions(expr):
        assert z3.is_eq(expr2), expr2
        for expr3 in iter_closure_maps(expr2):
            assert z3.is_eq(expr3), expr3
            yield expr3


def ForAllHindley(vs: list[ExprRef], body: ExprRef) -> Iterator[ExprRef]:
    """
    Universally quantifies over all free de Bruijn variables in a formula, then
    eliminates the quantifiers using Hindley's extensionality trick [1] and a
    Curry-style combinatory abstraction algorithm.

    Warning: none of the vs may appear inside ABS(-) terms.

    [1] Roger Hindley (1967) "Axioms for strong reduction in combinatory logic"
    """
    assert not free_vars(body)
    if not vs:
        yield body
        return

    # Convert nominal variables to de Bruijn indices
    for i, v in enumerate(reversed(vs)):
        body = z3.substitute(body, (v, VAR(i)))

    # Eliminate quantifiers
    for derived in iter_closures(body):
        assert z3.is_eq(derived)
        lhs, rhs = derived.children()
        if z3.eq(lhs, rhs):
            continue  # Skip trivial equalities.
        yield derived
