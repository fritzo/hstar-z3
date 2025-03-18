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
SHIFT = z3.Function("SHIFT", Term, z3.IntSort(), z3.IntSort(), Term)
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
CB = ABS(ABS(ABS(APP(v1, APP(v2, v0)))))
W = ABS(ABS(APP(APP(v1, v0), v0)))
S = ABS(ABS(ABS(APP(APP(v2, v0), APP(v1, v0)))))
Y = ABS(APP(ABS(APP(v1, APP(v0, v0))), ABS(APP(v1, APP(v0, v0)))))
Y_ = ABS(APP(ABS(APP(v0, v0)), ABS(APP(v1, APP(v0, v0)))))
V = ABS(APP(Y, ABS(JOIN(I, COMP(v1, v0)))))
V_ = ABS(APP(Y, ABS(JOIN(I, COMP(v0, v1)))))
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
            return SHIFT(term, z3.IntVal(start), z3.IntVal(delta))

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
    return SHIFT(term, z3.IntVal(start), z3.IntVal(delta))


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


def _maybe_abstract(term: ExprRef) -> tuple[bool, bool, ExprRef]:
    """
    Return a tuple of (has_v0, is_v0, abstracted_term).

    if has_v0, then abstracted_term is beta-eta equivalent to ABS(term).
    otherwise, abstracted_term is beta-eta equivalent to shift(term, delta=-1).
    """
    if z3.eq(term, v0):
        return True, True, I
    if 0 in free_vars(term):
        return True, False, _abstract(term)
    return False, False, shift(term, delta=-1)


@functools.lru_cache
def _abstract(term: ExprRef) -> ExprRef:
    """Returns a combinator beta-eta equivalent to ABS(term)."""
    # Handle terms that don't contain VAR(0) directly
    if not _has_v0(term):
        # K abstraction
        term_shift = shift(term, delta=-1)
        return APP(K, term_shift)

    if z3.is_app(term):
        decl = term.decl()
        decl_name = str(decl)
        if decl_name == "VAR":
            # I abstraction
            assert term.arg(0).as_long() == 0
            return I

        if decl_name == "APP":
            # I,K,C,S,W,COMP,eta abstraction
            lhs_has, lhs_is, lhs = _maybe_abstract(term.arg(0))
            rhs_has, rhs_is, rhs = _maybe_abstract(term.arg(1))
            if lhs_has:
                if rhs_is:
                    return APP(W, lhs)  # W lhs x = lhs x x
                if rhs_has:
                    return app(S, lhs, rhs)  # S lhs rhs x = lhs x (rhs x)
                return app(C, lhs, rhs)  # C lhs rhs x = lhs x rhs
            assert rhs_has
            if rhs_is:
                return lhs  # lhs x
            return COMP(lhs, rhs)  # lhs o rhs x = lhs (rhs x)

        elif decl_name == "COMP":
            # K,B,CB,C,S,COMP,eta abstraction
            lhs_has, lhs_is, lhs = _maybe_abstract(term.arg(0))
            rhs_has, rhs_is, rhs = _maybe_abstract(term.arg(1))
            if lhs_has:
                if rhs_has:
                    # S (B o lhs) rhs x = B (lhs x) (rhs x) = (lhs x) o (rhs x)
                    return app(S, COMP(B, lhs), rhs)
                if lhs_is:
                    return APP(CB, rhs)  # CB rhs x = x o rhs
                # (CB rhs) o lhs x = CB rhs (lhs x) = (lhs x) o rhs
                return COMP(APP(CB, rhs), lhs)
            assert rhs_has
            if rhs_is:
                return APP(B, lhs)  # B lhs x = lhs o x
            # (B lhs) o rhs x = B lhs (rhs x) = lhs o (rhs x)
            return COMP(APP(B, lhs), rhs)

        elif decl_name == "JOIN":
            # K-compose-eta-commutative-idempotent abstraction
            lhs_has, lhs_is, lhs = _maybe_abstract(term.arg(0))
            rhs_has, rhs_is, rhs = _maybe_abstract(term.arg(1))
            if lhs_is:
                if rhs_is:
                    return I  # I x = x = x | x
                if rhs_has:
                    return app(S, J, rhs)  # S J rhs x = J x (rhs x) = x | rhs x
                return APP(J, rhs)  # J rhs x = rhs | x
            if lhs_has:
                if rhs_is:
                    return app(S, J, lhs)  # S J lhs x = J x (lhs x) = x | lhs x
                if rhs_has:
                    return JOIN(lhs, rhs)
                return COMP(APP(J, rhs), lhs)  # (J rhs) o lhs x = J rhs (lhs x)
            if rhs_is:
                return APP(J, lhs)  # J lhs x = lhs | x
            assert rhs_has
            return COMP(APP(J, lhs), rhs)  # (J lhs) o rhs x = J lhs (rhs x)

    raise TypeError(f"Unsupported term: {term}")


def _has_v0(term: ExprRef) -> bool:
    """Check if VAR(0) appears in a term."""
    return 0 in free_vars(term)


def _is_v0(term: ExprRef) -> bool:
    """Check if a term is exactly VAR(0)."""
    if z3.is_app(term) and str(term.decl()) == "VAR":
        i: int = term.arg(0).as_long()
        return i == 0
    return False


def iter_eta_substitutions(
    expr: ExprRef, *, compose: bool = False
) -> Iterator[ExprRef]:
    """
    Iterate over Hindley-style substitutions:
        [x/x], [x/a], [x/APP x a] (and maybe [x/COMP x a])
    """
    varlist = sorted(free_vars(expr))
    assert varlist
    fresh = next(i for i in itertools.count() if i not in varlist)
    actions = range(4 if compose else 3)
    for cases in itertools.product(actions, repeat=len(varlist)):
        result = expr
        for var, case in zip(varlist, cases, strict=False):
            if case == 0:
                pass  # do nothing
            elif case == 1:
                result = subst(var, VAR(fresh), result)
            elif case == 2:
                result = subst(var, APP(VAR(var), VAR(fresh)), result)
            elif case == 3:
                result = subst(var, COMP(VAR(var), VAR(fresh)), result)
        if any(cases):
            yield abstract(result, fresh)
        else:
            yield result


def iter_closure_maps(expr: ExprRef) -> Iterator[ExprRef]:
    """Iterate over all closing abstractions, including variable coincidence."""
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
        logger.debug(f"Abstracted {expr} -> {abstracted}")
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


def _z3_occurs_in(subexpr: ExprRef, expr: ExprRef) -> bool:
    if z3.eq(expr, subexpr):
        return True
    if z3.is_app(expr):
        return any(_z3_occurs_in(subexpr, child) for child in expr.children())
    return False


def forall_to_open(expr: ExprRef) -> ExprRef:
    """
    Convert a z3.ForAll formula into an open formula with free VARs.

    Warning: none of the vs may appear inside ABS(-) terms.
    """
    assert expr.is_forall()
    body = expr.body()
    assert z3.is_eq(body)

    # Note z3 abstracts in reverse order.
    fv: list[ExprRef] = []
    for i in itertools.count():
        var = z3.Var(i, Term)
        if not _z3_occurs_in(var, body):
            break
        fv.append(var)
    for i, var in enumerate(reversed(fv)):
        body = z3.substitute(body, (var, VAR(i)))
    return body


def QEHindley(expr: ExprRef) -> set[ExprRef]:
    """
    Converts a z3.ForAll formula into a set of closed formulas.

    This eliminates the quantifiers using Hindley's extensionality trick [1] and
    a Curry-style combinatory abstraction algorithm.

    Warning: none of the vs may appear inside ABS(-) terms.

    [1] Roger Hindley (1967) "Axioms for strong reduction in combinatory logic"
    """
    result: set[ExprRef] = set()
    body = forall_to_open(expr)
    for derived in iter_closures(body):
        assert z3.is_eq(derived)
        lhs, rhs = derived.children()
        if z3.eq(lhs, rhs):
            continue  # Skip trivial equalities.
        result.add(derived)
    return result
