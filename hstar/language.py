"""
# Language of untyped λ-join-calculus in Z3.

This uses combinatory algebra to represent closed λ-join-calculus terms, with a
`LEQ` relation for the Scott ordering, and explicit `BOT` (bottom), `TOP` (top),
and binary `JOIN` operation wrt the Scott ordering.
"""

import functools
import itertools
import logging
from collections.abc import Iterator

import z3
from z3 import ExprRef, Not

from hstar.functools import weak_key_cache
from hstar.hashcons import intern
from hstar.itertools import iter_subsets

logger = logging.getLogger(__name__)

# Term sort
Term = z3.DeclareSort("Term")

# Term constructors as uninterpreted functions
APP = z3.Function("APP", Term, Term, Term)
COMP = z3.Function("COMP", Term, Term, Term)
JOIN = z3.Function("JOIN", Term, Term, Term)
TOP, BOT = z3.Consts("TOP BOT", Term)
I, K, KI, J, B, C, CI, CB = z3.Consts("I K KI J B C CI CB", Term)
W, S, Y, V, DIV, A = z3.Consts("W S Y V DIV A", Term)

# Scott ordering x [= y.
if LEQ_IS_Z3_PARTIAL_ORDER := False:  # https://github.com/Z3Prover/z3/issues/7592
    LEQ = z3.PartialOrder(Term, 0)
else:
    LEQ = z3.Function("LEQ", Term, Term, z3.BoolSort())


# Lambda calculus tools: variables and abstraction.


@functools.cache
def VAR(i: int) -> ExprRef:
    return intern(z3.Var(i, Term))


_EMPTY_VARS: frozenset[ExprRef] = frozenset()


@weak_key_cache
def free_vars(expr: ExprRef) -> frozenset[ExprRef]:
    """Return the set of free variables in a term."""
    if z3.is_var(expr):
        return frozenset((expr,))
    if z3.is_const(expr):
        return _EMPTY_VARS
    if z3.is_app(expr):
        return intern(frozenset.union(*map(free_vars, expr.children())))
    raise TypeError(f"Unknown term: {expr}")


def get_fresh(*exprs: ExprRef) -> ExprRef:
    """Return a fresh variable that does not appear in any of the exprs."""
    avoid = [free_vars(expr) for expr in exprs]
    for i in itertools.count():
        var = VAR(i)
        if not any(var in avoid for avoid in avoid):
            return var


def occurs_in(sub: ExprRef, expr: ExprRef) -> bool:
    """Return whether a subexpression occurs in an expression."""
    if z3.eq(sub, expr):
        return True
    if z3.is_var(expr):
        return False
    if z3.is_app(expr):
        return any(occurs_in(sub, arg) for arg in expr.children())
    raise TypeError(f"Unknown term: {expr}")


def _maybe_lam(var: ExprRef, body: ExprRef) -> tuple[bool, bool, ExprRef]:
    """
    Return a tuple of (has_var, is_var, lam_body).

    If has_var then lam_body is lam(var, body).
    Otherwise lam_body is just body.
    """
    if z3.eq(var, body):
        return True, True, I
    if var in free_vars(body):
        return True, False, lam(var, body)
    return False, False, body


@weak_key_cache
def lam(var: ExprRef, body: ExprRef) -> ExprRef:
    """
    A Curry-style combinatory abstraction algorithm.

    This abstracts the z3 nominal variable `var` from the `body`.
    This should satisfy `beta,eta |- APP(lam(var, body), var) == body`.
    """
    assert z3.is_var(var)

    # Handle terms that don't contain var directly
    if var not in free_vars(body):
        # K abstraction
        return APP(K, body)

    if z3.is_var(body):
        assert z3.eq(var, body)
        # I abstraction
        return I

    if z3.is_app(body):
        decl = body.decl()
        decl_name = str(decl)

        if decl_name == "APP":
            # I,K,C,S,W,COMP,eta abstraction
            lhs_has, lhs_is, lhs = _maybe_lam(var, body.arg(0))
            rhs_has, rhs_is, rhs = _maybe_lam(var, body.arg(1))
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
            lhs_has, lhs_is, lhs = _maybe_lam(var, body.arg(0))
            rhs_has, rhs_is, rhs = _maybe_lam(var, body.arg(1))
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
            lhs_has, lhs_is, lhs = _maybe_lam(var, body.arg(0))
            rhs_has, rhs_is, rhs = _maybe_lam(var, body.arg(1))
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

        elif decl_name == "LEQ":
            lhs, rhs = body.children()
            return LEQ(lam(var, lhs), lam(var, rhs))

    if z3.is_eq(body):
        lhs, rhs = body.children()
        return lam(var, lhs) == lam(var, rhs)

    raise TypeError(f"Unsupported term: {body}")


# Syntactic sugar for combinators.


def CONV(term: ExprRef) -> ExprRef:
    """Check whether a term converges (is not bottom)."""
    return Not(term == BOT)


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
    assert args
    result = args[0]
    for arg in args[1:]:
        result = APP(result, arg)
    return result


def comp(*args: ExprRef) -> ExprRef:
    r"""Compose a list of functions."""
    if not args:
        return I
    result = args[0]
    for arg in args[1:]:
        result = COMP(result, arg)
    return result


def conj(*args: ExprRef) -> ExprRef:
    r"""
    Conjunction `a -> b = \f. b o f o a`.
    Associates as a -> b -> c = a -> (b -> c).
    """
    assert args
    args, result = args[:-1], args[-1]
    while args:
        args, arg = args[:-1], args[-1]
        f = get_fresh(arg, result)
        result = lam(f, comp(result, f, arg))
    return result


def TUPLE(*args: ExprRef) -> ExprRef:
    r"""Barendregt tuples `<M1,...,Mn> = \f. f M1 ... Mn`."""
    var = get_fresh(*args)
    return lam(var, app(var, *args))


def simple(a: ExprRef, a_: ExprRef, body: ExprRef) -> ExprRef:
    """Constructor for simple types."""
    return APP(A, lam(a, lam(a_, body)))


# Types.
ANY = I
semi, unit = z3.Consts("semi unit", Term)
boool, bool_ = z3.Consts("boool bool", Term)
pre_box, box = z3.Consts("pre_box box", Term)
pre_pair, pair = z3.Consts("pre_pair pair", Term)


def INHABITS(x: ExprRef, t: ExprRef) -> ExprRef:
    """Check if x is of type t."""
    return LEQ(APP(t, x), x)


def hole_closure(expr: ExprRef) -> tuple[list[ExprRef], ExprRef]:
    """Replace free variables of an expression with fresh holes."""
    if not free_vars(expr):
        return [], expr
    holes: list[ExprRef] = []
    subs: list[tuple[ExprRef, ExprRef]] = []
    for var in free_vars(expr):
        hole = z3.FreshConst(Term, prefix="hole")
        holes.append(hole)
        subs.append((var, hole))
    return holes, z3.substitute(expr, *subs)


def _as_pattern(expr: ExprRef, parts: set[ExprRef]) -> None:
    """Extract patterns from a term."""
    if z3.is_not(expr) or z3.is_and(expr) or z3.is_or(expr) or z3.is_implies(expr):
        for arg in expr.children():
            _as_pattern(arg, parts)
    else:
        parts.add(expr)


def as_pattern(expr: ExprRef) -> z3.MultiPattern:
    """Extract maximal patterns from a term."""
    result: set[ExprRef] = set()
    _as_pattern(expr, result)
    return z3.MultiPattern(*sorted(result, key=str))


def is_eq_or_leq(expr: ExprRef) -> bool:
    if z3.is_eq(expr):
        return True
    if z3.is_app(expr) and expr.decl() == LEQ:
        return True
    return False


def iter_eta_substitutions(
    expr: ExprRef, *, compose: bool = False
) -> Iterator[ExprRef]:
    """
    Iterate over Hindley-style substitutions:
        [x/x], [x/a], [x/APP x a] (and maybe [x/COMP x a])
    """
    varlist = sorted(free_vars(expr), key=str)
    fresh = get_fresh(expr)
    actions = range(4 if compose else 3)
    for cases in itertools.product(actions, repeat=len(varlist)):
        result = expr
        for var, case in zip(varlist, cases, strict=True):
            if case == 0:
                pass  # do nothing
            elif case == 1:
                result = z3.substitute(result, (var, fresh))
            elif case == 2:
                result = z3.substitute(result, (var, APP(var, fresh)))
            elif case == 3:
                result = z3.substitute(result, (var, COMP(var, fresh)))
        if any(cases):
            yield lam(fresh, result)
        else:
            yield result


def iter_trivial_substitutions(expr: ExprRef) -> Iterator[ExprRef]:
    """Iterate over substitutions of [x/TOP] and [x/BOT]."""
    varlist = sorted(free_vars(expr), key=str)
    actions = [None, TOP, BOT]
    for cases in itertools.product(actions, repeat=len(varlist)):
        result = expr
        for var, case in zip(varlist, cases, strict=True):
            if case is not None:
                result = z3.substitute(result, (var, case))
        yield result


def iter_closure_maps(expr: ExprRef) -> Iterator[ExprRef]:
    """Iterate over all closing abstractions, including variable coincidence."""
    if not free_vars(expr):
        yield expr
        return
    for group in iter_subsets(free_vars(expr)):
        if not group:
            continue
        var = min(group, key=str)
        abstracted = expr
        for other in group - {var}:
            abstracted = z3.substitute(abstracted, (other, var))
        abstracted = lam(var, abstracted)
        # logger.debug(f"Abstracted {expr} -> {abstracted}")
        assert len(free_vars(abstracted)) < len(free_vars(expr))
        yield from iter_closure_maps(abstracted)


def iter_closures(expr: ExprRef, *, trivial: bool = False) -> Iterator[ExprRef]:
    assert is_eq_or_leq(expr), expr
    if not free_vars(expr):
        yield expr
        return
    for expr2 in iter_eta_substitutions(expr):
        if trivial:
            for expr3 in iter_trivial_substitutions(expr2):
                yield from iter_closure_maps(expr3)
        else:
            yield from iter_closure_maps(expr2)


# TODO consider defaulting to trivial=True.
def QEHindley(formula: ExprRef, *, trivial: bool = False) -> set[ExprRef]:
    """
    Converts a z3.ForAll formula into a set of closed formulas.

    This eliminates the quantifiers using Hindley's extensionality trick [1] and
    a Curry-style combinatory abstraction algorithm. This works only for
    positive relations: equality and LEQ.

    [1] Roger Hindley (1967) "Axioms for strong reduction in combinatory logic"

    Args:
        formula: A z3 quantified formula.
        trivial: Include trivial substitutions [x/TOP] and [x/BOT].
    """
    assert formula.sort() == z3.BoolSort()

    # Check whether the formula is a quantified equality.
    result: set[ExprRef] = set()
    if not z3.is_quantifier(formula) or not formula.is_forall():
        return result
    body = formula.body()
    if not is_eq_or_leq(body):
        return result
    lhs, rhs = body.children()
    if not lhs.sort() == Term or not rhs.sort() == Term:
        return result

    # Convert to a unique set of formulas.
    for expr in iter_closures(body, trivial=trivial):
        assert is_eq_or_leq(expr), expr
        lhs, rhs = expr.children()
        if z3.eq(lhs, rhs):
            continue  # Skip trivial equalities.
        result.add(expr)
    return result
