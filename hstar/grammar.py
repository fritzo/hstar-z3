"""
# Linear-normal-forms for λ-join-calculus.

Our behavior synthesis search grammar will be a subset of the λ-join-calculus,
namely those terms that are in a particular linear normal form, i.e. that are
simplified wrt a set of rewrite rules.
"""

from dataclasses import dataclass
from enum import Enum
from functools import cache

from immutables import Map

from .util import HashConsMeta, intern


class TermType(Enum):
    TOP = 0  # by contrast BOT is simply a nullary JOIN
    APP = 1
    ABS0 = 2  # zero occurrences of the bound variable
    ABS1 = 3  # one occurrence of the bound variable
    ABS = 4  # two or more occurrences of the bound variable
    VAR = 5


EMPTY_VARS: Map[int, int] = Map()


@dataclass(frozen=True, slots=True)
class Term(metaclass=HashConsMeta):
    """Linear normal form."""

    # Data.
    typ: TermType
    varname: int = 0  # For VAR.
    head: "JoinTerm | None" = None
    body: "JoinTerm | None" = None
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS


@dataclass(frozen=True, slots=True)
class JoinTerm(metaclass=HashConsMeta):
    """Join of terms in the Scott lattice."""

    # Data.
    args: frozenset[Term]
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS


@cache
def _JOIN(args: frozenset[Term]) -> JoinTerm:
    """Join of terms."""
    # Eagerly linearly reduce.
    if _TOP in args:
        args = frozenset({_TOP})
    return JoinTerm(
        args,
        free_vars=max_vars(*(a.free_vars for a in args)),
    )


def JOIN(*args: Term) -> JoinTerm:
    """Join of terms."""
    return _JOIN(frozenset(args))


_TOP = Term(TermType.TOP)
TOP: JoinTerm = _JOIN(_TOP)
"""Top element of the Scott lattice."""
BOT: JoinTerm = _JOIN()
"""Bottom element of the Scott lattice."""


@cache
def _APP(a: Term, b: JoinTerm) -> JoinTerm:
    """Application."""
    # Eagerly linearly reduce.
    if a is _TOP:
        return TOP
    if a.typ == TermType.ABS0:
        return a.head
    if a.typ == TermType.ABS1:
        return subst(a.head, 0, b)
    # Construct.
    return Term(
        TermType.APP,
        head=a,
        body=b,
        free_vars=add_vars(a.free_vars, b.free_vars),
    )


@cache
def APP(a: JoinTerm, b: JoinTerm) -> JoinTerm:
    """Application."""
    return JOIN(*(_APP(ai, b) for ai in a.args))


def _ABS0(a: Term) -> Term:
    """Constant function."""
    assert a.typ != TermType.TOP, "use LAM instead"
    # Construct.
    return Term(
        TermType.ABS0,
        head=a,
        free_vars=a.free_vars,
    )


@cache
def _ABS1(a: Term) -> Term:
    """Linear function."""
    assert a.typ != TermType.TOP, "use LAM instead"
    assert a.free_vars.get(0, 0) == 1, "use LAM instead"
    # Construct.
    return Term(TermType.ABS1, head=a, free_vars=a.free_vars)


@cache
def _ABS(a: Term) -> Term:
    """Nonlinear function."""
    assert a.typ != TermType.TOP, "use LAM instead"
    assert a.free_vars.get(0, 0) > 1, "use LAM instead"
    # Construct.
    free_vars = intern(Map({k - 1: v for k, v in a.free_vars.items() if k}))
    return Term(TermType.ABS, head=a, free_vars=free_vars)


def _LAM(a: Term) -> Term:
    """Lambda abstraction."""
    occurrences = a.free_vars.get(0, 0)
    if occurrences == 0:
        return _ABS0(a)
    if occurrences == 1:
        return _ABS1(a)
    return _ABS(a)


def LAM(a: JoinTerm) -> JoinTerm:
    """Lambda abstraction."""
    # Eagerly linearly reduce.
    if a is TOP:
        return TOP
    # Construct.
    return _LAM(frozenset(_LAM(ai) for ai in a.args))


@cache
def VAR(v: int) -> JoinTerm:
    """Anonymous substitution variable."""
    assert v >= 0
    return JOIN(Term(TermType.VAR, varname=v, free_vars=Map({v: 1})))


@cache
def add_vars(a: Map[int, int], b: Map[int, int]) -> Map[int, int]:
    """Add two maps of variables."""
    result = dict(a)
    for k, v in b.items():
        result[k] += v
    return Map(result)


@cache
def max_vars(*args: Map[int, int]) -> Map[int, int]:
    """Element-wise maximum of two maps of variables."""
    result = dict(args[0])
    for a in args[1:]:
        for k, v in a.items():
            result[k] = max(result.get(k, 0), v)
    return Map(result)


@cache
def shift(a: Term, min_var: int = 0) -> Term:
    """Increment all free VARs in a."""
    if all(v < min_var for v in a.free_vars):
        return a
    if a.typ == TermType.VAR:
        assert a.varname >= min_var
        return Term(TermType.VAR, varname=a.varname + 1)
    if a.typ == TermType.APP:
        head = shift(a.head, min_var)
        body = JOIN(*(shift(ai, min_var) for ai in a.body.args))
        return _APP(head, body)
    if a.typ == TermType.ABS0:
        return _ABS0(shift(a.head, min_var))
    if a.typ == TermType.ABS1:
        return _ABS1(shift(a.head, min_var + 1))
    if a.typ == TermType.ABS:
        return _ABS(shift(a.head, min_var + 1))
    raise ValueError(f"unexpected term type: {a.typ}")


@cache
def subst(a: Term, v: int, b: JoinTerm) -> JoinTerm:
    """Substitute a VAR v := b in a."""
    if a.free_vars.get(v, 0) == 0:
        return a
    if a.typ == TermType.VAR:
        assert a.varname == v
        return b
    if a.typ == TermType.APP:
        head = subst(a.head, v, b)
        body = JOIN(*(subst(ai, v, b) for ai in a.body.args))
        return APP(head, body)
    if a.typ == TermType.ABS0:
        return _ABS0(subst(a.head, v, b))
    if a.typ in (TermType.ABS1, TermType.ABS):
        b = shift(b)  # FIXME is min_var correct?
        return _LAM(subst(a.head, v, b))
    raise ValueError(f"unexpected term type: {a.typ}")
