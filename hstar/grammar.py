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

EMPTY_VARS: Map[int, int] = Map()


@cache
def max_vars(*args: Map[int, int]) -> Map[int, int]:
    """Element-wise maximum of two maps of variables."""
    if not args:
        return EMPTY_VARS
    result = dict(args[0])
    for a in args[1:]:
        for k, v in a.items():
            result[k] = max(result.get(k, 0), v)
    return Map(result)


@cache
def add_vars(a: Map[int, int], b: Map[int, int]) -> Map[int, int]:
    """Add two maps of variables."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return Map(result)


class TermType(Enum):
    TOP = 0  # by contrast BOT is simply a nullary JOIN
    APP = 1
    ABS0 = 2  # zero occurrences of the bound variable
    ABS1 = 3  # one occurrence of the bound variable
    ABS = 4  # two or more occurrences of the bound variable
    VAR = 5


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Term(metaclass=HashConsMeta):
    """Linear normal form."""

    # Data.
    typ: TermType
    varname: int = 0  # For VAR.
    head: "Term | None" = None
    body: "JoinTerm | None" = None
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS

    def __repr__(self) -> str:
        if self.typ == TermType.TOP:
            return "TOP"
        if self.typ == TermType.VAR:
            return f"VAR({self.varname})"
        if self.typ == TermType.APP:
            return f"APP({self.head}, {self.body})"
        if self.typ == TermType.ABS0:
            return f"ABS0({self.head})"
        if self.typ == TermType.ABS1:
            return f"ABS1({self.head})"
        if self.typ == TermType.ABS:
            return f"ABS({self.head})"
        raise ValueError(f"unexpected term type: {self.typ}")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JoinTerm(metaclass=HashConsMeta):
    """Join of terms in the Scott lattice."""

    # Data.
    parts: frozenset[Term]
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS

    def __repr__(self) -> str:
        return f"JOIN({', '.join(map(repr, self.parts))})"


def _JOIN(*parts: Term) -> JoinTerm:
    """Join of terms."""
    return _JOIN_cached(frozenset(parts))


@cache
def _JOIN_cached(parts: frozenset[Term]) -> JoinTerm:
    """Join of terms."""
    # Eagerly linearly reduce.
    if _TOP in parts and len(parts) > 1:
        parts = frozenset((_TOP,))
    return JoinTerm(
        parts,
        free_vars=max_vars(*(a.free_vars for a in parts)),
    )


def JOIN(*args: JoinTerm) -> JoinTerm:
    """Join of terms."""
    parts: list[Term] = []
    for a in args:
        parts.extend(a.parts)
    return _JOIN(*parts)


_TOP = Term(TermType.TOP)
TOP: JoinTerm = _JOIN(_TOP)
"""Top element of the Scott lattice."""
BOT: JoinTerm = JOIN()
"""Bottom element of the Scott lattice."""


@cache
def _APP(a: Term, b: JoinTerm) -> JoinTerm:
    """Application."""
    # Eagerly linearly reduce.
    if a is _TOP:
        return TOP
    if a.typ == TermType.ABS0:
        assert a.head is not None
        return _JOIN(a.head)
    if a.typ == TermType.ABS1:
        assert a.head is not None
        return _subst(a.head, 0, b)
    # Construct.
    arg = Term(
        TermType.APP,
        head=a,
        body=b,
        free_vars=add_vars(a.free_vars, b.free_vars),
    )
    return _JOIN(arg)


@cache
def APP(a: JoinTerm, b: JoinTerm) -> JoinTerm:
    """Application."""
    args: list[Term] = []
    for ai in a.parts:
        args.extend(_APP(ai, b).parts)
    return _JOIN(*args)


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
    return _JOIN(*(_LAM(ai) for ai in a.parts))


@cache
def VAR(v: int) -> JoinTerm:
    """Anonymous substitution variable."""
    assert v >= 0
    return _JOIN(Term(TermType.VAR, varname=v, free_vars=Map({v: 1})))


@cache
def _shift(a: Term, start: int = 0) -> Term:
    """Increment all free VARs in a."""
    if all(v < start for v in a.free_vars):
        return a
    if a.typ == TermType.VAR:
        assert a.varname >= start
        return Term(
            TermType.VAR, varname=a.varname + 1, free_vars=Map({a.varname + 1: 1})
        )
    if a.typ == TermType.APP:
        assert a.head is not None
        assert a.body is not None
        head = _shift(a.head, start)
        body = shift(a.body, start)  # Use shift on the body instead of substitution
        parts = _APP(head, body).parts
        assert len(parts) == 1
        return next(iter(parts))
    if a.typ == TermType.ABS0:
        assert a.head is not None
        return _ABS0(_shift(a.head, start))
    if a.typ == TermType.ABS1:
        assert a.head is not None
        return _ABS1(_shift(a.head, start + 1))
    if a.typ == TermType.ABS:
        assert a.head is not None
        return _ABS(_shift(a.head, start + 1))
    raise ValueError(f"unexpected term type: {a.typ}")


@cache
def _subst(a: Term, v: int, b: JoinTerm) -> JoinTerm:
    """Substitute a VAR v := b in a."""
    if a.free_vars.get(v, 0) == 0:
        return _JOIN(a)
    if a.typ == TermType.VAR:
        assert a.varname == v
        return b
    if a.typ == TermType.APP:
        assert a.head is not None
        assert a.body is not None
        head = _subst(a.head, v, b)
        body = subst(a.body, v, b)
        return APP(head, body)
    if a.typ == TermType.ABS0:
        assert a.head is not None
        head_subst = _subst(a.head, v + 1, shift(b))
        return LAM(head_subst)
    if a.typ in (TermType.ABS1, TermType.ABS):
        assert a.head is not None
        head_subst = _subst(a.head, v + 1, shift(b))
        return LAM(head_subst)
    raise ValueError(f"unexpected term type: {a.typ}")


@cache
def shift(a: JoinTerm, start: int = 0) -> JoinTerm:
    """Increment all free VARs in a JoinTerm."""
    return _JOIN(*(_shift(ai, start) for ai in a.parts))


@cache
def subst(a: JoinTerm, v: int, b: JoinTerm) -> JoinTerm:
    """Substitute a VAR v := b in a JoinTerm."""
    return JOIN(*(_subst(ai, v, b) for ai in a.parts))
