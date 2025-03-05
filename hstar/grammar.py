"""
# Linear-normal-forms for λ-join-calculus.

Our behavior synthesis search grammar will be a subset of the λ-join-calculus,
namely those terms that are in a particular linear normal form, i.e. that are
simplified wrt a set of rewrite rules.
"""

import itertools
from collections.abc import Iterator
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
        if not self.parts:
            return "BOT"
        if len(self.parts) == 1:
            return repr(next(iter(self.parts)))
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
def _APP(head: Term, body: JoinTerm) -> JoinTerm:
    """Application."""
    # Eagerly linearly reduce.
    if head is _TOP:
        return TOP
    if head.typ == TermType.ABS0:
        assert head.head is not None
        return _JOIN(head.head)
    if head.typ == TermType.ABS1:
        assert head.head is not None
        return _subst(head.head, 0, body)
    # Construct.
    arg = Term(
        TermType.APP,
        head=head,
        body=body,
        free_vars=add_vars(head.free_vars, body.free_vars),
    )
    return _JOIN(arg)


@cache
def APP(head: JoinTerm, body: JoinTerm) -> JoinTerm:
    """Application."""
    args: list[Term] = []
    for part in head.parts:
        args.extend(_APP(part, body).parts)
    return _JOIN(*args)


def _ABS0(head: Term) -> Term:
    """Constant function."""
    assert head.typ != TermType.TOP, "use LAM instead"
    # Construct.
    return Term(
        TermType.ABS0,
        head=head,
        free_vars=head.free_vars,
    )


@cache
def _ABS1(head: Term) -> Term:
    """Linear function."""
    assert head.typ != TermType.TOP, "use LAM instead"
    assert head.free_vars.get(0, 0) == 1, "use LAM instead"
    # Construct.
    return Term(TermType.ABS1, head=head, free_vars=head.free_vars)


@cache
def _ABS(head: Term) -> Term:
    """Nonlinear function."""
    assert head.typ != TermType.TOP, "use LAM instead"
    assert head.free_vars.get(0, 0) > 1, "use LAM instead"
    # Construct.
    free_vars = intern(Map({k - 1: v for k, v in head.free_vars.items() if k}))
    return Term(TermType.ABS, head=head, free_vars=free_vars)


def _LAM(head: Term) -> Term:
    """Lambda abstraction."""
    occurrences = head.free_vars.get(0, 0)
    if occurrences == 0:
        return _ABS0(head)
    if occurrences == 1:
        return _ABS1(head)
    return _ABS(head)


def LAM(head: JoinTerm) -> JoinTerm:
    """Lambda abstraction."""
    # Eagerly linearly reduce.
    if head is TOP:
        return TOP
    # Construct.
    return _JOIN(*(_LAM(part) for part in head.parts))


@cache
def _VAR(varname: int) -> Term:
    """Anonymous substitution variable."""
    assert varname >= 0
    return Term(TermType.VAR, varname=varname, free_vars=Map({varname: 1}))


@cache
def VAR(varname: int) -> JoinTerm:
    """Anonymous substitution variable."""
    assert varname >= 0
    return _JOIN(_VAR(varname))


@cache
def _shift(term: Term, start: int = 0) -> Term:
    """Increment all free VARs in a."""
    if all(v < start for v in term.free_vars):
        return term
    if term.typ == TermType.VAR:
        assert term.varname >= start
        return Term(
            TermType.VAR, varname=term.varname + 1, free_vars=Map({term.varname + 1: 1})
        )
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head = _shift(term.head, start)
        body = shift(term.body, start)  # Use shift on the body instead of substitution
        parts = _APP(head, body).parts
        assert len(parts) == 1
        return next(iter(parts))
    if term.typ == TermType.ABS0:
        assert term.head is not None
        return _ABS0(_shift(term.head, start))
    if term.typ == TermType.ABS1:
        assert term.head is not None
        return _ABS1(_shift(term.head, start + 1))
    if term.typ == TermType.ABS:
        assert term.head is not None
        return _ABS(_shift(term.head, start + 1))
    raise ValueError(f"unexpected term type: {term.typ}")


@cache
def _subst(term: Term, old: int, new: JoinTerm) -> JoinTerm:
    """Substitute a VAR v := b in a."""
    if term.free_vars.get(old, 0) == 0:
        return _JOIN(term)
    if term.typ == TermType.VAR:
        assert term.varname == old
        return new
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head = _subst(term.head, old, new)
        body = subst(term.body, old, new)
        return APP(head, body)
    if term.typ == TermType.ABS0:
        assert term.head is not None
        head_subst = _subst(term.head, old + 1, shift(new))
        return LAM(head_subst)
    if term.typ in (TermType.ABS1, TermType.ABS):
        assert term.head is not None
        head_subst = _subst(term.head, old + 1, shift(new))
        return LAM(head_subst)
    raise ValueError(f"unexpected term type: {term.typ}")


@cache
def shift(term: JoinTerm, start: int = 0) -> JoinTerm:
    """Increment all free VARs in a JoinTerm."""
    return _JOIN(*(_shift(part, start) for part in term.parts))


@cache
def subst(term: JoinTerm, old: int, new: JoinTerm) -> JoinTerm:
    """Substitute a VAR v := b in a JoinTerm."""
    return JOIN(*(_subst(part, old, new) for part in term.parts))


def _complexity(term: Term) -> int:
    """Complexity of a term."""
    if term.typ == TermType.TOP:
        return 1
    if term.typ == TermType.VAR:
        return 1 + term.varname
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        return 1 + _complexity(term.head) + complexity(term.body)
    if term.typ == TermType.ABS0:
        assert term.head is not None
        return 1 + _complexity(term.head)
    if term.typ == TermType.ABS1:
        assert term.head is not None
        return 1 + _complexity(term.head)
    if term.typ == TermType.ABS:
        assert term.head is not None
        return 1 + _complexity(term.head)
    raise ValueError(f"unexpected term type: {term.typ}")


def complexity(term: JoinTerm) -> int:
    """Complexity of a term."""
    if not term.parts:
        return 1
    if len(term.parts) == 1:
        return _complexity(next(iter(term.parts)))
    return 1 + max(_complexity(part) for part in term.parts)


class Enumerator:
    """Generator for all terms, sorted by complexity, then repr."""

    def __init__(self) -> None:
        self._levels: list[set[JoinTerm]] = [set()]

    def __iter__(self) -> Iterator[JoinTerm]:
        for complexity in itertools.count():
            level = list(self._get_level(complexity))
            level.sort(key=repr)
            yield from level

    def _get_level(self, complexity: int) -> set[JoinTerm]:
        while len(self._levels) <= complexity:
            self._add_level()
        return self._levels[complexity]

    def _add_level(self) -> None:
        self._levels.append(set())
        c = len(self._levels) - 1

        # Add nullary terms.
        if c == 1:
            self._add_term(BOT)
            self._add_term(TOP)
        # complexity(VAR(n)) = 1 + n
        self._add_term(VAR(c - 1))

        # Add LAM terms.
        # complexity(LAM(term)) <= 1 + complexity(term)
        for term in self._get_level(c - 1):
            self._add_term(LAM(term))

        # Add APP terms.
        # complexity(APP(lhs, rhs)) <= 1 + complexity(lhs) + complexity(rhs)
        for c_lhs in range(1, c):
            c_rhs = c - c_lhs - 1
            lhs_level = self._get_level(c_lhs)
            rhs_level = self._get_level(c_rhs)
            for lhs, rhs in itertools.product(lhs_level, rhs_level):
                self._add_term(APP(lhs, rhs))

        # Add JOIN terms.
        # complexity(JOIN(lhs, rhs)) <= 1 + max(complexity(lhs), complexity(rhs))
        lhs_level = self._get_level(c - 1)
        for c_rhs in range(1, c):
            rhs_level = self._get_level(c_rhs)
            for lhs, rhs in itertools.product(lhs_level, rhs_level):
                self._add_term(JOIN(lhs, rhs))

    def _add_term(self, term: JoinTerm) -> None:
        c = complexity(term)
        assert c < len(self._levels)
        self._levels[c].add(term)
