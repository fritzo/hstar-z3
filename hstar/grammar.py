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
    return intern(Map(result))


@cache
def add_vars(a: Map[int, int], b: Map[int, int]) -> Map[int, int]:
    """Add two maps of variables."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return intern(Map(result))


class TermType(Enum):
    TOP = 0  # by contrast BOT is simply a nullary JOIN
    APP = 1
    ABS0 = 2  # zero occurrences of the bound variable
    ABS1 = 3  # one occurrence of the bound variable
    ABS = 4  # two or more occurrences of the bound variable
    VAR = 5


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _Term(metaclass=HashConsMeta):
    """A join-free linear normal form."""

    # Data.
    typ: TermType
    varname: int = 0  # For VAR.
    head: "_Term | None" = None
    body: "Term | None" = None
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
class Term(metaclass=HashConsMeta):
    """A linear normal form."""

    # Data.
    parts: frozenset[_Term]
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS

    def __repr__(self) -> str:
        if not self.parts:
            return "BOT"
        if len(self.parts) == 1:
            return repr(next(iter(self.parts)))
        return f"JOIN({', '.join(sorted(map(repr, self.parts)))})"


def _JOIN(*parts: _Term) -> Term:
    """Join of terms."""
    return _JOIN_cached(frozenset(parts))


@cache
def _JOIN_cached(parts: frozenset[_Term]) -> Term:
    """Join of terms."""
    # Eagerly linearly reduce.
    if _TOP in parts and len(parts) > 1:
        parts = frozenset((_TOP,))
    free_vars = max_vars(*(a.free_vars for a in parts))
    return Term(parts, free_vars=free_vars)


def JOIN(*args: Term) -> Term:
    """Join of terms."""
    parts: list[_Term] = []
    for a in args:
        parts.extend(a.parts)
    return _JOIN(*parts)


_TOP = _Term(TermType.TOP)
TOP: Term = _JOIN(_TOP)
"""Top element of the Scott lattice."""
BOT: Term = JOIN()
"""Bottom element of the Scott lattice."""


@cache
def _APP(head: _Term, body: Term) -> Term:
    """Application."""
    # Eagerly linearly reduce.
    if head is _TOP:
        return TOP
    if head.typ == TermType.ABS0:
        assert head.head is not None
        return _JOIN(head.head)
    if head.typ == TermType.ABS1:
        assert head.head is not None
        # FIXME shift body, then shift result
        return _subst(head.head, Map({0: body}))
    # Construct.
    arg = _Term(
        TermType.APP,
        head=head,
        body=body,
        free_vars=add_vars(head.free_vars, body.free_vars),
    )
    return _JOIN(arg)


@cache
def APP(head: Term, body: Term) -> Term:
    """Application."""
    args: list[_Term] = []
    for part in head.parts:
        args.extend(_APP(part, body).parts)
    return _JOIN(*args)


def _ABS0(head: _Term) -> _Term:
    """
    Constant function, whose bound variable does not occur.
    This is an optimization allowing the `head` term to avoid shifting.
    """
    assert head.typ != TermType.TOP, "use LAM instead"
    # Construct.
    return _Term(TermType.ABS0, head=head, free_vars=head.free_vars)


@cache
def _ABS1(head: _Term) -> _Term:
    """Linear function, whose bound variable occurs exactly once."""
    assert head.typ != TermType.TOP, "use LAM instead"
    assert head.free_vars.get(0, 0) == 1, "use LAM instead"
    # Construct.
    free_vars = intern(Map({k - 1: v for k, v in head.free_vars.items() if k}))
    return _Term(TermType.ABS1, head=head, free_vars=free_vars)


@cache
def _ABS(head: _Term) -> _Term:
    """Nonlinear function, whose bound variable occurs at least twice."""
    assert head.typ != TermType.TOP, "use LAM instead"
    assert head.free_vars.get(0, 0) > 1, "use LAM instead"
    # Construct.
    free_vars = intern(Map({k - 1: v for k, v in head.free_vars.items() if k}))
    return _Term(TermType.ABS, head=head, free_vars=free_vars)


def _LAM(head: _Term) -> _Term:
    """Lambda abstraction."""
    occurrences = head.free_vars.get(0, 0)
    if occurrences == 0:
        # Unshift head since _ABS0 is designed to skip shifting
        unshifted_head = _shift(head, start=1, delta=-1)
        return _ABS0(unshifted_head)
    if occurrences == 1:
        return _ABS1(head)
    return _ABS(head)


def LAM(head: Term) -> Term:
    """Lambda abstraction."""
    # Eagerly linearly reduce.
    if head is TOP:
        return TOP
    # Construct.
    return _JOIN(*(_LAM(part) for part in head.parts))


@cache
def _VAR(varname: int) -> _Term:
    """Anonymous substitution variable for de Bruijn indexing."""
    assert varname >= 0
    free_vars = intern(Map({varname: 1}))
    return _Term(TermType.VAR, varname=varname, free_vars=free_vars)


@cache
def VAR(varname: int) -> Term:
    """Anonymous substitution variable for de Bruijn indexing."""
    assert varname >= 0
    return _JOIN(_VAR(varname))


Env = Map[int, Term]
EMPTY_ENV: Env = Map()


@cache
def _shift(term: _Term, start: int = 0, delta: int = 1) -> _Term:
    """
    Shift all free VARs in term by delta.

    Increments (delta > 0) or decrements (delta < 0) all free variables
    with indices >= start by abs(delta).
    """
    if all(v < start for v in term.free_vars):
        return term
    if term.typ == TermType.VAR:
        assert term.varname >= start
        varname = term.varname + delta
        free_vars = intern(Map({varname: 1}))
        return _Term(TermType.VAR, varname=varname, free_vars=free_vars)
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head = _shift(term.head, start, delta)
        body = shift(term.body, start, delta)
        parts = _APP(head, body).parts
        assert len(parts) == 1
        return next(iter(parts))
    if term.typ == TermType.ABS0:
        assert term.head is not None
        return _ABS0(_shift(term.head, start, delta))
    if term.typ == TermType.ABS1:
        assert term.head is not None
        return _ABS1(_shift(term.head, start + 1, delta))
    if term.typ == TermType.ABS:
        assert term.head is not None
        return _ABS(_shift(term.head, start + 1, delta))
    raise ValueError(f"unexpected term type: {term.typ}")


@cache
def shift(term: Term, start: int = 0, delta: int = 1) -> Term:
    """
    Shift all free VARs in a JoinTerm by delta.

    Increments (delta > 0) or decrements (delta < 0) all free variables
    with indices >= start by abs(delta).
    """
    return _JOIN(*(_shift(part, start, delta) for part in term.parts))


def env_shift(env: Env, start: int = 0, delta: int = 1) -> Env:
    """
    Shift all free VARs in an Env by delta.

    Increments (delta > 0) or decrements (delta < 0) all free variables
    with indices >= start by abs(delta).
    """
    result: dict[int, Term] = {}
    for k, v in env.items():
        if k >= start:
            k += delta
        v = shift(v, start, delta)
        result[k] = v
    return Map(result)


@cache
def _subst(term: _Term, env: Env) -> Term:
    """Substitute variables according to an environment mapping."""
    # Check if term has any free variables that are in the environment
    if not any(k in term.free_vars for k in env):
        return _JOIN(term)

    if term.typ == TermType.VAR:
        # If variable is in environment, return its substitution
        if term.varname in env:
            return env[term.varname]
        return _JOIN(term)

    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head = _subst(term.head, env)
        body = subst(term.body, env)
        return APP(head, body)

    if term.typ == TermType.ABS0:
        assert term.head is not None
        # ABS0 doesn't need environment shifting since the bound variable doesn't occur
        head_subst = _subst(term.head, env)
        return LAM(head_subst)

    if term.typ in (TermType.ABS1, TermType.ABS):
        assert term.head is not None
        # Shift all terms in environment for lambda body
        shifted_env = env_shift(env)
        head_subst = _subst(term.head, shifted_env)
        return LAM(head_subst)

    raise ValueError(f"unexpected term type: {term.typ}")


@cache
def subst(term: Term, env: Env) -> Term:
    """Substitute variables according to an environment mapping in a JoinTerm."""
    if not env:
        return term
    return JOIN(*(_subst(part, env) for part in term.parts))


def _complexity(term: _Term) -> int:
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


def complexity(term: Term) -> int:
    """
    Complexity of a term.

    This satisfies the compositionality condition
    ```
    complexity(JOIN(lhs, rhs)) <= 1 + complexity(lhs) + complexity(rhs)
    ```
    But due to eager linear reduction, this does not satisfy the
    compositionality conditions
    ```
    complexity(APP(lhs, rhs)) <= 1 + complexity(lhs) + complexity(rhs)
    complexity(LAM(term)) <= 1 + complexity(term)
    ```
    because those may result in more complex JOIN terms. However it this does
    satisfy that any term can be constructed from strictly simpler terms.
    """
    if not term.parts:
        return 1
    return sum(map(_complexity, term.parts)) + len(term.parts) - 1


class Enumerator:
    """Generator for all terms, sorted by complexity, then repr."""

    def __init__(self) -> None:
        self._levels: list[set[Term]] = [set()]

    def __iter__(self) -> Iterator[Term]:
        for complexity in itertools.count():
            level = list(self._get_level(complexity))
            level.sort(key=repr)
            yield from level

    def _get_level(self, complexity: int) -> set[Term]:
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
        self._add_term(VAR(c - 1))

        # Add LAM terms.
        for term in self._levels[c - 1]:
            self._add_term(LAM(term))

        # Add APP and JOIN terms.
        for c_lhs in range(1, c - 1):
            c_rhs = c - c_lhs - 1
            for lhs in self._levels[c_lhs]:
                for rhs in self._levels[c_rhs]:
                    self._add_term(APP(lhs, rhs))
                    self._add_term(JOIN(lhs, rhs))

    def _add_term(self, term: Term) -> None:
        c = complexity(term)
        if c < len(self._levels):
            self._levels[c].add(term)
        # Otherwise linear reduction has produced a more complex term that we
        # will discard here but reconstruct later.


class UnificationFailure(Exception):
    """Raised when unification fails."""


def refines(special: Term, general: Term) -> bool:
    """
    Check whether a term `special` can be constructed from a term
    `general` merely by substituting free variables.
    """
    raise NotImplementedError("TODO")
