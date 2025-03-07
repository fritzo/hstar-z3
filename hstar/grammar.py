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

from .util import HashConsMeta, intern, partitions

EMPTY_VARS: Map[int, int] = Map()


@cache
def max_vars(*args: Map[int, int]) -> Map[int, int]:
    """Element-wise maximum of multiple maps of variables."""
    if not args:
        return EMPTY_VARS
    result = dict(args[0])
    for arg in args[1:]:
        for k, v in arg.items():
            result[k] = max(result.get(k, 0), v)
    return intern(Map(result))


@cache
def add_vars(*args: Map[int, int]) -> Map[int, int]:
    """Add multiple maps of variables."""
    result = dict(args[0])
    for arg in args[1:]:
        for k, v in arg.items():
            result[k] = result.get(k, 0) + v
    return intern(Map(result))


class TermType(Enum):
    TOP = 0  # by contrast BOT is simply a nullary JOIN
    APP = 1
    LIN = 2  # one occurrence of the bound variable
    ABS = 3  # two or more occurrences of the bound variable
    VAR = 4


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
        if self.typ == TermType.LIN:
            return f"LIN({self.head})"
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
    if head.typ == TermType.LIN:
        assert head.head is not None
        # Shift body up to avoid variable capture, then substitute
        body = shift(body, delta=1)
        result = _subst(head.head, env=Map({0: body}))
        # Shift result down to account for the removed lambda
        return shift(result, delta=-1)
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


@cache
def _LIN(head: _Term) -> _Term:
    """Linear function, whose bound variable occurs at most once."""
    assert head.typ != TermType.TOP, "use LAM instead"
    assert head.free_vars.get(0, 0) <= 1, "use LAM instead"
    # Construct.
    free_vars = intern(Map({k - 1: v for k, v in head.free_vars.items() if k}))
    return _Term(TermType.LIN, head=head, free_vars=free_vars)


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
    return _LIN(head) if occurrences <= 1 else _ABS(head)


def LAM(head: Term) -> Term:
    """Lambda abstraction."""
    # Eagerly linearly reduce.
    if head is TOP:
        return TOP  # Since \x.TOP = TOP.
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
def _shift(term: _Term, *, start: int = 0, delta: int = 1) -> _Term:
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
        head = _shift(term.head, start=start, delta=delta)
        body = shift(term.body, start=start, delta=delta)
        parts = _APP(head=head, body=body).parts
        assert len(parts) == 1
        return next(iter(parts))
    if term.typ == TermType.LIN:
        assert term.head is not None
        return _LIN(_shift(term.head, start=start + 1, delta=delta))
    if term.typ == TermType.ABS:
        assert term.head is not None
        return _ABS(_shift(term.head, start=start + 1, delta=delta))
    raise ValueError(f"unexpected term type: {term.typ}")


@cache
def shift(term: Term, *, start: int = 0, delta: int = 1) -> Term:
    """
    Shift all free VARs in a JoinTerm by delta.

    Increments (delta > 0) or decrements (delta < 0) all free variables
    with indices >= start by abs(delta).
    """
    return _JOIN(*(_shift(part, start=start, delta=delta) for part in term.parts))


def env_shift(env: Env, *, start: int = 0, delta: int = 1) -> Env:
    """
    Shift all free VARs in an Env by delta.

    Increments (delta > 0) or decrements (delta < 0) all free variables
    with indices >= start by abs(delta).
    """
    result: dict[int, Term] = {}
    for k, v in env.items():
        if k >= start:
            k += delta
        v = shift(v, start=start, delta=delta)
        result[k] = v
    return intern(Map(result))


@cache
def _subst(term: _Term, env: Env) -> Term:
    """Substitute variables according to an environment mapping."""
    # Check if term has any free variables that are in the environment
    if not any(k in term.free_vars for k in env):
        return _JOIN(term)

    if term.typ == TermType.VAR:
        if term.varname in env:
            return env[term.varname]
        return _JOIN(term)

    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head = _subst(term.head, env)
        body = subst(term.body, env)
        return APP(head, body)

    if term.typ in (TermType.LIN, TermType.ABS):
        assert term.head is not None
        head = _subst(term.head, env_shift(env))
        return LAM(head)

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
    if term.typ == TermType.LIN:
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


def env_complexity(env: Env) -> int:
    """Complexity of an environment."""
    return sum(complexity(term) for term in env.values())


class Enumerator:
    """Generator for all terms, sorted by complexity, then repr."""

    def __init__(self) -> None:
        self._levels: list[set[Term]] = [set()]

    def __iter__(self) -> Iterator[Term]:
        for complexity in itertools.count():
            yield from self.level(complexity)

    def level(self, c: int) -> Iterator[Term]:
        """Iterate over terms of a given complexity."""
        return iter(sorted(self._get_level(c), key=repr))

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

        # Add unary terms.
        for term in self._levels[c - 1]:
            self._add_term(LAM(term))

        # Add binary terms.
        for c_lhs in range(1, c - 1):
            c_rhs = c - c_lhs - 1
            for lhs in self._levels[c_lhs]:
                for rhs in self._levels[c_rhs]:
                    self._add_term(APP(lhs, rhs))
                    self._add_term(JOIN(lhs, rhs))

    def _add_term(self, term: Term) -> None:
        c = complexity(term)
        if c >= len(self._levels):
            # Eager linear reduction has produced a more complex term that we
            # will discard here but reconstruct later.
            return
        self._levels[c].add(term)


class EnvEnumerator:
    """Generator for all environments, sorted by complexity, then repr."""

    def __init__(self, keys: frozenset[int]) -> None:
        self._keys = keys
        self._enumerator = Enumerator()
        self._levels: list[set[Env]] = [set()]

    def __iter__(self) -> Iterator[Env]:
        for complexity in itertools.count():
            yield from self.level(complexity)

    def level(self, c: int) -> Iterator[Env]:
        """Iterate over environments of a given complexity."""
        return iter(sorted(self._get_level(c), key=repr))

    def _get_level(self, complexity: int) -> set[Env]:
        while len(self._levels) <= complexity:
            self._add_level()
        return self._levels[complexity]

    def _add_level(self) -> None:
        self._levels.append(set())
        c = len(self._levels) - 1

        # Iterate over all partitions of c into len(self._keys) parts.
        for partition in partitions(c, len(self._keys)):
            factors = [self._enumerator.level(p) if p else [None] for p in partition]
            for cs in itertools.product(*factors):
                env = Map(
                    (k, v)
                    for k, v in zip(self._keys, cs, strict=True)
                    if v is not None
                    if v is not VAR(k)
                )
                self._levels[c].add(env)


class RefinementEnumerator:
    """
    Generator for all refinements of a sketch, approximately sorted by complexity.

    Each yielded refinement substitutes terms for free variables in the sketch.
    """

    def __init__(self, sketch: Term) -> None:
        self._sketch = sketch
        self._env_enumerator = EnvEnumerator(frozenset(sketch.free_vars))

    def __iter__(self) -> Iterator[Term]:
        seen: set[Term] = set()
        for env in self._env_enumerator:
            term = subst(self._sketch, env)
            if term in seen:
                continue
            seen.add(term)
            yield term


class UnificationFailure(Exception):
    """Raised when unification fails."""


def refines(special: Term, general: Term) -> bool:
    """
    Check whether a term `special` can be constructed from a term
    `general` merely by substituting free variables.
    """
    raise NotImplementedError("TODO")
