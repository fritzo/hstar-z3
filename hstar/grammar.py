"""
# Linear-normal-forms for λ-join-calculus.

Our behavior synthesis search grammar will be a subset of the λ-join-calculus,
namely those terms that are in a particular linear normal form, i.e. that are
simplified wrt a set of rewrite rules.

## Linear Reduction Rules

The following eager linear reductions are applied during term construction:

- Join reductions JOIN(...):
  - JOIN(TOP, any) → TOP (TOP absorbs other terms)
  - JOIN(BOT, t) → t (BOT is the identity element for JOIN)
  - JOIN(t, t) → t (idempotence)
  - JOIN is associative and commutative

- Lambda/abstraction reductions ABS(...):
  - ABS(TOP) → TOP (abstraction over TOP is TOP)
  - ABS(BOT) → BOT (abstraction over BOT is BOT)

- Beta reductions APP(ABS(...), ...):
  - APP(ABS(body), arg) → subst(body, [0 ↦ arg]) when:
    - The bound variable occurs at most once in body, or
    - arg is BOT, TOP, or a simple variable

These rules ensure terms are maintained in a canonical normal form,
which helps avoid redundant term exploration during synthesis.

Warning: eager linear reduction may increase term complexity. Hence we need to
take care when enumerating terms, allowing the complexity of a construction
`JOIN(...)`, `ABS(...)`, or `APP(...)` to more than the sum of their parts + 1.
"""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache

from immutables import Map

from .functools import weak_key_cache
from .hashcons import HashConsMeta, intern

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
    VAR = 1
    ABS = 2
    APP = 3


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

    @weak_key_cache
    def __repr__(self) -> str:
        if self.typ == TermType.TOP:
            return "TOP"
        if self.typ == TermType.VAR:
            return f"VAR({self.varname})"
        if self.typ == TermType.APP:
            return f"APP({repr(self.head)}, {repr(self.body)})"
        if self.typ == TermType.ABS:
            return f"ABS({repr(self.head)})"
        raise ValueError(f"unexpected term type: {self.typ}")

    @weak_key_cache
    def __str__(self) -> str:
        if self.typ == TermType.TOP:
            return "⊤"
        if self.typ == TermType.VAR:
            return str(self.varname)
        if self.typ == TermType.ABS:
            assert self.head is not None
            return f"λ {self.head}"
        if self.typ == TermType.APP:
            assert self.head is not None
            assert self.body is not None

            # Apply parentheses based on precedence
            head_str = str(self.head)
            if self.head.typ == TermType.ABS:
                head_str = f"({head_str})"

            body_str = str(self.body)
            # Add parentheses to body if it contains a join
            if len(self.body.parts) > 1:
                body_str = f"({body_str})"
            # Add parentheses to body if it's an application (right associativity)
            elif (
                len(self.body.parts) == 1
                and next(iter(self.body.parts)).typ == TermType.APP
            ):
                body_str = f"({body_str})"

            return f"{head_str} {body_str}"

        raise ValueError(f"unexpected term type: {self.typ}")

    def __lt__(self, other: "_Term") -> bool:
        self_key = (_complexity(self), repr(self))
        other_key = (_complexity(other), repr(other))
        return self_key < other_key


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Term(metaclass=HashConsMeta):
    """A linear normal form."""

    # Data.
    parts: frozenset[_Term]
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS

    @weak_key_cache
    def __repr__(self) -> str:
        if not self.parts:
            return "BOT"
        if len(self.parts) == 1:
            return repr(next(iter(self.parts)))
        return f"JOIN({', '.join(sorted(map(repr, self.parts)))})"

    @weak_key_cache
    def __str__(self) -> str:
        if not self.parts:
            return "⊥"
        if len(self.parts) == 1:
            return str(next(iter(self.parts)))

        # Sort parts for consistent output
        sorted_parts = sorted(self.parts)
        return " | ".join(str(part) for part in sorted_parts)

    def __lt__(self, other: "Term") -> bool:
        self_key = (complexity(self), repr(self))
        other_key = (complexity(other), repr(other))
        return self_key < other_key


def _JOIN(*parts: _Term) -> Term:
    """Join of terms."""
    parts_ = frozenset(parts)
    # Eagerly linearly reduce.
    if _TOP in parts_ and len(parts_) > 1:
        parts_ = frozenset((_TOP,))
    free_vars = max_vars(*(a.free_vars for a in parts_))
    return Term(parts_, free_vars=free_vars)


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


@weak_key_cache
def _ABS(head: _Term) -> _Term:
    """Lambda abstraction, binding de Bruijn variable `VAR(0)`."""
    assert head.typ != TermType.TOP, "use ABS instead"
    # Construct.
    free_vars = intern(Map({k - 1: v for k, v in head.free_vars.items() if k}))
    return _Term(TermType.ABS, head=head, free_vars=free_vars)


@weak_key_cache
def ABS(head: Term) -> Term:
    """Lambda abstraction, binding de Bruijn variable `VAR(0)`."""
    # Eagerly linearly reduce.
    if head is TOP:
        return TOP  # Since \x.TOP = TOP.
    # Construct.
    return _JOIN(*(_ABS(part) for part in head.parts))


@weak_key_cache
def _APP(head: _Term, body: Term) -> Term:
    """Application."""
    # Eagerly linearly reduce.
    if head is _TOP:
        return TOP
    if head.typ == TermType.ABS:
        assert head.head is not None
        if (
            head.head.free_vars.get(0, 0) <= 1
            or body is BOT
            or body is TOP
            or len(body.parts) == 1
            and next(iter(body.parts)).typ == TermType.VAR
        ):
            body = shift(body, delta=1)
            result = _subst(head.head, env=Env({0: body}))
            return shift(result, delta=-1)
    # Construct.
    arg = _Term(
        TermType.APP,
        head=head,
        body=body,
        free_vars=add_vars(head.free_vars, body.free_vars),
    )
    return _JOIN(arg)


@weak_key_cache
def APP(head: Term, body: Term) -> Term:
    """Application."""
    args: list[_Term] = []
    for part in head.parts:
        args.extend(_APP(part, body).parts)
    return _JOIN(*args)


def app(*args: Term) -> Term:
    """Chained application."""
    assert args
    result = args[0]
    for arg in args[1:]:
        result = APP(result, arg)
    return result


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Env(Mapping[int, Term], metaclass=HashConsMeta):
    """An environment mapping variables to terms."""

    _map: Map[int, Term]

    def __init__(self, *items: Mapping[int, Term] | Iterable[tuple[int, Term]]) -> None:
        object.__setattr__(self, "_map", intern(Map(*items)))

    def __getitem__(self, key: int) -> Term:
        return self._map[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    @weak_key_cache
    def __repr__(self) -> str:
        return f"Env({{{', '.join(f'{k}: {repr(v)}' for k, v in self.items())}}})"

    @weak_key_cache
    def __str__(self) -> str:
        if not self._map:
            return "{}"
        return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(self.items())) + "}"

    def __lt__(self, other: "Env") -> bool:
        self_key = (env_complexity(self), repr(self))
        other_key = (env_complexity(other), repr(other))
        return self_key < other_key


@lru_cache
def env_free_vars(env: Env) -> Map[int, int]:
    """Free variable counts in an environment."""
    result: Counter[int] = Counter()
    for v in env.values():
        result.update(v.free_vars)
    for k in env.keys():
        result.pop(k, None)
    return Map(result)


@weak_key_cache
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
    if term.typ == TermType.ABS:
        assert term.head is not None
        return _ABS(_shift(term.head, start=start + 1, delta=delta))
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head = _shift(term.head, start=start, delta=delta)
        body = shift(term.body, start=start, delta=delta)
        parts = _APP(head=head, body=body).parts
        assert len(parts) == 1
        return next(iter(parts))
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
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
    return Env(result)


@weak_key_cache
def _subst(term: _Term, env: Env) -> Term:
    """Substitute variables according to an environment mapping."""
    # Check if term has any free variables that are in the environment
    if not any(k in term.free_vars for k in env):
        return _JOIN(term)
    if term.typ == TermType.VAR:
        if term.varname in env:
            return env[term.varname]
        return _JOIN(term)
    if term.typ == TermType.ABS:
        assert term.head is not None
        head = _subst(term.head, env_shift(env))
        return ABS(head)
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head = _subst(term.head, env)
        body = subst(term.body, env)
        return APP(head, body)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
def subst(term: Term, env: Env) -> Term:
    """Substitute variables according to an environment mapping in a JoinTerm."""
    if not env:
        return term
    return JOIN(*(_subst(part, env) for part in term.parts))


@lru_cache
def env_compose(lhs: Env, rhs: Env) -> Env:
    """
    Compose two environments.

    `subst(term, env_compose(lhs, rhs)) == subst(subst(term, lhs), rhs)`
    """
    result: dict[int, Term] = {}
    for k, v in lhs.items():
        result[k] = subst(v, rhs)
    for k, v in rhs.items():
        if k not in lhs:
            result[k] = v
    return Env(result)


@weak_key_cache
def _complexity(term: _Term) -> int:
    """Complexity of a term."""
    if term.typ == TermType.TOP:
        return 1
    if term.typ == TermType.VAR:
        return 1 + term.varname
    if term.typ == TermType.ABS:
        assert term.head is not None
        return 1 + _complexity(term.head)
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        return 1 + _complexity(term.head) + complexity(term.body)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
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
    complexity(ABS(term)) <= 1 + complexity(term)
    ```
    because those may result in more complex JOIN terms. However it this does
    satisfy that any term can be constructed from strictly simpler terms.
    """
    if not term.parts:
        return 1
    return sum(map(_complexity, term.parts)) + len(term.parts) - 1


@lru_cache
def env_complexity(env: Env) -> int:
    """Complexity of an environment."""
    return sum(complexity(term) for term in env.values())


@lru_cache
def subst_complexity(free_vars: Map[int, int], env: Env) -> int:
    """
    Complexity of a substitution.

    This accounts for change in complexity due to substituting terms for free
    variables at multiple locations, but ignores eager linear reduction.
    """
    result = 0
    for k, v in env.items():
        count = free_vars.get(k, 0)
        result += count * (complexity(v) - complexity(VAR(k)))
    return result


@weak_key_cache
def _is_deterministic(term: _Term) -> bool:
    """Returns whether a term is deterministic."""
    if term.typ == TermType.TOP:
        return False
    if term.typ == TermType.VAR:
        return True
    if term.typ == TermType.ABS:
        assert term.head is not None
        return _is_deterministic(term.head)
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        return _is_deterministic(term.head) and is_deterministic(term.body)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
def is_deterministic(term: Term) -> bool:
    """
    Returns whether a term is deterministic, i.e. whether it is definable from
    the pure λ-calculus without JOIN and TOP.
    """
    if not term.parts:
        return True  # BOT is definable as e.g. (λx.x x) (λx.x x).
    if term is TOP or len(term.parts) > 1:
        return False
    return all(map(_is_deterministic, term.parts))


@weak_key_cache
def _is_normal(term: _Term) -> bool:
    """Returns whether a term is in beta normal form."""
    if term.typ == TermType.TOP:
        return True
    if term.typ == TermType.VAR:
        return True
    if term.typ == TermType.ABS:
        assert term.head is not None
        if term.head.typ == TermType.ABS:
            return False  # unreduced beta redex
        return _is_normal(term.head)
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        return _is_normal(term.head) and is_normal(term.body)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
def is_normal(term: Term) -> bool:
    """Returns whether a term is in beta normal form."""
    return all(map(_is_normal, term.parts))
