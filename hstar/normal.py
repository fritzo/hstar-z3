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
  - ABS(APP(lhs), VAR(0)) -> shift(lhs, delta=-1)
    if VAR(0) does not occur in lhs. (η-conversion)

- Beta reductions APP(ABS(...), ...):
  - APP(ABS(body), arg) → subst(body, [0 ↦ arg]) when:
    - The bound variable occurs at most once in body, or
    - arg is BOT, TOP, or VAR(...). Note it would be safe to widen this
      condition from can_safely_copy(arg) to is_linear(arg), but that would push
      many nonlinear terms to higher complexity, as can be measured by
      examples/terms.ipynb.

These rules ensure terms are maintained in a canonical normal form,
which helps avoid redundant term exploration during synthesis.

Warning: eager linear reduction may increase term complexity. Hence we need to
take care when enumerating terms, allowing the complexity of a construction
`JOIN(...)`, `ABS(...)`, or `APP(...)` to more than the sum of their parts + 1.
"""

import itertools
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache

from immutables import Map

from hstar.itertools import partitions

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


class Precedence(Enum):
    APP_HEAD = 0
    APP_BODY = 1
    ABS = 2
    JOIN = 3


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

    def __str__(self) -> str:
        return self.pretty(Precedence.JOIN)

    @weak_key_cache
    def pretty(self, context: Precedence) -> str:
        if self.typ == TermType.TOP:
            return "⊤"
        if self.typ == TermType.VAR:
            return str(self.varname)
        if self.typ == TermType.ABS:
            assert self.head is not None
            result = self.head.pretty(Precedence.ABS)
            result = f"λ {result}"
            if context == Precedence.APP_HEAD or context == Precedence.APP_BODY:
                result = f"({result})"
            return result
        if self.typ == TermType.APP:
            assert self.head is not None
            assert self.body is not None
            head_str = self.head.pretty(Precedence.APP_HEAD)
            body_str = self.body.pretty(Precedence.APP_BODY)
            result = f"{head_str} {body_str}"
            if context == Precedence.APP_BODY:
                result = f"({result})"
            return result

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

    def __str__(self) -> str:
        return self.pretty(Precedence.JOIN)

    @weak_key_cache
    def pretty(self, context: Precedence) -> str:
        if not self.parts:
            return "⊥"
        if len(self.parts) == 1:
            part = next(iter(self.parts))
            return part.pretty(context)

        # Sort parts for consistent output
        sorted_parts = sorted(self.parts)
        result = " | ".join(part.pretty(Precedence.JOIN) for part in sorted_parts)
        if context != Precedence.JOIN:
            result = f"({result})"
        return result

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
    # Eagerly linearly reduce.
    if head is _TOP:
        return _TOP
    if head.typ == TermType.APP:
        # η-conversion: ABS(APP(lhs), VAR(0)) -> shift(lhs, delta=-1)
        assert head.head is not None
        assert head.body is not None
        if head.body is VAR(0) and not head.head.free_vars.get(0):
            return _shift(head.head, delta=-1)
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


def can_safely_copy(body: Term) -> bool:
    """Check whether a term can be safely copied."""
    if body is TOP or body is BOT:
        return True
    if len(body.parts) > 1:
        return False
    if next(iter(body.parts)).typ == TermType.VAR:
        # Note if varname > 0, complexity will increase.
        return True
    return False


@weak_key_cache
def _APP(head: _Term, body: Term) -> Term:
    """Application."""
    # Eagerly linearly reduce.
    if head is _TOP:
        return TOP
    if head.typ == TermType.ABS:
        assert head.head is not None
        if head.head.free_vars.get(0, 0) <= 1 or can_safely_copy(body):
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


def _is_closed(term: _Term) -> bool:
    """Returns whether a term is closed, i.e. has no free variables."""
    return not term.free_vars


def is_closed(term: Term) -> bool:
    """Returns whether a term is closed, i.e. has no free variables."""
    return all(map(_is_closed, term.parts))


@weak_key_cache
def _is_linear(term: _Term) -> bool:
    """Returns whether a term is linear."""
    if term.typ == TermType.TOP:
        return True
    if term.typ == TermType.VAR:
        return True
    if term.typ == TermType.ABS:
        assert term.head is not None
        if any(v > 1 for v in term.head.free_vars.values()):
            return False
        return _is_linear(term.head)
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        return _is_linear(term.head) and is_linear(term.body)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
def is_linear(term: Term) -> bool:
    """Returns whether a term is linear."""
    return all(map(_is_linear, term.parts))


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
        return _is_normal(term.head)
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        if term.head.typ == TermType.ABS:
            return False  # unreduced beta redex
        return _is_normal(term.head) and is_normal(term.body)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
def is_normal(term: Term) -> bool:
    """Returns whether a term is in beta normal form."""
    return all(map(_is_normal, term.parts))


@weak_key_cache
def _leq(lhs: _Term, rhs: _Term) -> bool | None:
    """Returns whether lhs <= rhs, or None if the relation is unknown."""
    both_normal = _is_normal(lhs) and _is_normal(rhs)
    if rhs.typ == TermType.TOP:
        return True
    if lhs.typ != rhs.typ:
        if both_normal:
            return False
        return None
    if lhs.typ == TermType.VAR:
        return lhs.varname == rhs.varname
    if lhs.typ == TermType.ABS:
        assert lhs.head is not None
        assert rhs.head is not None
        return _leq(lhs.head, rhs.head)
    if lhs.typ == TermType.APP:
        assert lhs.head is not None
        assert lhs.body is not None
        assert rhs.head is not None
        assert rhs.body is not None
        leq_head = _leq(lhs.head, rhs.head)
        leq_body = leq(lhs.body, rhs.body)
        if leq_head and leq_body:
            return True
        if both_normal and leq_head is False or leq_body is False:
            return False
        return None
    raise ValueError(f"unexpected term type: {lhs.typ}")


@weak_key_cache
def leq(lhs: Term, rhs: Term) -> bool | None:
    """Returns whether lhs <= rhs, or None if the relation is unknown."""
    if lhs is rhs:
        return True
    if lhs is BOT or rhs is TOP:
        return True
    if lhs.parts <= rhs.parts:
        return True
    # Compare element wise
    result: bool | None = True
    for l in lhs.parts:
        result_l: bool | None = False
        for r in rhs.parts:
            result_lr = _leq(l, r)
            if result_lr is True:
                result_l = True
                break
            if result_lr is None:
                result_l = None
                break
        if result_l is False:
            # If l is normal, then it must be dominated by a single r.
            if _is_normal(l) or len(rhs.parts) <= 1:
                return False
            result = None
        if result_l is None:
            result = None
    return result


@weak_key_cache
def _approximate_lb(term: _Term) -> Term:
    """
    Approximates a term by replacing nonlinear beta redexes APP(ABS(-),-) with
    APP(ABS(-),BOT).
    """
    if term.typ == TermType.TOP:
        return TOP
    if term.typ == TermType.VAR:
        return _JOIN(term)
    if term.typ == TermType.ABS:
        assert term.head is not None
        return ABS(_approximate_lb(term.head))
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        # Approximate bottom-up
        head = _approximate_lb(term.head)
        body = approximate_lb(term.body)
        if head is not _JOIN(term.head) or body is not term.body:
            return approximate_lb(APP(head, body))
        # Check for beta redex
        _head = term.head
        if _head.typ == TermType.ABS:
            # Replace APP(ABS(-),-) with APP(ABS(-),BOT)
            assert _head.head is not None
            assert _head.head.free_vars.get(0, 0) > 1
            return approximate_lb(_APP(_head, BOT))
        return _JOIN(term)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
def _approximate_ub(term: _Term) -> Term:
    """
    Approximates a term by replacing nonlinear beta redexes APP(ABS(-),-) with
    APP(ABS(-),TOP).
    """
    if term.typ == TermType.TOP:
        return TOP
    if term.typ == TermType.VAR:
        return _JOIN(term)
    if term.typ == TermType.ABS:
        assert term.head is not None
        return ABS(_approximate_ub(term.head))
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None
        # Approximate bottom-up
        head = _approximate_ub(term.head)
        body = approximate_ub(term.body)
        if head is not _JOIN(term.head) or body is not term.body:
            return approximate_lb(APP(head, body))
        # Check for beta redex
        _head = term.head
        if _head.typ == TermType.ABS:
            # Replace APP(ABS(-),-) with APP(ABS(-),TOP)
            assert _head.head is not None
            assert _head.head.free_vars.get(0, 0) > 1
            return approximate_lb(_APP(_head, TOP))
        return _JOIN(term)
    raise ValueError(f"unexpected term type: {term.typ}")


@weak_key_cache
def approximate_lb(term: Term) -> Term:
    """Approximates a term by replacing APP(ABS(-),-) nodes with APP(ABS(-),BOT)."""
    return JOIN(*(_approximate_lb(part) for part in term.parts))


@weak_key_cache
def approximate_ub(term: Term) -> Term:
    """Approximates a term by replacing APP(ABS(-),-) nodes with APP(ABS(-),TOP)."""
    return JOIN(*(_approximate_ub(part) for part in term.parts))


def approximate(term: Term) -> tuple[Term, Term]:
    """Produces a linear interval approximation of a term."""
    return approximate_lb(term), approximate_ub(term)


def _beta_shell(term: _Term, radius: int) -> set[Term]:
    """
    Collects a set of beta reducts of a term at a given radius.
    """
    if not radius:
        return {_JOIN(term)}
    if _is_normal(term):
        return set()

    # Process based on term type
    result: set[Term] = set()
    if term.typ == TermType.APP:
        assert term.head is not None
        assert term.body is not None

        # Case 1: Beta reduce at this level
        if term.head.typ == TermType.ABS:
            assert term.head.head is not None
            head_body = term.head.head
            body_shifted = shift(term.body, delta=1)
            reduced = _subst(head_body, env=Env({0: body_shifted}))
            reduced = shift(reduced, delta=-1)
            result.update(beta_shell(reduced, radius - 1))

        # Case 2: Independently reduce the head and body
        for head_radius, body_radius in partitions(radius, 2):
            head_shell = _beta_shell(term.head, head_radius)
            body_shell = beta_shell(term.body, body_radius)
            for head, body in itertools.product(head_shell, body_shell):
                result.add(APP(head, body))

    elif term.typ == TermType.ABS:
        assert term.head is not None
        # Reduce inside the abstraction
        for head_reduct in _beta_shell(term.head, radius):
            result.add(ABS(head_reduct))

    return result


def beta_shell(term: Term, radius: int) -> set[Term]:
    """
    Collects a set of beta reducts of a term at a given radius.
    """
    if not radius:
        return {term}
    if is_normal(term):
        return set()

    # Compute nested shells (onions) for each part
    parts = list(term.parts)
    onions: list[list[set[Term]]] = [[] for _ in parts]
    for part, onion in zip(parts, onions, strict=True):
        for r in range(radius + 1):
            onion.append(_beta_shell(part, r))

    # Combine nested shells (onions) at given total radius
    result: set[Term] = set()
    for radii in partitions(radius, len(parts)):
        strata = [onion[r] for r, onion in zip(radii, onions, strict=True)]
        for reduced_parts in itertools.product(*strata):
            result.add(JOIN(*reduced_parts))

    return result


def beta_ball(term: Term, radius: int) -> set[Term]:
    """
    Collects a set of beta reducts of a term, up to a given radius.
    """
    result: set[Term] = {term}
    for r in range(1, radius + 1):
        result.update(beta_shell(term, r))
    return result
