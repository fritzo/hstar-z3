"""De Bruijn SKJA terms with hash consing."""

import sys
from abc import ABCMeta
from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from typing import TypeVar
from weakref import WeakKeyDictionary, ref

from immutables import Map

counter = Counter()

_V = TypeVar("_V", bound=Hashable)

_INTERN: WeakKeyDictionary[Hashable, ref[Hashable]] = WeakKeyDictionary()


def intern(x: _V) -> _V:
    """Return a canonical object for `x`, useful for hash consing."""
    counter["intern.hit"] += 1
    if x is None or x is False or x is True:
        return x  # type: ignore
    if isinstance(x, str):
        return sys.intern(x)  # type: ignore
    try:
        return _INTERN[x]()  # type: ignore
    except KeyError:
        counter["intern.hit"] -= 1
        counter["intern.miss"] += 1
        _INTERN[x] = ref(x)
        return x


class HashConsMeta(ABCMeta):
    """Metaclass to hash cons instances."""

    def __call__(self, *args, **kwargs):
        # TODO intern args and kwargs values?
        return intern(super().__call__(*args, **kwargs))

    def __new__(mcs, name, bases, namespace):
        # Support copy.deepcopy(-).
        def __deepcopy__(self, memo):
            return self

        # Support pickle.loads(pickle.dumps(-)) for dataclasses.
        def __reduce__(self):
            args = tuple(getattr(self, f) for f in self.__dataclass_fields__)
            return type(self), args

        namespace["__deepcopy__"] = __deepcopy__
        namespace["__reduce__"] = __reduce__
        return super().__new__(mcs, name, bases, namespace)

    def __getitem__(self, params):
        """Binding a generic type has no runtime effect."""
        return self


class TermType(Enum):
    TOP = 0
    APP = 1
    K = 2
    LIN = 3
    ABS = 4
    VAR = 5
    A = 6
    AJ = 7
    RVAR = 8
    SVAR = 9


EMPTY_VARS: Map[int, int] = Map()


@dataclass(frozen=True, slots=True)
class Term(metaclass=HashConsMeta):
    """Linear normal form."""

    # Data.
    typ: TermType
    varname: int = 0  # For VAR, RVAR, SVAR.
    head: "JoinTerm | None" = None
    body: "JoinTerm | None" = None
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS
    free_rvars: Map[int, int] = EMPTY_VARS
    free_svars: Map[int, int] = EMPTY_VARS


@dataclass(frozen=True, slots=True)
class JoinTerm(metaclass=HashConsMeta):
    """Join of terms in the Scott lattice."""

    # Data.
    args: frozenset[Term]
    # Metadata.
    free_vars: Map[int, int] = EMPTY_VARS
    free_rvars: Map[int, int] = EMPTY_VARS
    free_svars: Map[int, int] = EMPTY_VARS


@cache
def _JOIN(args: frozenset[Term]) -> JoinTerm:
    """Join of terms."""
    # Eagerly linearly reduce.
    if _TOP in args:
        args = frozenset({_TOP})
    return JoinTerm(
        args,
        free_vars=max_vars(*(a.free_vars for a in args)),
        free_rvars=max_vars(*(a.free_rvars for a in args)),
        free_svars=max_vars(*(a.free_svars for a in args)),
    )


def JOIN(*args: Term) -> JoinTerm:
    """Join of terms."""
    return _JOIN(frozenset().union(*args))


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
    if a.typ == TermType.K:
        return a.head
    if a.typ == TermType.LIN:
        return subs(a.head, 0, b)
    # Construct.
    return Term(
        TermType.APP,
        head=a,
        body=b,
        free_vars=add_vars(a.free_vars, b.free_vars),
        free_rvars=add_vars(a.free_rvars, b.free_rvars),
        free_svars=add_vars(a.free_svars, b.free_svars),
    )


@cache
def APP(a: JoinTerm, b: JoinTerm) -> JoinTerm:
    """Application."""
    return JOIN(*(_APP(ai, b) for ai in a.args))


def _K(a: Term) -> Term:
    """Constant function."""
    assert a.typ != TermType.TOP, "use LAM instead"
    # Construct.
    return Term(
        TermType.K,
        head=a,
        free_vars=a.free_vars,
        free_rvars=a.free_rvars,
        free_svars=a.free_svars,
    )


@cache
def _LIN(a: Term) -> Term:
    """Linear function."""
    assert a.typ != TermType.TOP, "use LAM instead"
    assert a.free_vars.get(0, 0) == 1, "use LAM instead"
    # Construct.
    free_rvars = intern(Map({k - 1: v for k, v in a.free_rvars.items() if k}))
    free_svars = intern(Map({k - 1: v for k, v in a.free_svars.items() if k}))
    return Term(
        TermType.LIN,
        head=a,
        free_vars=a.free_vars,
        free_rvars=free_rvars,
        free_svars=free_svars,
    )


@cache
def _ABS(a: Term) -> Term:
    """Nonlinear function."""
    assert a.typ != TermType.TOP, "use LAM instead"
    assert a.free_vars.get(0, 0) > 1, "use LAM instead"
    # Construct.
    free_vars = intern(Map({k - 1: v for k, v in a.free_vars.items() if k}))
    return Term(
        TermType.ABS,
        head=a,
        free_vars=free_vars,
        free_rvars=a.free_rvars,
        free_svars=a.free_svars,
    )


def _LAM(a: Term) -> Term:
    """Lambda abstraction."""
    occurrences = a.free_vars.get(0, 0)
    if occurrences == 0:
        return _K(a)
    if occurrences == 1:
        return _LIN(a)
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
def RVAR(v: int) -> JoinTerm:
    """Anonymous retract variable."""
    assert v >= 0
    return JOIN(Term(TermType.RVAR, varname=v, free_rvars=Map({v: 1})))


@cache
def SVAR(v: int) -> JoinTerm:
    """Anonymous section variable."""
    assert v >= 0
    return JOIN(Term(TermType.SVAR, varname=v, free_svars=Map({v: 1})))


@cache
def _AJ(a: Term) -> Term:
    """Binder for (section,retract) variables."""
    # Eagerly linearly reduce.
    if a.free_rvars.get(0, 0) == 0 and a.free_svars.get(0, 0) == 0:
        return a
    # Construct.
    free_rvars = intern(Map({k - 1: v for k, v in a.free_rvars.items() if k}))
    free_svars = intern(Map({k - 1: v for k, v in a.free_svars.items() if k}))
    return Term(
        TermType.A,
        head=a,
        free_vars=a.free_vars,
        free_rvars=free_rvars,
        free_svars=free_svars,
    )


def AJ(a: JoinTerm) -> JoinTerm:
    """Binder for (section,retract) variables."""
    return JOIN(*(_AJ(ai) for ai in a.args))


@cache
def A(a: JoinTerm) -> Term:
    """Type constructor. `A t f = f | AJ (t (RVAR 0) (SVAR 0)) (A t f)`."""
    return Term(
        TermType.A,
        head=a,
        free_vars=a.free_vars,
        free_rvars=a.free_rvars,
        free_svars=a.free_svars,
    )


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
def inc_vars(a: Term, min_var: int = 0) -> Term:
    """Increment all free VARs in a."""
    if all(v < min_var for v in a.free_vars):
        return a
    if a.typ == TermType.VAR:
        assert a.varname >= min_var
        return Term(TermType.VAR, varname=a.varname + 1)
    if a.typ == TermType.APP:
        head = inc_vars(a.head, min_var)
        body = JOIN(*(inc_vars(ai, min_var) for ai in a.body.args))
        return _APP(head, body)
    if a.typ == TermType.K:
        return _K(inc_vars(a.head, min_var))
    if a.typ == TermType.LIN:
        return _LIN(inc_vars(a.head, min_var + 1))
    if a.typ == TermType.ABS:
        return _ABS(inc_vars(a.head, min_var + 1))
    raise ValueError(f"unexpected term type: {a.typ}")


@cache
def inc_rsvars(a: Term, min_var: int = 0) -> Term:
    """Increment all free RVARs and SVARs in a."""
    if all(v < min_var for vs in (a.free_rvars, a.free_svars) for v in vs):
        return a
    if a.typ in (TermType.RVAR, TermType.SVAR):
        assert a.varname >= min_var
        return Term(a.type, varname=a.varname + 1)
    if a.typ == TermType.APP:
        head = inc_rsvars(a.head, min_var)
        body = JOIN(*(inc_rsvars(ai, min_var) for ai in a.body.args))
        return _APP(head, body)
    if a.typ == TermType.K:
        return _K(inc_rsvars(a.head, min_var))
    if a.typ == TermType.LIN:
        return _LIN(inc_rsvars(a.head, min_var))
    if a.typ == TermType.ABS:
        return _ABS(inc_rsvars(a.head, min_var))
    raise ValueError(f"unexpected term type: {a.typ}")


@cache
def subs(a: Term, v: int, b: JoinTerm) -> JoinTerm:
    """Substitute a VAR v := b in a."""
    if a.free_vars.get(v, 0) == 0:
        return a
    if a.typ == TermType.VAR:
        assert a.varname == v
        return b
    if a.typ == TermType.APP:
        head = subs(a.head, v, b)
        body = JOIN(*(subs(ai, v, b) for ai in a.body.args))
        return APP(head, body)
    if a.typ == TermType.K:
        return _K(subs(a.head, v, b))
    if a.typ in (TermType.LIN, TermType.ABS):
        b = inc_vars(b)  # FIXME is min_var correct?
        return _LAM(subs(a.head, v, b))
    raise ValueError(f"unexpected term type: {a.typ}")


@dataclass
class Reducer:
    """Memoizing term reducer."""

    edges: dict[Term, JoinTerm] = field(default_factory=dict)

    def reduce_step(self, a: JoinTerm) -> JoinTerm:
        """Reduce a term."""
        return JOIN(*(self._reduce_step(ai) for ai in a.args))

    def _reduce_step(self, a: Term) -> JoinTerm:
        if a in self.edges:
            return self.reduce_step(self.edges[a])
        if a.typ in (TermType.TOP, TermType.VAR, TermType.RVAR, TermType.SVAR):
            return a
        if a.typ == TermType.APP:
            if a.head.typ == TermType.ABS:
                return subs(a.head.head, 0, a.body)
            return JOIN(APP(self._reduce_step(a.head), self._reduce_step(a.body)))
        if a.typ == TermType.K or a.typ == TermType.ABS:
            head = self._reduce_step(a.head)
            return JOIN(_LAM(head))
        if a.typ == TermType.A:
            return A(self._reduce_step(a.head))
        raise ValueError(f"unexpected term type: {a.typ}")
