"""
# An engine for memoized shared nonlinear reduction.

This module implements an engine for simultaneously reducing a population of
terms in an E-graph. Following hstar.normal, we eagerly linearly reduce terms,
so that the engine need only consider big-steps of nonlinear beta reduction.
Following Dolstra [1], we implement sharing by hash-consing terms and memoizing
beta steps from each term.

An alternative approach to this module's first-order sharing would be to
implement higher-order sharing via interaction networks (Asperti [2]). It
remains to be seen which paradigm will better scale to populations of programs.

[1] Eelco Dolstra (2008) "Maximal Laziness"
    https://edolstra.github.io/pubs/laziness-ldta2008-final.pdf
[2] Andrea Asperti (1998)
    "The optimal implementation of functional programming languages"

## Technical details

- The language is λ-join-calculus with de Bruijn indices,
  with term constructors {TOP, VAR, ABS, APP, JOIN}.
  - Term constructors perform eager linear reduction.
- E-graph nodes are equivalence classes of λ-join-calculus linear normal forms,
  implemented as integer ids.
- There is a single global data structure that grows monotonically, and
  includes:
  - Syntax tables for term constructors, including inverse tables used in
    congruence closure.
  - A union-find structure for tracking the canonical representatives of
    equivalence classes.
  - Metadata for each term, including:
    - Free variables.
    - Head normal form.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import NewType

from immutables import Map

from .hashcons import HashConsMeta, intern
from .metrics import COUNTERS
from .varsets import add_vars, max_vars, min_vars

counter = COUNTERS[__name__]

# There is a single global E-graph that monotonically grows.
Node = NewType("Node", int)
_next_node: int = 0


def _new_node() -> Node:
    """Create a new unique node."""
    global _next_node
    result = Node(_next_node)
    _next_node += 1
    counter["new_node"] += 1
    return result


# Language.
TOP: Node = _new_node()
_VAR: list[Node] = []
_ABS: dict[Node, Node] = {}
_APP: dict[tuple[Node, Node], Node] = {}
_JOIN: dict[frozenset[Node], Node] = {}
_ABS_INDEX: dict[Node, set[Node]] = defaultdict(set)
_APP_INDEX: dict[Node, set[tuple[Node, Node]]] = defaultdict(set)
_JOIN_INDEX: dict[Node, set[frozenset[Node]]] = defaultdict(set)


# Term constructors.


def VAR(i: int) -> Node:
    while len(_VAR) <= i:
        node = _new_node()
        _VAR.append(node)
        _HEAD[node] = Head(HeadType.VAR, i)
    return _VAR[i]


def ABS(body: Node, *, result: Node | None = None) -> Node:
    body = find(body)
    if body in _ABS:
        return _ABS[body]

    # Eagerly linearly reduce.
    head = _HEAD[body]
    if head.typ == HeadType.TOP:
        return TOP
    elif head.typ == HeadType.APP:
        assert isinstance(head.args, tuple)
        lhs, rhs = head.args
        if _FREE_VARS[lhs].get(0, 0) == 0 and rhs == VAR(0):
            return shift(lhs, delta=-1)  # η-reduction
    elif head.typ == HeadType.JOIN:
        assert isinstance(head.args, frozenset)
        parts = frozenset(ABS(part) for part in head.args)
        return JOIN(parts)

    # Create a new node for the abstraction.
    if result is None:
        result = _new_node()
    _ABS[body] = result
    _ABS_INDEX[result].add(body)

    # Update free variables.
    free_vars = add_vars(_FREE_VARS[body])
    if (old_free_vars := _FREE_VARS.get(result)) is not None:
        free_vars = min_vars(old_free_vars, free_vars)
    _FREE_VARS[result] = free_vars

    # Update head structure.
    head = Head(HeadType.ABS, body)
    if (old_head := _HEAD.get(result)) is not None:
        head = head_union(old_head, head)
    _HEAD[result] = head
    return result


def can_safely_copy(node: Node) -> bool:
    """Check whether a term can be safely copied."""
    head = _HEAD[node]
    if head.typ == HeadType.TOP:
        return True
    if head.typ == HeadType.VAR:
        return True
    if head.typ == HeadType.JOIN:
        assert isinstance(head.args, frozenset)
        return len(head.args) == 0
    return False


def APP(lhs: Node, rhs: Node, *, result: Node | None = None) -> Node:
    lhs = find(lhs)
    rhs = find(rhs)
    key = intern((lhs, rhs))
    if key in _APP:
        return _APP[key]

    # Eagerly linearly reduce.
    head = _HEAD[lhs]
    if head.typ == HeadType.TOP:
        return TOP
    elif head.typ == HeadType.ABS:
        assert isinstance(head.args, tuple)
        body = head.args[0]
        if _FREE_VARS[body].get(0, 0) <= 1 or can_safely_copy(rhs):
            rhs = shift(rhs, delta=1)
            result = subst(body, 0, rhs)
            return shift(result, delta=-1)
    elif head.typ == HeadType.JOIN:
        assert isinstance(head.args, frozenset)
        parts = frozenset(APP(part, rhs) for part in head.args)
        return JOIN(parts)

    # Create a new node for the application.
    if result is None:
        result = _new_node()
    _APP[key] = result
    _APP_INDEX[result].add(key)

    # Update free variables.
    free_vars = add_vars(*(_FREE_VARS[part] for part in key))
    if (old_free_vars := _FREE_VARS.get(result)) is not None:
        free_vars = min_vars(old_free_vars, free_vars)
    _FREE_VARS[result] = free_vars

    # Update head structure.
    head = Head(HeadType.APP, key)
    if (old_head := _HEAD.get(result)) is not None:
        head = head_union(old_head, head)
    _HEAD[result] = head
    return result


def JOIN(parts: frozenset[Node], *, result: Node | None = None) -> Node:
    """Join a set of nodes."""
    parts = frozenset(find(part) for part in parts)
    if parts in _JOIN:
        return _JOIN[parts]

    # Eagerly linearly reduce, applying rules for ACI and TOP.
    parts_in = set(parts)
    parts_out: set[Node] = set()
    while parts_in:
        part = parts_in.pop()
        head = _HEAD[part]
        if head.typ == HeadType.TOP:
            return TOP
        elif head.typ == HeadType.JOIN:
            assert isinstance(head.args, frozenset)
            for part in head.args:
                part = find(part)
                if part not in parts_out:
                    parts_in.add(part)
        else:
            parts_out.add(part)
    parts = intern(frozenset(parts_out))

    # Create a new node for the join.
    if result is None:
        result = _new_node()
    _JOIN[parts] = result
    _JOIN_INDEX[result].add(parts)

    # Update free variables.
    free_vars = max_vars(*(_FREE_VARS[part] for part in parts))
    if (old_free_vars := _FREE_VARS.get(result)) is not None:
        free_vars = min_vars(old_free_vars, free_vars)
    _FREE_VARS[result] = free_vars

    # Update head structure.
    head = Head(HeadType.JOIN, parts)
    if (old_head := _HEAD.get(result)) is not None:
        head = head_union(old_head, head)
    _HEAD[result] = head

    return result


# Free variables and substitution.
_FREE_VARS: dict[Node, Map[int, int]] = {}


def shift(node: Node, *, start: int = 0, delta: int = 1) -> Node:
    """Shift the free variables in a term."""
    head = _HEAD[node]
    if head.typ == HeadType.TOP:
        return node
    if head.typ == HeadType.VAR:
        assert isinstance(head.args, int)
        if head.args >= start:
            return VAR(head.args + delta)
        return node
    if head.typ == HeadType.ABS:
        assert isinstance(head.args, tuple)
        body = head.args[0]
        body = shift(body, start=start + 1, delta=delta)
        return ABS(body)
    if head.typ == HeadType.APP:
        assert isinstance(head.args, tuple)
        lhs, rhs = head.args
        lhs = shift(lhs, start=start, delta=delta)
        rhs = shift(rhs, start=start, delta=delta)
        return APP(lhs, rhs)
    if head.typ == HeadType.JOIN:
        assert isinstance(head.args, frozenset)
        parts = frozenset(shift(part, start=start, delta=delta) for part in head.args)
        return JOIN(parts)
    raise TypeError(f"Unknown head type: {head.typ}")


def subst(node: Node, i: int, rhs: Node) -> Node:
    """Substitute a term for a free variable."""
    head = _HEAD[node]
    if head.typ == HeadType.TOP:
        return node
    if head.typ == HeadType.VAR:
        assert isinstance(head.args, int)
        if head.args == i:
            return rhs
        return node
    if head.typ == HeadType.ABS:
        assert isinstance(head.args, tuple)
        body = head.args[0]
        body = subst(body, i + 1, shift(rhs, delta=1))
        return ABS(body)
    if head.typ == HeadType.APP:
        assert isinstance(head.args, tuple)
        lhs, rhs = head.args
        lhs = subst(lhs, i, rhs)
        rhs = subst(rhs, i, rhs)
        return APP(lhs, rhs)
    if head.typ == HeadType.JOIN:
        assert isinstance(head.args, frozenset)
        parts = frozenset(subst(part, i, rhs) for part in head.args)
        return JOIN(parts)
    raise TypeError(f"Unknown head type: {head.typ}")


# Head normalization.


class HeadType(Enum):
    TOP = "TOP"
    VAR = "VAR"
    ABS = "ABS"
    APP = "APP"
    JOIN = "JOIN"


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Head(metaclass=HashConsMeta):
    typ: HeadType
    args: None | int | tuple[Node, ...] | frozenset[Node]


_HEAD: dict[Node, Head] = {}
_HEAD[TOP] = Head(HeadType.TOP, None)

# Congruence closure, equality saturation, union-find.
_REP: dict[Node, Node] = {}  # representatives
_DEP: set[Node] = set()  # deprecated nodes, pending merge


def find(x: Node) -> Node:
    """Find the representative of a node."""
    x = _REP.get(x, x)
    if x == _REP.get(x, x):
        return x
    _REP[x] = find(_REP[x])
    return _REP[x]


def union(x: Node, y: Node) -> Node:
    """Enqueue merger of two nodes."""
    x = find(x)
    y = find(y)
    if x == y:
        return x
    if x > y:
        x, y = y, x
    _DEP.add(y)
    _REP[y] = x
    return x


def head_union(x: Head, y: Head) -> Head:
    """Union two heads."""
    if x.typ == HeadType.TOP or y.typ == HeadType.TOP:
        assert x == y
        return x
    if x.typ == HeadType.VAR or y.typ == HeadType.VAR:
        assert x == y
        return x
    if x.typ == HeadType.ABS or y.typ == HeadType.ABS:
        assert isinstance(x.args, tuple)
        assert isinstance(y.args, tuple)
        arg = union(x.args[0], y.args[0])
        return Head(HeadType.ABS, (arg,))
    raise NotImplementedError("TODO")


def saturate() -> None:
    """Saturate the E-graph."""
    while _DEP:
        saturate_step()


def saturate_step() -> None:
    """Perform one unit of saturation work."""
    counter["saturate_step"] += 1

    x = _DEP.pop()
    y = find(x)
    assert y < x

    # Merge abstraction structure.
    for body in _ABS_INDEX.pop(x, ()):
        if _ABS.get(body) == x:
            del _ABS[body]
        body = find(body)
        ABS(body, result=y)

    # Merge application structure.
    raise NotImplementedError("TODO: APP")

    # Merge join structure.
    raise NotImplementedError("TODO: JOIN")

    # Merge head normalization.
    _HEAD[y] = head_union(_HEAD.pop(x), _HEAD[y])
