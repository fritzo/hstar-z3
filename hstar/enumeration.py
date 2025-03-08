"""
# Linear-normal-forms for λ-join-calculus.

Our behavior synthesis search grammar will be a subset of the λ-join-calculus,
namely those terms that are in a particular linear normal form, i.e. that are
simplified wrt a set of rewrite rules.
"""

import heapq
import itertools
from collections.abc import Iterator
from functools import cache
from typing import NamedTuple

from immutables import Map

from .grammar import (
    ABS,
    APP,
    BOT,
    JOIN,
    TOP,
    VAR,
    Env,
    Term,
    complexity,
    env_complexity,
    env_compose,
    subst,
    subst_complexity,
)
from .util import partitions, weighted_partitions


class Enumerator:
    """Generator for all terms, sorted by complexity, then repr."""

    def __init__(self) -> None:
        self._levels: list[set[Term]] = [set()]

    def __iter__(self) -> Iterator[Term]:
        """Iterates over all terms."""
        for c in itertools.count():
            yield from self.level(c)

    def up_to(self, ub: int) -> Iterator[Term]:
        """Iterates over terms of complexity up to an inclusive upper bound."""
        for c in range(1, ub + 1):
            yield from self.level(c)

    def level(self, c: int) -> Iterator[Term]:
        """Iterates over terms of a given complexity."""
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
            self._add_term(ABS(term))

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


enumerator = Enumerator()


class EnvEnumerator:
    """Generator for all environments, sorted by env_complexity, then repr."""

    def __init__(self, keys: frozenset[int]) -> None:
        self._keys = keys
        self._levels: list[set[Env]] = [set()]

    def __iter__(self) -> Iterator[Env]:
        for c in itertools.count():
            yield from self.level(c)

    def level(self, c: int) -> Iterator[Env]:
        """Iterates over environments of a given complexity."""
        return iter(sorted(self._get_level(c), key=repr))

    def _get_level(self, complexity: int) -> set[Env]:
        while len(self._levels) <= complexity:
            self._add_level()
        return self._levels[complexity]

    def _add_level(self) -> None:
        self._levels.append(set())
        c = len(self._levels) - 1
        for partition in partitions(c, len(self._keys)):
            factors = [enumerator.level(p) if p else [None] for p in partition]
            for vs in itertools.product(*factors):
                env = Map(
                    (k, v) for k, v in zip(self._keys, vs, strict=True) if v is not None
                )
                assert env_complexity(env) == c
                self._levels[-1].add(env)


@cache
def env_enumerator(keys: frozenset[int]) -> EnvEnumerator:
    """Enumerator for all environments, sorted by env_complexity, then repr."""
    return EnvEnumerator(keys)


class SubstEnumerator:
    """Generator for all substitutions, sorted by subst_complexity, then repr."""

    def __init__(self, free_vars: Map[int, int]) -> None:
        assert all(count > 0 for count in free_vars.values())
        self._free_vars = free_vars
        self._keys = tuple(sorted(self._free_vars, reverse=True))
        self._weights = tuple(self._free_vars[k] for k in self._keys)
        self._k_baseline = sum(
            complexity(VAR(k)) * v for k, v in self._free_vars.items()
        )
        self._v_baseline = sum(self._free_vars.values())
        self._levels: list[set[Env]] = []

    def __iter__(self) -> Iterator[Env]:
        for c in itertools.count():
            yield from self.level(c)

    def level(self, c: int) -> Iterator[Env]:
        """Iterates over substitutions of a given complexity."""
        return iter(sorted(self._get_level(c), key=repr))

    def _get_level(self, complexity: int) -> set[Env]:
        while len(self._levels) <= complexity:
            self._add_level()
        return self._levels[complexity]

    def _add_level(self) -> None:
        self._levels.append(set())
        c = len(self._levels) - 1 + self._v_baseline
        for partition in weighted_partitions(c, self._weights):
            factors = [enumerator.level(p) for p in partition]
            for vs in itertools.product(*factors):
                env = Map(
                    (k, v)
                    for k, v in zip(self._keys, vs, strict=True)
                    if v is not VAR(k)
                )
                assert subst_complexity(self._free_vars, env) == c - self._k_baseline
                self._levels[-1].add(env)


@cache
def subst_enumerator(free_vars: Map[int, int]) -> SubstEnumerator:
    """Enumerator for all substitutions, sorted by subst_complexity, then repr."""
    return SubstEnumerator(free_vars)


class RefinementEnumerator:
    """
    Generator for all refinements of a sketch, approximately sorted by complexity.

    Each yielded refinement substitutes terms for free variables in the sketch.
    """

    def __init__(self, sketch: Term) -> None:
        self._sketch = sketch
        self._env_enumerator = env_enumerator(frozenset(sketch.free_vars))

    def __iter__(self) -> Iterator[Term]:
        seen: set[Term] = set()
        for env in self._env_enumerator:
            term = subst(self._sketch, env)
            if term in seen:
                continue
            seen.add(term)
            yield term


class GrowthNode(NamedTuple):
    subst_c: int
    term: Term


class Refinery:
    """
    Data structure representing the DAG of refinements of a sketch, together
    with pairwise refinement edges `env : general -> special` between pairs of
    terms.
    """

    def __init__(self, sketch: Term) -> None:
        assert sketch.free_vars
        # Persistent state.
        self._sketch = sketch
        self._terms: dict[Term, Env] = {}
        self._edges: dict[tuple[Term, Term], Env] = {}  # (special, general) -> env
        self._generalize: dict[Term, set[Term]] = {}  # special -> {general}
        self._specialize: dict[Term, set[Term]] = {}  # general -> {special}
        self._validity: dict[Term, bool | None] = {}
        # Ephemeral state, used while growing.
        self._candidate_heap: list[Term] = [sketch]
        self._candidates_seen: set[Term] = set()
        self._growth_heap: list[GrowthNode] = [GrowthNode(0, sketch)]

    def next_candidate(self) -> Term:
        """Return the next candidate term to check."""
        while not self._candidate_heap:
            self._grow()
        return heapq.heappop(self._candidate_heap)

    def mark_valid(self, term: Term, validity: bool | None) -> None:
        """Mark a candidate as valid or invalid."""
        # Propagate validity to all terms that refine it.
        raise NotImplementedError("TODO")

    def _grow(self) -> None:
        """Grow the refinement DAG."""
        # Find a term to refine.
        if not self._growth_heap:
            raise ValueError("Refinery is exhausted.")
        c, general = heapq.heappop(self._growth_heap)
        if self._validity[general] is False:
            return
        heapq.heappush(self._growth_heap, GrowthNode(c + 1, general))

        # Specialize the term via every env of complexity c.
        # FIXME what is c, the complexity of the final term?
        for env in subst_enumerator(general.free_vars).level(c):
            special = subst(general, env)
            if special in self._terms:
                continue
            self._terms[special] = env_compose(self._terms[general], env)
            self._edges[special, general] = env
            self._generalize.setdefault(special, set()).add(general)
            self._specialize.setdefault(general, set()).add(special)
            heapq.heappush(self._growth_heap, GrowthNode(0, special))
            heapq.heappush(self._candidate_heap, special)

    def validate(self) -> None:
        """Validate the refinement DAG, for testing."""
        for term, env in self._terms.items():
            assert subst(term, env) == self._sketch
        for (special, general), env in self._edges.items():
            assert subst(special, env) == general
        for special, generals in self._generalize.items():
            for general in generals:
                assert special in self._specialize[general]
                if special in self._validity and self._validity[special] is True:
                    assert self._validity[general] is True
        for general, specials in self._specialize.items():
            for special in specials:
                assert general in self._generalize[special]
                if general in self._validity and self._validity[general] is False:
                    assert self._validity[special] is False
        for term, _ in self._validity.items():
            assert term in self._terms
