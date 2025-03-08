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
    env_free_vars,
    subst,
    subst_complexity,
)
from .itertools import weighted_partitions
from .metrics import COUNTERS

counter = COUNTERS[__name__]


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
        counter["enumerator.add_level"] += 1
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
        self.baseline = self._v_baseline - self._k_baseline
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
        counter["env_enumerator.add_level"] += 1
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
def env_enumerator(free_vars: Map[int, int]) -> EnvEnumerator:
    """Enumerator for all substitutions, sorted by subst_complexity, then repr."""
    return EnvEnumerator(free_vars)


class Refiner:
    """
    Data structure representing the DAG of refinements of a term sketch,
    propagating validity along general-special edges.
    """

    def __init__(self, sketch: Term) -> None:
        assert sketch.free_vars
        # Persistent state.
        self._sketch = sketch
        self._nodes: dict[Term, Env] = {sketch: Map()}  # term -> substitution
        self._edges: dict[tuple[Term, Term], Env] = {}  # (special, general) -> env
        self._generalize: dict[Term, set[Term]] = {}  # special -> {general}
        self._specialize: dict[Term, set[Term]] = {}  # general -> {special}
        self._validity: dict[Term, bool | None] = {}
        # Ephemeral state, used while growing.
        self._candidate_heap: list[Term] = [sketch]
        self._growth_heap: list[tuple[int, Term]] = []
        self._start_refining(sketch)

    def next_candidate(self) -> Term:
        """Return the next candidate term to check."""
        counter["refiner.next_candidate"] += 1
        while not self._candidate_heap:
            self._grow()
        return heapq.heappop(self._candidate_heap)

    def mark_valid(self, candidate: Term, validity: bool) -> None:
        """Mark a candidate as valid or invalid."""
        counter["refiner.mark_valid"] += 1
        assert candidate in self._nodes
        pending = {candidate}
        if validity:
            # Propagate validity upward to generalizations.
            while pending:
                general = pending.pop()
                old = self._validity.get(general)
                if old is None:
                    self._validity[general] = True
                    pending.update(self._generalize.get(general, set()))
                elif old is False:
                    raise ValueError("contradiction")
        else:
            # Propagate invalidity downward to specializations.
            while pending:
                special = pending.pop()
                old = self._validity.get(special)
                if old is None:
                    self._validity[special] = False
                    pending.update(self._specialize.get(special, set()))
                elif old is False:
                    raise ValueError("contradiction")

    def _grow(self) -> None:
        """Grow the refinement DAG."""
        counter["refiner.grow"] += 1
        # Find a term to refine.
        if not self._growth_heap:
            raise StopIteration("Refiner is exhausted.")
        c, general = heapq.heappop(self._growth_heap)
        if self._validity.get(general) is False:
            return
        heapq.heappush(self._growth_heap, (c + 1, general))

        # Specialize the term via every env of complexity c.
        refinements = env_enumerator(general.free_vars)
        level = c - refinements.baseline
        for env in refinements.level(level):
            special = subst(general, env)
            if special in self._nodes:
                continue
            self._nodes[special] = env_compose(self._nodes[general], env)
            self._edges[special, general] = env
            self._generalize.setdefault(special, set()).add(general)
            self._specialize.setdefault(general, set()).add(special)
            heapq.heappush(self._candidate_heap, special)
            if special.free_vars:
                self._start_refining(special)

    def _start_refining(self, general: Term) -> None:
        c = complexity(general) + env_enumerator(general.free_vars).baseline
        heapq.heappush(self._growth_heap, (c, general))

    def validate(self) -> None:
        """Validate the refinement DAG, for testing."""
        for special, env in self._nodes.items():
            assert subst(self._sketch, env) == special
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
        for candidate, _ in self._validity.items():
            assert candidate in self._nodes


class EnvRefiner:
    """
    Data structure representing the DAG of refinements of an environment sketch,
    propagating validity along general-special edges.
    """

    def __init__(self, sketch: Env) -> None:
        # Persistent state.
        self._sketch = sketch
        self._free_vars = env_free_vars(sketch)
        self._nodes: dict[Env, Env] = {sketch: sketch}
        self._edges: dict[tuple[Env, Env], Env] = {}
        self._generalize: dict[Env, set[Env]] = {}
        self._specialize: dict[Env, set[Env]] = {}
        self._validity: dict[Env, bool | None] = {}
        # Ephemeral state, used while growing.
        self._candidate_heap: list[Env] = [sketch]
        self._growth_heap: list[tuple[int, Env]] = []
        self._start_refining(sketch)

    def next_candidate(self) -> Env:
        """Return the next candidate environment to check."""
        counter["env_refiner.next_candidate"] += 1
        while not self._candidate_heap:
            self._grow()
        return heapq.heappop(self._candidate_heap)

    def mark_valid(self, candidate: Env, validity: bool) -> None:
        """Mark a candidate as valid or invalid."""
        counter["env_refiner.mark_valid"] += 1
        assert candidate in self._nodes
        pending = {candidate}
        if validity:
            # Propagate validity upward to generalizations.
            while pending:
                general = pending.pop()
                old = self._validity.get(general)
                if old is None:
                    self._validity[general] = True
                    pending.update(self._generalize.get(general, set()))
                elif old is False:
                    raise ValueError("contradiction")
        else:
            # Propagate invalidity downward to specializations.
            while pending:
                special = pending.pop()
                old = self._validity.get(special)
                if old is None:
                    self._validity[special] = False
                    pending.update(self._specialize.get(special, set()))
                elif old is False:
                    raise ValueError("contradiction")

    def _grow(self) -> None:
        """Grow the refinement DAG."""
        counter["env_refiner.grow"] += 1
        # Find an environment to refine.
        if not self._growth_heap:
            raise StopIteration("EnvRefiner is exhausted.")
        c, general = heapq.heappop(self._growth_heap)
        if self._validity.get(general) is False:
            return
        heapq.heappush(self._growth_heap, (c + 1, general))

        # Specialize the environment via every env of complexity c.
        refinements = env_enumerator(env_free_vars(general))
        level = c - refinements.baseline
        for env in refinements.level(level):
            special = env_compose(general, env)
            if special in self._nodes:
                continue
            self._nodes[special] = env_compose(self._nodes[general], env)
            self._edges[special, general] = env
            self._generalize.setdefault(special, set()).add(general)
            self._specialize.setdefault(general, set()).add(special)
            heapq.heappush(self._candidate_heap, special)
            if env_free_vars(special):
                self._start_refining(special)

    def _start_refining(self, general: Env) -> None:
        c = env_complexity(general) + env_enumerator(env_free_vars(general)).baseline
        heapq.heappush(self._growth_heap, (c, general))

    def validate(self) -> None:
        """Validate the refinement DAG, for testing."""
        for special, env in self._nodes.items():
            assert env_compose(self._sketch, env) == special
        for special, general in self._edges:
            assert env_compose(special, self._edges[special, general]) == general
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
        for candidate, _ in self._validity.items():
            assert candidate in self._nodes
