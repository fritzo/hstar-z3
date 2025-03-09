"""
# Linear-normal-forms for λ-join-calculus.

Our behavior synthesis search grammar will be a subset of the λ-join-calculus,
namely those terms that are in a particular linear normal form, i.e. that are
simplified wrt a set of rewrite rules.
"""

import heapq
import itertools
import math
from collections import defaultdict
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

        log21p = round(math.log2(1 + len(self._levels[-1])))
        counter[f"enumerator.level.log21p.{log21p}"] += 1

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
        for level in itertools.count():
            yield from self.level(level)

    def level(self, level: int) -> Iterator[Env]:
        """
        Iterates over substitutions at a given complexity level, starting at zero.
        """
        assert level >= 0
        return iter(sorted(self._get_level(level), key=repr))

    def _get_level(self, level: int) -> set[Env]:
        while len(self._levels) <= level:
            self._add_level()
        return self._levels[level]

    def _add_level(self) -> None:
        counter["env_enumerator.add_level"] += 1
        self._levels.append(set())
        c = len(self._levels) - 1 + self._v_baseline
        for partition in weighted_partitions(c, self._weights):
            factors = [enumerator.level(p) for p in partition]
            for vs in itertools.product(*factors):
                env = Env(
                    (k, v)
                    for k, v in zip(self._keys, vs, strict=True)
                    if v is not VAR(k)
                )
                assert subst_complexity(self._free_vars, env) == c - self._k_baseline
                self._levels[-1].add(env)
        log21p = round(math.log2(1 + len(self._levels[-1])))
        counter[f"env_enumerator.level.log21p.{log21p}"] += 1


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
        # Persistent state.
        self._sketch = sketch
        self._nodes: dict[Term, Env] = {sketch: Env()}  # (candidate, env) -> env
        self._generalize: dict[Term, set[Term]] = defaultdict(set)  # special -> general
        self._specialize: dict[Term, set[Term]] = defaultdict(set)  # general -> special
        self._validity: dict[Term, bool] = {}
        # Ephemeral state, used while growing.
        self._candidate_heap: list[Term] = [sketch]
        self._growth_heap: list[tuple[int, int, Term]] = []
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
                    pending.update(self._generalize[general])
                elif old is False:
                    raise ValueError("contradiction")
        else:
            # Propagate invalidity downward to specializations.
            while pending:
                special = pending.pop()
                old = self._validity.get(special)
                if old is None:
                    self._validity[special] = False
                    pending.update(self._specialize[special])
                elif old is False:
                    raise ValueError("contradiction")

    def _grow(self) -> None:
        """Grow the refinement DAG."""
        counter["refiner.grow"] += 1
        # Find a term to refine.
        if not self._growth_heap:
            raise StopIteration("Refiner is exhausted.")
        c, level, general = heapq.heappop(self._growth_heap)
        if self._validity.get(general) is False:
            return
        heapq.heappush(self._growth_heap, (c + 1, level + 1, general))

        # Specialize the term via every env of complexity c.
        for env in env_enumerator(general.free_vars).level(level):
            special = subst(general, env)
            if special in self._nodes:
                continue
            self._nodes[special] = env_compose(self._nodes[general], env)
            self._add_edge(general, special)
            heapq.heappush(self._candidate_heap, special)
            if special.free_vars:
                self._start_refining(special)

    def _start_refining(self, general: Term) -> None:
        assert general.free_vars, "cannot refine a closed term"
        level = 0
        c = complexity(general) + env_enumerator(general.free_vars).baseline
        heapq.heappush(self._growth_heap, (c, level, general))

    def _add_edge(self, general: Term, special: Term) -> None:
        counter["refiner.add_edge"] += 1
        self._generalize[special].add(general)
        self._specialize[general].add(special)
        # Propagate validity along the new edge.
        if self._validity.get(special) is True:
            if self._validity.get(general) is None:
                self.mark_valid(general, True)
        if self._validity.get(general) is False:
            if self._validity.get(special) is None:
                self.mark_valid(special, False)

    def validate(self) -> None:
        """Validate the refinement DAG, for testing."""
        for special, env in self._nodes.items():
            assert subst(self._sketch, env) == special
        for special, generals in self._generalize.items():
            for general in generals:
                assert special in self._specialize[general]
                if self._validity.get(special) is True:
                    assert self._validity.get(general) is True
        for general, specials in self._specialize.items():
            for special in specials:
                assert general in self._generalize[special]
                if self._validity.get(general) is False:
                    assert self._validity.get(special) is False
        for candidate in self._validity:
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
        self._nodes: dict[Env, Env] = {sketch: Env()}  # (candidate, env) -> env
        self._generalize: dict[Env, set[Env]] = defaultdict(set)  # special -> general
        self._specialize: dict[Env, set[Env]] = defaultdict(set)  # general -> special
        self._validity: dict[Env, bool] = {}
        # Ephemeral state, used while growing.
        self._candidate_heap: list[Env] = [sketch]
        self._growth_heap: list[tuple[int, int, Env]] = []
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
                    pending.update(self._generalize[general])
                elif old is False:
                    raise ValueError("contradiction")
        else:
            # Propagate invalidity downward to specializations.
            while pending:
                special = pending.pop()
                old = self._validity.get(special)
                if old is None:
                    self._validity[special] = False
                    pending.update(self._specialize[special])
                elif old is False:
                    raise ValueError("contradiction")

    def _grow(self) -> None:
        """Grow the refinement DAG."""
        counter["env_refiner.grow"] += 1
        # Find an environment to refine.
        if not self._growth_heap:
            raise StopIteration("EnvRefiner is exhausted.")
        c, level, general = heapq.heappop(self._growth_heap)
        if self._validity.get(general) is False:
            return
        heapq.heappush(self._growth_heap, (c + 1, level + 1, general))

        # Specialize the environment via every env of complexity c.
        for env in env_enumerator(env_free_vars(general)).level(level):
            special = env_compose(general, env)
            if special in self._nodes:
                continue
            self._nodes[special] = env_compose(self._nodes[general], env)
            self._add_edge(general, special)
            heapq.heappush(self._candidate_heap, special)
            if env_free_vars(special):
                self._start_refining(special)

    def _start_refining(self, general: Env) -> None:
        assert env_free_vars(general), "cannot refine a closed environment"
        level = 0
        c = env_complexity(general) + env_enumerator(env_free_vars(general)).baseline
        heapq.heappush(self._growth_heap, (c, level, general))

    def _add_edge(self, general: Env, special: Env) -> None:
        counter["env_refiner.add_edge"] += 1
        self._generalize[special].add(general)
        self._specialize[general].add(special)
        # Propagate validity along the new edge.
        if self._validity.get(special) is True:
            if self._validity.get(general) is None:
                self.mark_valid(general, True)
        if self._validity.get(general) is False:
            if self._validity.get(special) is None:
                self.mark_valid(special, False)

    def validate(self) -> None:
        """Validate the refinement DAG, for testing."""
        for special, env in self._nodes.items():
            assert env_compose(self._sketch, env) == special
        for special, generals in self._generalize.items():
            for general in generals:
                assert special in self._specialize[general]
                if self._validity.get(special) is True:
                    assert self._validity.get(general) is True
        for general, specials in self._specialize.items():
            for special in specials:
                assert general in self._generalize[special]
                if self._validity.get(general) is False:
                    assert self._validity.get(special) is False
        for candidate in self._validity:
            assert candidate in self._nodes
