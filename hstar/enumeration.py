"""
# Enumeration algorithms for Syntax-Guided Synthesis (SyGuS).

This module provides algorithms for enumerating linear normal forms,
environments thereof, and refinements of term sketches. The enumerators are
intended for use by synthesis algorithms to explore the space of candidate terms
satisfying constraints.
"""

import heapq
import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Iterator
from functools import cache

from immutables import Map

from .itertools import weighted_partitions
from .metrics import COUNTERS
from .normal import (
    ABS,
    APP,
    BOT,
    JOIN,
    TOP,
    VAR,
    Env,
    Term,
    complexity,
    compress_free_vars,
    iter_permute_free_vars,
    subst,
    subst_complexity,
)

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

    The dataflow is as follows:
    - A synthesizer pulls candidates from the refiner via `.next_candidate()`.
    - A synthesizer pushes validity information to the refiner via `.mark_valid()`.
    - The refiner pushes facts to the synthesizer via the `on_fact()` callback.

    Warning: Candidates are discovered in approximately but not exactly
    increasing order of complexity. The first reason is that exploration queue
    is ordered by the complexity of unevaluated substitution pairs
    `(candidate,env)`, but `subst(candidate,env)` may be more or less complex
    than the pair complexity, due to eager linear reduction. The second reason
    is that validity may be learned out of order via `.mark_valid()`.
    """

    def __init__(self, sketch: Term, on_fact: Callable[[Term, bool], None]) -> None:
        # Persistent state.
        self.on_fact = on_fact
        self._sketch = sketch
        self._specialize: dict[Term, set[Term]] = defaultdict(set)  # general -> special
        self._specialize[sketch].add(sketch)
        self._validity: dict[Term, bool] = {}
        # Ephemeral state, used while growing.
        self._candidate_heap: list[Term] = [sketch]
        self._growth_heap: list[tuple[int, int, Term]] = []
        self._start_refining(sketch)

    def next_candidate(self) -> Term:
        """Return the next candidate term to check."""
        counter["refiner.next_candidate"] += 1
        while True:
            while not self._candidate_heap:
                self._grow()
            term = heapq.heappop(self._candidate_heap)
            if self._validity.get(term) is None:
                return term

    def mark_valid(self, candidate: Term, validity: bool) -> None:
        """
        Mark a candidate as universally valid or universally invalid.

        While closed terms in a complete theory are either valid or invalid,
        open terms may be neither universally valid nor universally invalid.
        """
        counter["refiner.mark_valid"] += 1
        assert candidate in self._specialize
        # Propagate validity to specializations.
        pending = {candidate}
        while pending:
            current = pending.pop()
            old = self._validity.get(current)
            if old is None:
                self._validity[current] = validity
                pending.update(self._specialize[current])
                self.on_fact(current, validity)
            elif old is not validity:
                raise ValueError(
                    f"contradiction: {candidate} is {validity} but {current} is {old}"
                )

    def _grow(self) -> None:
        """Grow the refinement DAG."""
        counter["refiner.grow"] += 1
        # Find a term to refine.
        if not self._growth_heap:
            raise StopIteration("Refiner is exhausted.")
        c, level, general = heapq.heappop(self._growth_heap)
        if self._validity.get(general) is not None:
            return
        heapq.heappush(self._growth_heap, (c + 1, level + 1, general))

        # Specialize the term via every env of complexity c.
        for env in env_enumerator(general.free_vars).level(level):
            special = subst(general, env)
            special = compress_free_vars(special)
            # Note special may be more or less complex than the pair complexity,
            # due to eager linear reduction and free variable compression.
            if special in self._specialize:
                continue
            # Add all permutations, whose universal closures are all equivalent.
            equiv_class = set(iter_permute_free_vars(special))
            for x in equiv_class:
                self._add_edge(general, x)
                for y in equiv_class:
                    self._add_edge(x, y)
                heapq.heappush(self._candidate_heap, x)
                if x.free_vars:
                    self._start_refining(x)

    def _start_refining(self, general: Term) -> None:
        assert general.free_vars, "cannot refine a closed term"
        level = 0
        c = complexity(general) + env_enumerator(general.free_vars).baseline
        heapq.heappush(self._growth_heap, (c, level, general))

    def _add_edge(self, general: Term, special: Term) -> None:
        counter["refiner.add_edge"] += 1
        self._specialize[general].add(special)
        # Propagate validity along the new edge.
        valid = self._validity.get(general)
        if valid is not None:
            self.mark_valid(general, valid)

    def validate(self) -> None:
        """Validate the refinement DAG, for testing."""
        for general, specials in self._specialize.items():
            for special in specials:
                valid = self._validity.get(general)
                if valid is not None:
                    assert self._validity.get(special) is valid
        for candidate in self._validity:
            assert candidate in self._specialize
