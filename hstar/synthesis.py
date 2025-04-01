"""
# Synthesis algorithms for λ-join-calculus.

This module provides algorithms for synthesizing λ-join-calculus terms
that satisfy given constraints.
"""

import abc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import z3
from z3 import Not

from . import language
from .enumeration import Refiner
from .metrics import COUNTERS
from .normal import Term
from .theory import add_beta_ball, add_theory

logger = logging.getLogger(__name__)
counter = COUNTERS[__name__]

DEFAULT_RADIUS: int = 3
DEFAULT_TIMEOUT_MS: int = 1000
LEMMAS: WeakKeyDictionary[z3.Solver, list[z3.ExprRef]] = WeakKeyDictionary()


def lemma_forall(solver: z3.Solver, holes: list[z3.ExprRef], lemma: z3.ExprRef) -> None:
    lemmas = LEMMAS.setdefault(solver, [])
    counter["lemmas"] += 1
    name = f"lemma_{len(lemmas)}"
    if holes:
        counter["lemmas.forall"] += 1
        lemma = z3.ForAll(
            holes,
            lemma,
            patterns=[language.as_pattern(lemma)],
            qid=name,
        )
    logger.debug(f"{name}: {lemma}")
    solver.assert_and_track(lemma, name)
    lemmas.append(lemma)


class SynthesizerBase(metaclass=abc.ABCMeta):
    sketch: Term
    constraint: Callable[[Term], z3.ExprRef]
    refiner: Refiner
    solver: z3.Solver

    @abc.abstractmethod
    def step(self) -> None:
        """Performs a unit of inference work."""


class Synthesizer(SynthesizerBase):
    """
    A synthesis algorithm that searches through refinements of a sketch.

    Args:
        sketch: The term sketch to refine.
        constraint: A function that takes a candidate term and returns a Z3
            expression representing a constraint on the candidate.
        on_fact: A callback that is called when a fact is proven.
        timeout_ms: The timeout for the solver in milliseconds.
        radius: The radius of the beta ball around the candidate.
    """

    def __init__(
        self,
        sketch: Term,
        constraint: Callable[[Term], z3.ExprRef],
        on_fact: Callable[[Term, bool], None],
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        radius: int = DEFAULT_RADIUS,
    ) -> None:
        self.sketch = sketch
        self.constraint = constraint
        self.refiner = Refiner(sketch, on_fact)
        self.solver = z3.Solver()
        self.solver.set(timeout=timeout_ms)
        self.solver.timeout_ms = timeout_ms  # for hstar use only
        self.radius = radius
        add_theory(self.solver)

    def step(self) -> None:
        """Performs a unit of inference work."""
        counter["synthesizer.step"] += 1
        candidate = self.refiner.next_candidate()
        logger.debug(f"Checking: {candidate}")
        constraint = self.constraint(candidate)
        holes, constraint = language.hole_closure(constraint)
        not_constraint = Not(constraint)

        # Check if the constraint is unsatisfiable,
        # which means the candidate is universally invalid.
        if self._is_unsat(candidate, constraint):
            counter["pos.unsat"] += 1
            logger.debug(f"Rejected: {candidate}")
            self.refiner.mark_valid(candidate, False)
            # By DeMorgan's law: ¬∃x.φ(x) ≡ ∀x.¬φ(x)
            lemma_forall(self.solver, holes, not_constraint)

        # Check if the negation of the constraint is unsatisfiable,
        # which means the original constraint is universally valid.
        if self._is_unsat(candidate, not_constraint):
            counter["neg.unsat"] += 1
            logger.debug(f"Accepted: {candidate}")
            self.refiner.mark_valid(candidate, True)
            # By DeMorgan's law: ¬∃x.¬φ(x) ≡ ∀x.φ(x)
            lemma_forall(self.solver, holes, constraint)

    def _is_unsat(self, candidate: Term, formula: z3.ExprRef) -> bool:
        with self.solver:
            add_beta_ball(self.solver, candidate, self.radius)
            # We expect sat to never occur, as our base theory has no finite model,
            # hence we distinguish only between unsat and unknown/sat.
            return bool(self.solver.check(formula) == z3.unsat)


@dataclass(frozen=True, slots=True)
class Claim:
    term: Term
    holes: list[z3.ExprRef]
    constraint: z3.ExprRef
    not_constraint: z3.ExprRef

    def negate(self) -> "Claim":
        return Claim(
            term=self.term,
            holes=self.holes,
            constraint=self.not_constraint,
            not_constraint=self.constraint,
        )


class BatchingSynthesizer(SynthesizerBase):
    """
    A synthesis algorithm that searches through refinements of a sketch.

    Args:
        sketch: The term sketch to refine.
        constraint: A function that takes a candidate term and returns a Z3
            expression representing a constraint on the candidate.
        on_fact: A callback that is called when a fact is proven.
        batch_size: The number of claims to simultaneously check.
        timeout_ms: The timeout for the solver in milliseconds.
        radius: The radius of the beta ball around the candidate.
    """

    def __init__(
        self,
        sketch: Term,
        constraint: Callable[[Term], z3.ExprRef],
        on_fact: Callable[[Term, bool], None],
        *,
        batch_size: int = 1,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        radius: int = DEFAULT_RADIUS,
    ) -> None:
        self.sketch = sketch
        self.constraint = constraint
        self.refiner = Refiner(sketch, on_fact)
        self.batch_size = batch_size
        self.solver = z3.Solver()
        self.solver.set(timeout=timeout_ms)
        self.solver.set("unsat_core", True)
        self.solver.timeout_ms = timeout_ms
        self.radius = radius
        add_theory(self.solver)

        # Candidates remain pending until either they are decided or hit a timeout.
        self._next_id = 0
        self._pending: tuple[dict[str, Claim], dict[str, Claim]] = {}, {}

    def step(self) -> None:
        """Performs a unit of inference work."""
        counter["batch_synthesizer.step"] += 1
        for valid in (False, True):
            while max(map(len, self._pending)) < self.batch_size:
                self._add_candidate()
            if self._pending[valid]:
                self._step(valid)

    def _add_candidate(self) -> None:
        counter["batch_synthesizer.add_candidate"] += 1
        candidate = self.refiner.next_candidate()
        logger.debug(f"Checking: {candidate}")
        constraint = self.constraint(candidate)
        holes, constraint = language.hole_closure(constraint)
        key = f"candidate_{self._next_id}"
        self._next_id += 1
        claim = Claim(
            term=candidate,
            holes=holes,
            constraint=constraint,
            not_constraint=Not(constraint),
        )
        self._pending[True][key] = claim
        self._pending[False][key] = claim.negate()

    def _step(self, valid: bool) -> None:
        counter["batch_synthesizer.check"] += 1
        pending = self._pending[valid]
        proved, discard = self._check(pending)
        for key in discard:
            pending.pop(key)
        for key in proved:
            claim = pending.pop(key)
            self._pending[not valid].pop(key, None)
            action = "Accepted" if valid else "Rejected"
            logger.debug(f"{action}: {claim.term}")
            self.refiner.mark_valid(claim.term, valid)  # TODO move into _check

    def _check(self, claims: dict[str, Claim]) -> tuple[set[str], set[str]]:
        """
        Returns (proved, discard) claim keys. Remaining claims should be retried.
        """
        # Assume the query is convex, i.e. if any subset of the claims is
        # unsatisfiable, then at least one of the claims is unsatisfiable.
        counter["batch_synthesizer.check"] += 1
        proved: set[str] = set()
        discard: set[str] = set()

        # Base case is a single claim.
        if len(claims) == 1:
            key = next(iter(claims))
            claim = claims[key]
            # Check whether the claim is valid.
            with self.solver:
                add_beta_ball(self.solver, claim.term, self.radius)
                if self.solver.check(claim.not_constraint) != z3.unsat:
                    discard.add(key)
                    return proved, discard
            counter["batch_synthesizer.unsat"] += 1
            lemma_forall(self.solver, claim.holes, claim.constraint)
            proved.add(key)
            return proved, discard

        # Check whether any claim is valid, and examine the unsat core.
        with self.solver:
            for key, claim in claims.items():
                add_beta_ball(self.solver, claim.term, self.radius)
                self.solver.assert_and_track(claim.not_constraint, key)
            if self.solver.check() != z3.unsat:
                discard.update(claims)
                return proved, discard
            unsat_core = self.solver.unsat_core()
        maybe_unsat = set(map(str, unsat_core)) & set(claims)
        assert maybe_unsat

        # If a single claim is unsatisfiable, then it is to blame.
        if len(maybe_unsat) == 1:
            counter["batch_synthesizer.unsat_core"] += 1
            key = next(iter(maybe_unsat))
            claim = claims[key]
            lemma_forall(self.solver, claim.holes, claim.constraint)
            proved.add(key)
            return proved, discard

        # Divide suspects and interrogate separately.
        keys = sorted(maybe_unsat)
        x = {k: claims[k] for k in keys[: len(keys) // 2]}
        y = {k: claims[k] for k in keys[len(keys) // 2 :]}
        proved_x, discard_x = self._check(x)
        proved_y, discard_y = self._check(y)
        proved.update(proved_x, proved_y)
        discard.update(discard_x, discard_y)
        return proved, discard
