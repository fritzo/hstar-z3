"""
# Synthesis algorithms for λ-join-calculus.

This module provides algorithms for synthesizing λ-join-calculus terms
that satisfy given constraints.
"""

import logging
from collections.abc import Callable

import z3

from . import language
from .enumeration import EnvRefiner, Refiner
from .metrics import COUNTERS
from .normal import Env, Term
from .solvers import find_counterexample, solver_timeout, try_prove
from .theory import add_theory

logger = logging.getLogger(__name__)
counter = COUNTERS[__name__]


class Synthesizer:
    """
    A synthesis algorithm that searches through refinements of a sketch.

    Args:
        sketch: The term sketch to refine.
        constraint: A function that takes a candidate term and returns a Z3
            expression representing a constraint on the candidate.
    """

    def __init__(self, sketch: Term, constraint: Callable[[Term], z3.ExprRef]) -> None:
        self.sketch = sketch
        self.constraint = constraint
        self.refiner = Refiner(sketch)
        self.solver = z3.Solver()
        self.lemmas: list[z3.ExprRef] = []
        add_theory(self.solver)

    def step(self, *, timeout_ms: int = 1000) -> tuple[Term, bool | None]:
        """
        Generate the next candidate and check it.

        Returns:
            A tuple `(candidate, valid)` where `candidate` is the next candidate
            sketch (possibly with holes) and `valid` is a boolean indicating
            whether the candidate satisfies the constraint, or None if the
            validity could not be determined.
        """
        counter["synthesizer.step"] += 1
        candidate = self.refiner.next_candidate()
        logger.debug(f"Checking candidate: {candidate}")
        constraint = self.constraint(candidate)
        holes, constraint = language.hole_closure(constraint)
        not_constraint = z3.Not(constraint)

        # Check if the constraint is unsatisfiable,
        # which means the candidate is invalid.
        if self._is_unsat(constraint, timeout_ms):
            counter["pos.unsat"] += 1
            logger.debug(f"Rejected: {candidate}")
            self.refiner.mark_valid(candidate, False)
            # By DeMorgan's law: ¬∃x.φ(x) ≡ ∀x.¬φ(x)
            self._lemma_forall(holes, not_constraint)
            return candidate, False

        # Check if the negation of the constraint is unsatisfiable,
        # which means the original constraint is valid.
        if self._is_unsat(not_constraint, timeout_ms):
            counter["neg.unsat"] += 1
            logger.debug(f"Found solution: {candidate}")
            self.refiner.mark_valid(candidate, True)
            # By DeMorgan's law: ¬∃x.¬φ(x) ≡ ∀x.φ(x)
            self._lemma_forall(holes, constraint)
            return candidate, True

        # The solver couldn't determine validity within the timeout.
        return candidate, None

    def _is_unsat(self, formula: z3.ExprRef, timeout_ms: int) -> bool:
        # We expect sat to never occur, as our base theory has no finite model,
        # hence we distinguish only between unsat and unknown/sat.
        with solver_timeout(self.solver, timeout_ms=timeout_ms):
            return bool(self.solver.check(formula) == z3.unsat)

    def _lemma_forall(self, holes: list[z3.ExprRef], lemma: z3.ExprRef) -> None:
        counter["lemmas"] += 1
        name = f"lemma_{len(self.lemmas)}"
        if holes:
            counter["lemmas.forall"] += 1
            lemma = z3.ForAll(
                holes,
                lemma,
                patterns=[language.as_pattern(lemma)],
                qid=name,
            )
        self.solver.assert_and_track(lemma, name)
        self.lemmas.append(lemma)


class EnvSynthesizer:
    """
    A synthesis algorithm that searches through refinements of a sketch.

    Args:
        sketch: The environment sketch to refine.
        constraint: A function that takes a candidate env and returns a Z3
            expression representing a constraint on the candidate.
    """

    def __init__(self, sketch: Env, constraint: Callable[[Env], z3.ExprRef]) -> None:
        self.sketch = sketch
        self.constraint = constraint
        self.refiner = EnvRefiner(sketch)
        self.solver = z3.Solver()
        add_theory(self.solver)

    def step(self, *, timeout_ms: int = 1000) -> tuple[Env, bool | None]:
        """
        Generate the next candidate and check it.

        Returns:
            A tuple `(candidate,valid)` where `candidate` is the next candidate
            sketch (possibly with holes) and `valid` is a boolean indicating
            whether the candidate satisfies the constraint
        """
        # TODO update to be closer to Synthesizer.step().
        counter["env_synthesizer.step"] += 1
        candidate = self.refiner.next_candidate()
        logger.debug(f"Checking candidate: {candidate}")
        constraint = self.constraint(candidate)
        valid, _ = try_prove(self.solver, constraint, timeout_ms=timeout_ms)
        if valid is not None:
            self.refiner.mark_valid(candidate, valid)
            self.solver.add(constraint if valid else z3.Not(constraint))
        return candidate, valid


class CEGISSynthesizer:
    """
    A synthesis algorithm that uses Counterexample-Guided Inductive Synthesis
    (CEGIS) to find a term that satisfies a universally quantified constraint.

    Args:
        sketch: The term sketch to refine.
        constraint: A function that takes a candidate term and a symbolic input,
            and returns a Z3 expression representing a constraint that should
            hold for all possible values of the input.
    """

    def __init__(
        self,
        sketch: Term,
        constraint: Callable[[Term, z3.ExprRef], z3.ExprRef],
    ) -> None:
        self.sketch = sketch
        self.constraint = constraint
        self.refiner = Refiner(sketch)
        self.solver = z3.Solver()
        add_theory(self.solver)
        self.counterexamples: list[z3.ExprRef] = []

    def step(self, *, timeout_ms: int = 1000) -> tuple[Term, bool | None]:
        """
        Perform one step of the CEGIS algorithm.

        Returns:
            A tuple (candidate, valid) where candidate is the next candidate
            term and valid indicates whether it satisfies the constraint.
        """
        counter["cegis.step"] += 1

        # Generate next candidate
        candidate = self.refiner.next_candidate()
        logger.debug(f"Generated candidate: {candidate}")

        # Check against counterexamples
        if self.counterexamples:
            with self.solver, solver_timeout(self.solver, timeout_ms=timeout_ms):
                for counterexample in self.counterexamples:
                    # Create constraint for this counterexample
                    constraint_formula = self.constraint(candidate, counterexample)
                    self.solver.add(constraint_formula)

                # Check whether candidate satisfies all counterexample constraints
                result = self.solver.check()
                if result != z3.sat:
                    # This candidate doesn't work with our counterexamples
                    # Mark it as invalid and return
                    self.refiner.mark_valid(candidate, False)
                    return candidate, False

        # Verify candidate against the full specification
        example = z3.Const("example", language.Term)
        formula = z3.ForAll([example], self.constraint(candidate, example))
        valid, counterexample = find_counterexample(
            self.solver, formula, example, timeout_ms=timeout_ms
        )
        if valid is not None:
            self.refiner.mark_valid(candidate, valid)
        if valid is False:
            assert counterexample is not None
            logger.debug(f"Found counterexample: {counterexample}")
            self.counterexamples.append(counterexample)
        elif valid is True and not candidate.free_vars:
            logger.debug(f"Found solution: {candidate}")

        return candidate, valid
