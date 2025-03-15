"""
# Synthesis algorithms for λ-join-calculus.

This module provides algorithms for synthesizing λ-join-calculus terms
that satisfy given constraints.
"""

import logging
from collections.abc import Callable

import z3

from . import solvers
from .enumeration import EnvRefiner, Refiner
from .metrics import COUNTERS
from .normal import Env, Term
from .solvers import add_theory, find_counterexample, try_prove

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
        add_theory(self.solver)

    def step(self, *, timeout_ms: int = 1000) -> tuple[Term, bool | None]:
        """
        Generate the next candidate and check it.

        Returns:
            A tuple `(candidate,valid)` where `candidate` is the next candidate
            sketch (possibly with holes) and `valid` is a boolean indicating
            whether the candidate satisfies the constraint.
        """
        counter["synthesizer.step"] += 1
        candidate = self.refiner.next_candidate()
        logger.debug(f"Checking candidate: {candidate}")
        constraint = self.constraint(candidate)
        valid, _ = try_prove(self.solver, constraint, timeout_ms=timeout_ms)
        if valid is True and not candidate.free_vars:
            logger.info(f"Found solution: {candidate}")
        if valid is not None:
            self.refiner.mark_valid(candidate, valid)
        return candidate, valid


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
        counter["env_synthesizer.step"] += 1
        candidate = self.refiner.next_candidate()
        logger.debug(f"Checking candidate: {candidate}")
        constraint = self.constraint(candidate)
        valid, _ = try_prove(self.solver, constraint, timeout_ms=timeout_ms)
        if valid is not None:
            self.refiner.mark_valid(candidate, valid)
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
            with self.solver:
                for counterexample in self.counterexamples:
                    # Create constraint for this counterexample
                    constraint_formula = self.constraint(candidate, counterexample)
                    self.solver.add(constraint_formula)

                # Check whether candidate satisfies all counterexample constraints
                result = self.solver.check(timeout=timeout_ms)
                if result != z3.sat:
                    # This candidate doesn't work with our counterexamples
                    # Mark it as invalid and return
                    self.refiner.mark_valid(candidate, False)
                    return candidate, False

        # Verify candidate against the full specification
        example = z3.Const("example", solvers.Term)
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
            logger.info(f"Found solution: {candidate}")

        return candidate, valid
