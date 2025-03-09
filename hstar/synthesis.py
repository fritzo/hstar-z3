"""
# Synthesis algorithms for λ-join-calculus.

This module provides algorithms for synthesizing λ-join-calculus terms
that satisfy given constraints.
"""

import logging
from collections.abc import Callable

import z3

from .enumeration import EnvRefiner, Refiner
from .grammar import Env, Term
from .metrics import COUNTERS
from .solvers import add_theory, try_prove

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
        """Generate the next candidate and check it."""
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
        """Generate the next candidate and check it."""
        counter["env_synthesizer.step"] += 1
        candidate = self.refiner.next_candidate()
        logger.debug(f"Checking candidate: {candidate}")
        constraint = self.constraint(candidate)
        valid, _ = try_prove(self.solver, constraint, timeout_ms=timeout_ms)
        if valid is not None:
            self.refiner.mark_valid(candidate, valid)
        return candidate, valid
