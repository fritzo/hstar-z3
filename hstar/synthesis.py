"""
# Synthesis algorithms for λ-join-calculus.

This module provides algorithms for synthesizing λ-join-calculus terms
that satisfy given constraints.
"""

import logging
from collections.abc import Callable

import z3
from z3 import ForAll, Not

from . import language
from .enumeration import EnvRefiner, Refiner
from .grammars import Grammar
from .metrics import COUNTERS
from .normal import Env, Term
from .solvers import try_prove
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

    def __init__(
        self,
        sketch: Term,
        constraint: Callable[[Term], z3.ExprRef],
        *,
        timeout_ms: int = 1000,
    ) -> None:
        self.sketch = sketch
        self.constraint = constraint
        self.refiner = Refiner(sketch)
        self.solver = z3.Solver()
        self.solver.set(timeout=timeout_ms)
        self.solver.timeout_ms = timeout_ms  # for hstar use only
        self.lemmas: list[z3.ExprRef] = []
        add_theory(self.solver)

    def step(self) -> tuple[Term, bool | None]:
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
        not_constraint = Not(constraint)

        # Check if the constraint is unsatisfiable,
        # which means the candidate is invalid.
        if self._is_unsat(constraint):
            counter["pos.unsat"] += 1
            logger.debug(f"Rejected: {candidate}")
            self.refiner.mark_valid(candidate, False)
            # By DeMorgan's law: ¬∃x.φ(x) ≡ ∀x.¬φ(x)
            self._lemma_forall(holes, not_constraint)
            return candidate, False

        # Check if the negation of the constraint is unsatisfiable,
        # which means the original constraint is valid.
        if self._is_unsat(not_constraint):
            counter["neg.unsat"] += 1
            logger.debug(f"Found solution: {candidate}")
            self.refiner.mark_valid(candidate, True)
            # By DeMorgan's law: ¬∃x.¬φ(x) ≡ ∀x.φ(x)
            self._lemma_forall(holes, constraint)
            return candidate, True

        # The solver couldn't determine validity within the timeout.
        return candidate, None

    def _is_unsat(self, formula: z3.ExprRef) -> bool:
        # We expect sat to never occur, as our base theory has no finite model,
        # hence we distinguish only between unsat and unknown/sat.
        return bool(self.solver.check(formula) == z3.unsat)

    def _lemma_forall(self, holes: list[z3.ExprRef], lemma: z3.ExprRef) -> None:
        counter["lemmas"] += 1
        name = f"lemma_{len(self.lemmas)}"
        if holes:
            counter["lemmas.forall"] += 1
            lemma = ForAll(
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
            self.solver.add(constraint if valid else Not(constraint))
        return candidate, valid


# FIXME the correct instance may not be the last instance.
_INSTANCE: list[z3.ExprRef] = []


def _on_clause(pr: z3.ExprRef, clause: list, parents: list) -> None:
    global _INSTANCE
    if not z3.is_app(pr) or pr.decl().name() != "inst":
        return
    quant = pr.arg(0)
    if quant.qid() != "sygus":
        return
    for child in pr.children():
        if not z3.is_app(child) or child.decl().name() != "bind":
            continue
        _INSTANCE = child.children()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Instance: {_INSTANCE}")
        break


def sygus(
    grammar: Grammar,
    constraint: Callable[[z3.ExprRef], z3.ExprRef],
) -> z3.ExprRef | None:
    """
    EXPERIMENTAL Synthesize using SyGuS entirely inside Z3.

    FIXME this does not work yet.

    Args:
        grammar: The grammar to synthesize terms from.
        constraint: A function : Term -> BoolSort.
        solver: The Z3 solver to use.
    Returns:
        A term of grammar.sort satisfying the constraint,
            or None if no term was found.
    """
    # This attempts to extract an example instance from Z3's proof of
    # unsatisfiability. It currently fails for at least two reasons:
    # 1. The instance is not guaranteed to be the correct one.
    #    This might be addressed by recording many instances and filtering.
    # 2. The desired "sygus" quantifier is not instantiated often enough.
    #    This might be addressed by weighting other quantifiers or
    #    implementing a custom tactic to instantiate the quantifier.
    global _INSTANCE
    hole = z3.FreshConst(grammar.sort, "hole")
    formula = ForAll([hole], Not(constraint(grammar.eval(hole))), qid="sygus")
    solver = z3.Solver()
    add_theory(solver)
    logger.info("Synthesizing")
    solver.add(*grammar.eval_theory)
    z3.OnClause(solver, _on_clause)
    _INSTANCE = []
    result = solver.check(formula)
    if result != z3.unsat:
        return None
    if not _INSTANCE:
        return None
    logger.info(f"Found instance: {_INSTANCE[0]}")
    return _INSTANCE[0]
