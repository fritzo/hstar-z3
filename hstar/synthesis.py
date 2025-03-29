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
    solver.assert_and_track(lemma, name)
    lemmas.append(lemma)


class SynthesizerBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self) -> tuple[Term, bool | None]:
        """Generate the next candidate and check it."""


class Synthesizer(SynthesizerBase):
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
            lemma_forall(self.solver, holes, not_constraint)
            return candidate, False

        # Check if the negation of the constraint is unsatisfiable,
        # which means the original constraint is valid.
        if self._is_unsat(not_constraint):
            counter["neg.unsat"] += 1
            logger.debug(f"Found solution: {candidate}")
            self.refiner.mark_valid(candidate, True)
            # By DeMorgan's law: ¬∃x.¬φ(x) ≡ ∀x.φ(x)
            lemma_forall(self.solver, holes, constraint)
            return candidate, True

        # The solver couldn't determine validity within the timeout.
        return candidate, None

    def _is_unsat(self, formula: z3.ExprRef) -> bool:
        # We expect sat to never occur, as our base theory has no finite model,
        # hence we distinguish only between unsat and unknown/sat.
        return bool(self.solver.check(formula) == z3.unsat)


@dataclass(slots=True)
class Candidate:
    term: Term
    holes: list[z3.ExprRef]
    constraint: z3.ExprRef
    not_constraint: z3.ExprRef
    valid: bool | None = None


class BatchingSynthesizer(SynthesizerBase):
    def __init__(
        self,
        sketch: Term,
        constraint: Callable[[Term], z3.ExprRef],
        *,
        batch_size: int = 1,
        timeout_ms: int = 1000,
    ) -> None:
        self.sketch = sketch
        self.constraint = constraint
        self.batch_size = batch_size
        self.refiner = Refiner(sketch)
        self.solver = z3.Solver()
        self.solver.set(timeout=timeout_ms)
        self.solver.set("unsat_core", True)
        self.solver.timeout_ms = timeout_ms
        add_theory(self.solver)

        # Candidates remain pending until either they are decided or hit a timeout.
        self._next_id = 0
        self._pending_valid: dict[str, Candidate] = {}
        self._pending_invalid: dict[str, Candidate] = {}
        self._new_solutions: list[Candidate] = []

    def step(self) -> tuple[Term, bool]:
        """Performs work to evaluate a single candidate."""
        while not self._new_solutions:
            self._fill()
            if self._pending_invalid:
                self._step_invalid()
            self._fill()
            if self._pending_valid:
                self._step_valid()
        c = self._new_solutions.pop()
        assert c.valid is not None
        return c.term, c.valid

    @property
    def _pending_size(self) -> int:
        return max(len(self._pending_valid), len(self._pending_invalid))

    def _fill(self) -> None:
        while self._pending_size < self.batch_size:
            candidate = self.refiner.next_candidate()
            logger.debug(f"Proposing candidate: {candidate}")
            constraint = self.constraint(candidate)
            holes, constraint = language.hole_closure(constraint)
            not_constraint = Not(constraint)
            key = f"candidate_{self._next_id}"
            self._next_id += 1
            c = Candidate(
                term=candidate,
                holes=holes,
                constraint=constraint,
                not_constraint=not_constraint,
            )
            self._pending_invalid[key] = c
            self._pending_valid[key] = c

    def _step_invalid(self) -> None:
        counter["batch_synthesizer.invalid"] += 1
        pending = self._pending_invalid
        keys = sorted(pending.keys())
        constraints = {key: pending[key].constraint for key in keys}

        # Check invalid candidates.
        logger.debug("Checking for invalid candidates")
        rejected, discard = self._check(constraints)
        for key in discard:
            counter["pos.unknown"] += 1
            pending.pop(key)
        for key in rejected:
            counter["pos.unsat"] += 1
            c = pending.pop(key)
            self._pending_valid.pop(key, None)
            c.valid = False
            logger.debug(f"Rejected: {c.term}")
            self.refiner.mark_valid(c.term, False)
            lemma_forall(self.solver, c.holes, c.not_constraint)
            self._new_solutions.append(c)

    def _step_valid(self) -> None:
        counter["batch_synthesizer.step_valid"] += 1
        pending = self._pending_valid
        keys = sorted(pending.keys())
        not_constraints = {key: pending[key].not_constraint for key in keys}

        # Check valid candidates.
        logger.debug("Checking for valid candidates")
        accepted, discard = self._check(not_constraints)
        for key in discard:
            counter["neg.unknown"] += 1
            pending.pop(key)
        for key in accepted:
            counter["neg.unsat"] += 1
            c = pending.pop(key)
            self._pending_invalid.pop(key, None)
            c.valid = True
            logger.debug(f"Accepted: {c.term}")
            self.refiner.mark_valid(c.term, True)
            lemma_forall(self.solver, c.holes, c.constraint)
            self._new_solutions.append(c)

    def _check(self, formulas: dict[str, z3.ExprRef]) -> tuple[set[str], set[str]]:
        """
        Returns (unsat,discard) formulas. Remaining formulas should be retried.
        """
        # Assume the query is convex, i.e. if any subset of the formulas is
        # unsatisfiable, then at least one of the formulas is unsatisfiable.
        unsat: set[str] = set()
        discard: set[str] = set()

        # Base case is a single formula.
        if len(formulas) == 1:
            # Check whether the single formula is unsatisfiable.
            if self.solver.check(*formulas.values()) != z3.unsat:
                discard.update(formulas)
                return unsat, discard
            unsat.update(formulas)
            return unsat, discard

        # Check whether any formula is unsatisfiable, and examine the unsat core.
        with self.solver:
            for name, formula in formulas.items():
                self.solver.assert_and_track(formula, name)
            if self.solver.check() != z3.unsat:
                discard.update(formulas)
                return unsat, discard
            unsat_core = self.solver.unsat_core()
        maybe_unsat = set(map(str, unsat_core)) & set(formulas)
        assert maybe_unsat

        # If a single formula is unsatisfiable, then it is to blame.
        if len(maybe_unsat) == 1:
            unsat.update(maybe_unsat)
            return unsat, discard

        # Recurse.
        keys = sorted(maybe_unsat)
        x = {k: formulas[k] for k in keys[: len(keys) // 2]}
        y = {k: formulas[k] for k in keys[len(keys) // 2 :]}
        unsat_x, discard_x = self._check(x)
        # TODO move lemma_forall here, rather than later
        unsat_y, discard_y = self._check(y)
        unsat.update(unsat_x, unsat_y)
        discard.update(discard_x, discard_y)
        return unsat, discard


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
