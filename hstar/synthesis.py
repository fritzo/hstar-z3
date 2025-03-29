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
from .enumeration import Refiner
from .grammars import Grammar
from .metrics import COUNTERS
from .normal import Term
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
        candidate, valid = self.refiner.next_candidate()
        if valid is not None:
            # Candidate merely specializes a previous candidate.
            return candidate, valid
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
class Claim:
    term: Term
    holes: list[z3.ExprRef]
    constraint: z3.ExprRef
    not_constraint: z3.ExprRef
    valid: bool | None = None

    def negate(self) -> "Claim":
        return Claim(
            term=self.term,
            holes=self.holes,
            constraint=self.not_constraint,
            not_constraint=self.constraint,
            valid=None,
        )


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
        self._pending: tuple[dict[str, Claim], dict[str, Claim]] = {}, {}
        self._new_facts: dict[Term, bool] = {}

    def step(self) -> tuple[Term, bool]:
        """Performs inference until a new fact is learned."""
        counter["batch_synthesizer.step"] += 1
        while not self._new_facts:
            for valid in (False, True):
                while max(map(len, self._pending)) < self.batch_size:
                    self._add_candidate()
                    if self._new_facts:
                        return self._new_facts.popitem()
                if self._pending[valid]:
                    self._step(valid)
        return self._new_facts.popitem()

    def _add_candidate(self) -> None:
        counter["batch_synthesizer.add_candidate"] += 1
        candidate, valid = self.refiner.next_candidate()
        if valid is not None:
            # Candidate merely specializes a previous candidate.
            counter["batch_synthesizer.specialize"] += 1
            action = "Accepted" if valid else "Rejected"
            logger.debug(f"{action}: {candidate}")
            self._new_facts[candidate] = valid
            return
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
            claim.valid = valid
            action = "Accepted" if valid else "Rejected"
            logger.debug(f"{action}: {claim.term}")
            self.refiner.mark_valid(claim.term, valid)
            self._new_facts[claim.term] = valid

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
                self.solver.assert_and_track(claim.not_constraint, key)
            if self.solver.check() != z3.unsat:
                discard.update(claims.keys())
                return proved, discard
            unsat_core = self.solver.unsat_core()
        maybe_unsat = set(map(str, unsat_core)) & set(claims.keys())
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
