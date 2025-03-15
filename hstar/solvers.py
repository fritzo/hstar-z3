"""
# Solver tools wrapping Z3.
"""

from collections.abc import Generator
from contextlib import contextmanager

import z3

from .metrics import COUNTERS

counter = COUNTERS[__name__]

# https://microsoft.github.io/z3guide/programming/Parameters/#global-parameters
DEFAULT_TIMEOUT_MS = 4294967295


@contextmanager
def solver_timeout(
    solver: z3.Solver, *, timeout_ms: int
) -> Generator[None, None, None]:
    """Context manager to set a timeout on a Z3 solver."""
    # This works around latch of .get() interface by patching the solver object
    # with a .timeout_ms attribute.
    old_timeout_ms = getattr(solver, "timeout_ms", DEFAULT_TIMEOUT_MS)
    solver.set(timeout=timeout_ms)
    solver.timeout_ms = timeout_ms
    try:
        yield
    finally:
        solver.set(timeout=old_timeout_ms)
        solver.timeout_ms = old_timeout_ms


def try_prove(
    solver: z3.Solver, formula: z3.ExprRef, *, timeout_ms: int = 1000
) -> tuple[bool | None, str | None]:
    """
    Try to prove a formula is valid or invalid.

    Args:
        solver: Z3 solver to use
        formula: Formula to check validity of
        timeout_seconds: Maximum time (in seconds) to spend on the check

    Returns:
        Tuple of:
        - True if formula proved valid
        - False if formula proved invalid
        - None if formula is satisfiable but not valid
        And the counterexample model string (if formula is not valid)
    """
    counter["try_prove"] += 1
    with solver, solver_timeout(solver, timeout_ms=timeout_ms):
        solver.add(z3.Not(formula))
        result = solver.check()
        if result == z3.unsat:
            return True, None
        if result == z3.sat:
            model = solver.model()
            assert model is not None, "Got sat result but no model!"
            # Format model while still in context
            model_str = "\n".join(f"{d} = {model[d]}" for d in model.decls())
            return False, model_str
        if result == z3.unknown:
            return None, None
        raise ValueError(f"Z3 returned unexpected result: {result}")


def find_counterexample(
    solver: z3.Solver,
    formula: z3.ExprRef,
    input_var: z3.ExprRef,
    *,
    timeout_ms: int = 1000,
) -> tuple[bool | None, z3.ExprRef | None]:
    """
    Try to prove a formula is valid. If it is not valid,
    return a counterexample for the input variable.

    Args:
        solver: Z3 solver to use
        formula: Formula to check validity of (should be a ForAll quantified formula)
        input_var: The input variable to extract a counterexample for
        timeout_ms: Maximum time to spend on the check

    Returns:
        Tuple of:
        - True if formula proved valid
        - False if formula proved invalid
        - None if unknown
        And the counterexample value (if formula is invalid)
    """
    counter["find_counterexample"] += 1
    with solver, solver_timeout(solver, timeout_ms=timeout_ms):
        # Negate the formula to find a counterexample
        solver.add(z3.Not(formula))
        result = solver.check()

        if result == z3.unsat:
            # Formula is valid (no counterexample exists)
            return True, None
        elif result == z3.sat:
            # Formula is invalid, extract counterexample
            model = solver.model()
            assert model is not None, "Got sat result but no model!"

            # Find the value of the input variable in the model
            # This is the counterexample
            for d in model.decls():
                if d.name() == input_var.decl().name():
                    counterexample = model[d]
                    return False, counterexample
            raise ValueError("Input variable not found in model!")
        else:
            assert result == z3.unknown
            return None, None
