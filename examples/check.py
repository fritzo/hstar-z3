#!/usr/bin/env python3
"""
Z3 Theory Soundness Checker: runs indefinitely and prints unsat core on error.
"""

import logging

import z3

from hstar.logging import setup_color_logging
from hstar.theory import add_theory

logger = logging.getLogger(__name__)


def main() -> None:
    # Set up solver
    solver = z3.Solver()
    solver.set("proof", True)  # Enable proof generation

    # Add theory
    logger.info("Adding theory...")
    add_theory(solver, unsat_core=True)

    # Check soundness
    logger.info("Checking soundness...")
    result = solver.check()
    if result == z3.unsat:
        logger.error(f"Unsat core:\n{solver.unsat_core()}")
    elif result == z3.sat:
        logger.info("Satisfiable")
    else:
        logger.warning("Satisfiability unknown")


if __name__ == "__main__":
    setup_color_logging(level=logging.INFO)
    main()
