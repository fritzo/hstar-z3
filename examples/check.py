#!/usr/bin/env python3
"""
Z3 Theory Soundness Checker: runs indefinitely and prints unsat core on error.
"""

import argparse
import logging

import z3

from hstar.bridge import nf_to_z3
from hstar.language import A
from hstar.logging import setup_color_logging
from hstar.theory import add_beta_ball, add_theory, simple_definition

logger = logging.getLogger(__name__)


def add_simple_def(solver: z3.Solver) -> None:
    A_def, _, _ = simple_definition()
    add_beta_ball(solver, A_def, radius=3)
    solver.assert_and_track(A == nf_to_z3(A_def), "simple_def")


def main(args: argparse.Namespace) -> None:
    # Set up solver
    solver = z3.Solver()
    solver.set("proof", True)  # Enable proof generation

    # Add theory
    logger.info("Adding theory...")
    add_theory(solver, unsat_core=True)
    if args.simple:
        add_simple_def(solver)

    # Check soundness
    logger.info("Checking soundness...")
    result = solver.check()
    if result == z3.unsat:
        logger.error(f"Unsat core:\n{solver.unsat_core()}")
    elif result == z3.sat:
        logger.info("Satisfiable")
    else:
        logger.warning("Satisfiability unknown")


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--simple",
    action="store_true",
    help="Add the conjectured simple type constructor definition as an axiom",
)

if __name__ == "__main__":
    setup_color_logging(level=logging.INFO)
    args = parser.parse_args()
    main(args)
