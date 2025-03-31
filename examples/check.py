#!/usr/bin/env python3
"""
Z3 Theory Soundness Checker: runs indefinitely and prints unsat core on error.
"""

import argparse
import logging

import z3

from hstar.ast import APP, BOT, TOP, to_ast
from hstar.bridge import ast_to_z3
from hstar.language import SIMPLE
from hstar.logging import setup_color_logging
from hstar.theory import add_theory

logger = logging.getLogger(__name__)


def add_simple_def(solver: z3.Solver) -> None:
    I = to_ast(lambda x: x)
    Y = to_ast(lambda f: APP(lambda x: f(x(x)), lambda x: f(x(x))))
    DIV = Y(lambda div, x: x | div(x, TOP))
    raise_ = to_ast(lambda x, _: x)
    lower = to_ast(lambda x: x(TOP))
    pull = to_ast(lambda x, y: x | DIV(y))
    push = to_ast(lambda x: x(BOT))
    A = Y(
        lambda s, f: (
            f(I, I)
            | f(raise_, lower)
            | f(pull, push)
            | s(lambda a, a_: s(lambda b, b_: f(a_ >> b, b_ >> a)))
        )
    )
    solver.assert_and_track(SIMPLE == ast_to_z3(A), "simple_def")


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
    help="Add the conjectured SIMPLE definition as an axiom",
)

if __name__ == "__main__":
    setup_color_logging(level=logging.INFO)
    args = parser.parse_args()
    main(args)
