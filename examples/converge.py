#!/usr/bin/env python3
"""
Convergence synthesis example.

This script continuously runs the synthesis algorithm to generate closed terms
that are provably convergent, until interrupted by the user.
"""

import argparse
import logging

import z3

from hstar.bridge import nf_to_z3
from hstar.language import CONV
from hstar.logging import setup_color_logging
from hstar.normal import VAR, Term
from hstar.synthesis import Synthesizer

logger = logging.getLogger(__name__)
setup_color_logging(level=logging.DEBUG)


def main(args: argparse.Namespace) -> None:
    # A trivial sketch.
    sketch = VAR(0)

    # Constraint: The term must converge
    def constraint(candidate: Term) -> z3.ExprRef:
        return CONV(nf_to_z3(candidate))

    def on_fact(term: Term, valid: bool) -> None:
        if valid:
            logger.info(f"Found convergent term: {term}")

    synthesizer = Synthesizer(
        sketch,
        constraint,
        on_fact,
        timeout_ms=args.timeout_ms,
    )

    logger.info(f"Synthesizing convergent terms with timeout_ms={args.timeout_ms}")
    for _ in range(args.steps):
        synthesizer.step()


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--timeout-ms",
    type=int,
    default=1000,
    help="Timeout for each Z3 invocation in milliseconds",
)
parser.add_argument(
    "--steps",
    type=int,
    default=1_000_000_000,
    help="Number of synthesis steps to run before stopping",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
