#!/usr/bin/env python3
"""
Simple type constructor synthesis example.

This script attempts to synthesize a finitary definition for the simple type
constructor defined as: A = ⨆ { <r,s> | r ◦ s ⊑ I }.
"""

import argparse
import logging

import z3

from hstar import normal
from hstar.ast import APP, BOT, TOP, VAR, to_ast
from hstar.bridge import ast_to_nf, nf_to_z3
from hstar.language import A
from hstar.logging import setup_color_logging
from hstar.synthesis import Synthesizer

logger = logging.getLogger(__name__)
setup_color_logging(level=logging.DEBUG)


def main(args: argparse.Namespace) -> None:
    I = to_ast(lambda x: x)
    Y = to_ast(lambda f: APP(lambda x: f(x(x)), lambda x: f(x(x))))
    DIV = Y(lambda div, x: x | div(x, TOP))
    raise_ = to_ast(lambda x, _: x)
    lower = to_ast(lambda x: x(TOP))
    pull = to_ast(lambda x, y: x | DIV(y))
    push = to_ast(lambda x: x(BOT))

    # Create a sketch for A
    sketch = Y(
        lambda s, f: (
            VAR(0)(f)
            | f(I, I)
            | f(lower, raise_)
            | f(push, pull)
            | s(lambda a, a_: s(lambda b, b_: f(b >> a_, a >> b_)))
        )
    )

    # Define a constraint that captures the simple type definition
    def constraint(candidate: normal.Term) -> z3.ExprRef:
        return A == nf_to_z3(candidate)

    def on_fact(term: normal.Term, valid: bool) -> None:
        if valid:
            logger.info(f"Found a simple type: {term}")

    synthesizer = Synthesizer(
        ast_to_nf(sketch),
        constraint,
        on_fact,
        timeout_ms=args.timeout_ms,
    )

    logger.info(f"Synthesizing simple type with timeout_ms={args.timeout_ms}")
    for _ in range(args.steps):
        synthesizer.step()


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--timeout-ms",
    type=int,
    default=500,
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
