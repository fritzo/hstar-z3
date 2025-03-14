#!/usr/bin/env python3
"""
SIMPLE type constructor synthesis example.

This script attempts to synthesize a finitary definition for the SIMPLE type
constructor defined as: SIMPLE = ⨆ { <r,s> | s ◦ r ⊑ I }
"""

import argparse

import z3

from hstar import normal
from hstar.ast import APP, BOT, TOP, VAR, to_ast
from hstar.bridge import ast_to_nf, nf_to_z3
from hstar.solvers import EQ, SIMPLE
from hstar.synthesis import Synthesizer


def main(args: argparse.Namespace) -> None:
    I = to_ast(lambda x: x)
    Y = to_ast(lambda f: APP(lambda x: f(x(x)), lambda x: f(x(x))))
    DIV = Y(lambda div, x: x | div(x(TOP)))
    raise_ = to_ast(lambda x, _: x)
    lower = to_ast(lambda x: x(TOP))
    pull = to_ast(lambda x, y: x | DIV(y))
    push = to_ast(lambda x: x(BOT))

    # Create a sketch for SIMPLE
    sketch = Y(
        lambda s, f: (
            VAR(0)(f)
            | f(I, I)
            | f(raise_, lower)
            | f(pull, push)
            | s(lambda a, a_: s(lambda b, b_: f(a_ >> b, b_ >> a)))
        )
    )

    # Define a constraint that captures the SIMPLE type definition
    def constraint(candidate: normal.Term) -> z3.ExprRef:
        return EQ(SIMPLE, nf_to_z3(candidate))

    synthesizer = Synthesizer(ast_to_nf(sketch), constraint)

    print(f"Synthesizing SIMPLE type with timeout_ms={args.timeout_ms}")
    while True:
        candidate, valid = synthesizer.step(timeout_ms=args.timeout_ms)
        if not valid or candidate.free_vars:
            continue
        print(f"Potential SIMPLE implementation: {candidate}")


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--timeout-ms",
    type=int,
    default=500,
    help="Timeout for each Z3 invocation in milliseconds",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
