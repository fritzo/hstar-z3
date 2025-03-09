#!/usr/bin/env python3
"""
Convergence synthesis example.

This script continuously runs the synthesis algorithm to generate closed terms
that are provably convergent, until interrupted by the user.
"""

import argparse

import z3

from hstar.bridge import py_to_z3
from hstar.grammar import VAR, Term
from hstar.solvers import CONV
from hstar.synthesis import Synthesizer


def main(args: argparse.Namespace) -> None:
    # A trivial sketch.
    sketch = VAR(0)
    
    # Constraint: The term must converge
    def constraint(candidate: Term) -> z3.ExprRef:
        return CONV(py_to_z3(candidate))
    
    synthesizer = Synthesizer(sketch, constraint)
    
    print(f"Synthesizing convergent terms with per-step timeout_ms={args.timeout_ms}")
    while True:
        candidate, valid = synthesizer.step(timeout_ms=args.timeout_ms)
        if not valid or candidate.free_vars:
            continue
        
        print(f"Found convergent term: {candidate}")


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--timeout-ms",
    type=int,
    default=100,
    help="Timeout for each Z3 invocation in milliseconds",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args) 