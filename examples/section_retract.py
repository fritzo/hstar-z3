#!/usr/bin/env python3
"""
Section-retraction synthesis example.

This script continuously runs the synthesis algorithm to generate <r,s> pairs with
s o r [= I, until interrupted by the user.
"""

import argparse

import z3

from hstar.bridge import py_to_z3
from hstar.grammar import ABS, APP, VAR, Term, app
from hstar.solvers import LEQ, solver_timeout
from hstar.synthesis import Synthesizer


def main(args: argparse.Namespace) -> None:
    # Sketch: (\x. x r s) == <r,s>
    sketch = ABS(app(VAR(0), VAR(1), VAR(2)))

    # Constraint: s o r [= I
    def constraint(candidate: Term) -> z3.ExprRef:
        r = APP(candidate, ABS(ABS(VAR(1))))
        s = APP(candidate, ABS(ABS(VAR(0))))
        s_o_r = ABS(APP(s, APP(r, VAR(0))))
        I = ABS(VAR(0))
        return LEQ(py_to_z3(s_o_r), py_to_z3(I))

    synthesizer = Synthesizer(sketch, constraint)
    solver = synthesizer.solver

    print(f"Synthesizing with per-step timeout_ms={args.timeout_ms}")
    solutions: list[Term] = []
    solutions_z3: list[z3.ExprRef] = []
    while True:
        # Find a solution to the constraints.
        candidate, valid = synthesizer.step(timeout_ms=args.timeout_ms)
        if not valid:
            continue

        # Filter to solutions not dominated by previous solutions.
        # FIXME this appears to be ineffectual.
        candidate_z3 = py_to_z3(candidate)
        with solver, solver_timeout(solver, timeout_ms=args.timeout_ms):
            dominated = z3.Or(*(LEQ(candidate_z3, prev) for prev in solutions_z3))
            if solver.check(dominated) == z3.sat:
                print(".", end="", flush=True)
                continue

        # Accept the solution.
        solutions.append(candidate)
        solutions_z3.append(candidate_z3)
        r = APP(candidate, ABS(ABS(VAR(1))))
        s = APP(candidate, ABS(ABS(VAR(0))))
        print(f"Found solution: <{r}, {s}>")


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
