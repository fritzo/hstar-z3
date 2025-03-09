#!/usr/bin/env python3
"""
Section-retraction synthesis example.

This script continuously runs the synthesis algorithm to generate <r,s> pairs with
s o r [= I, until interrupted by the user.
"""

import argparse

import z3

from hstar.bridge import py_to_z3
from hstar.grammar import ABS, APP, VAR, Term, shift
from hstar.solvers import LEQ, solver_timeout
from hstar.synthesis import Synthesizer


def main(args: argparse.Namespace) -> None:
    # Basic lambda calculus definitions.
    I = ABS(VAR(0))

    def compose(f: Term, g: Term) -> Term:
        return ABS(APP(f, APP(g, VAR(0))))

    def pair(*args: Term) -> Term:
        result = VAR(0)
        for arg in args:
            result = APP(result, shift(arg))
        return ABS(result)

    def fst(pair: Term) -> Term:
        return APP(pair, ABS(ABS(VAR(1))))

    def snd(pair: Term) -> Term:
        return APP(pair, ABS(ABS(VAR(0))))

    # Sketch: (\x. x r s) == <r,s>
    sketch = pair(VAR(0), VAR(1))

    # Constraint: s o r [= I
    def constraint(candidate: Term) -> z3.ExprRef:
        r = fst(candidate)
        s = snd(candidate)
        s_o_r = compose(s, r)
        return LEQ(py_to_z3(s_o_r), py_to_z3(I))

    synthesizer = Synthesizer(sketch, constraint)
    solver = synthesizer.solver

    print(f"Synthesizing with per-step timeout_ms={args.timeout_ms}")
    solutions: list[Term] = []
    while True:
        # Find a solution to the constraints.
        candidate, valid = synthesizer.step(timeout_ms=args.timeout_ms)
        if not valid:
            continue

        # Filter to solutions not dominated by previous solutions.
        # FIXME this appears to be ineffectual.
        r = fst(candidate)
        s = snd(candidate)
        with solver, solver_timeout(solver, timeout_ms=args.timeout_ms):
            clauses: list[z3.ExprRef] = []
            for prev in solutions:
                r_leq = LEQ(py_to_z3(r), py_to_z3(fst(prev)))
                s_leq = LEQ(py_to_z3(s), py_to_z3(snd(prev)))
                clauses.append(z3.And(r_leq, s_leq))
            dominated = z3.Or(*clauses)
            if solver.check(dominated) == z3.sat:
                print(".", end="", flush=True)
                continue

        # Accept the solution.
        print(f"Found solution: <{r}, {s}>")
        solutions.append(candidate)


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
