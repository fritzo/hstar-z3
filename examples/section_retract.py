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
from hstar.solvers import CONV, LEQ
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

    # Constraints: s o r [= I, and optionally both s and r converge.
    def constraint(candidate: Term) -> z3.ExprRef:
        r = fst(candidate)
        s = snd(candidate)
        s_o_r = compose(s, r)
        result = LEQ(py_to_z3(s_o_r), py_to_z3(I))
        if args.nontrivial:
            result = z3.And(result, CONV(py_to_z3(r)), CONV(py_to_z3(s)))
        return result

    synthesizer = Synthesizer(sketch, constraint)

    print(f"Synthesizing with per-step timeout_ms={args.timeout_ms}")
    while True:
        candidate, valid = synthesizer.step(timeout_ms=args.timeout_ms)
        if not valid or candidate.free_vars:
            continue
        r = fst(candidate)
        s = snd(candidate)
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
parser.add_argument(
    "--nontrivial",
    action="store_true",
    help="Require both r and s to be convergent",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
