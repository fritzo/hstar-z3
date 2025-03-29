#!/usr/bin/env python3
"""
Type inhabitant synthesis example.

This script continuously runs the synthesis algorithm to generate terms that
inhabit a specified type, until interrupted by the user.
"""

import argparse

import z3

from hstar.bridge import nf_to_z3
from hstar.language import OFTYPE, bool_, boool, pair, pre_pair, semi, unit
from hstar.normal import VAR, Term
from hstar.synthesis import Synthesizer

TYPES = {
    "semi": semi,
    "boool": boool,
    "pre_pair": pre_pair,
    "unit": unit,
    "bool": bool_,
    "pair": pair,
}


def main(args: argparse.Namespace) -> None:
    if args.type not in TYPES:
        valid_types = ", ".join(sorted(TYPES.keys()))
        raise ValueError(f"Unknown type: {args.type}. Valid types are: {valid_types}")
    target_type = TYPES[args.type]

    # Create a sketch for a closed term
    sketch = VAR(0)

    # Constraint: term must be of the specified type
    def constraint(candidate: Term) -> z3.ExprRef:
        return OFTYPE(nf_to_z3(candidate), target_type)

    def on_fact(term: Term, valid: bool) -> None:
        if valid:
            print(f"Found inhabitant: {term}")

    synthesizer = Synthesizer(
        sketch,
        constraint,
        on_fact,
        timeout_ms=args.timeout_ms,
    )

    print(f"Finding inhabitants of type '{args.type}'")
    while True:
        synthesizer.step()


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
    "--type",
    type=str,
    default="bool",
    help="Type to synthesize inhabitants for (semi, boool, pre_pair, unit, bool, pair)",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
