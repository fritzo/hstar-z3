#!/usr/bin/env python3
"""
Section-retraction synthesis example.

This script continuously runs the synthesis algorithm to generate <r,s> pairs with
r o s [= I, until interrupted by the user.
"""

import argparse
import logging

import z3

from hstar.bridge import nf_to_z3
from hstar.language import CONV, LEQ
from hstar.logging import setup_color_logging
from hstar.normal import ABS, APP, VAR, Term, shift
from hstar.synthesis import BatchingSynthesizer, Synthesizer, SynthesizerBase

logger = logging.getLogger(__name__)
setup_color_logging(level=logging.DEBUG)


def main(args: argparse.Namespace) -> None:
    # Basic lambda calculus definitions.
    I = ABS(VAR(0))

    def compose(f: Term, g: Term) -> Term:
        f = shift(f)
        g = shift(g)
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

    # Constraints: r o s [= I, and optionally both r and s converge.
    def constraint(candidate: Term) -> z3.ExprRef:
        r = fst(candidate)
        s = snd(candidate)
        r_o_s = compose(r, s)
        result = LEQ(nf_to_z3(r_o_s), nf_to_z3(I))
        if args.nontrivial:
            result = z3.And(result, CONV(nf_to_z3(r)), CONV(nf_to_z3(s)))
        return result

    def on_fact(term: Term, valid: bool) -> None:
        if valid:
            r = fst(term)
            s = snd(term)
            logger.info(f"Found solution: <{r}, {s}>")

    synthesizer: SynthesizerBase
    if args.batch_size == 1:
        synthesizer = Synthesizer(
            sketch,
            constraint,
            on_fact,
            timeout_ms=args.timeout_ms,
        )
    else:
        synthesizer = BatchingSynthesizer(
            sketch,
            constraint,
            on_fact,
            timeout_ms=args.timeout_ms,
            batch_size=args.batch_size,
        )

    logger.info(f"Synthesizing with timeout_ms={args.timeout_ms}")
    for _ in range(args.steps):
        synthesizer.step()

    solutions = sorted(synthesizer.refiner.most_general_solutions())
    solutions_str = "\n".join(f"<{fst(pair)}, {snd(pair)}>" for pair in solutions)
    logger.info(f"Found {len(solutions)} general solutions:\n{solutions_str}")


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
parser.add_argument(
    "--steps", type=int, default=1_000_000_000, help="Maximum number of synthesis steps"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="Synthesizer batch size",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
